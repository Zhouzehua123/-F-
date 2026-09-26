# -*- coding: utf-8 -*-
# 本补充程序于 2026-09-27 使用 OpenAI Codex 辅助整理及核验。
# 仅增加审稿所需的独立诊断；不运行原求解入口、不修改原参数或原结果。
"""Q2/Q3 的评分尺度、质量指数与配比收益敏感性。

从任意工作目录运行：
  python q23_尺度与结构敏感性.py --data-dir /path/to/real_attachments
可用 --output-dir 指定独立输出目录；默认写入本脚本旁的 q23_结果。
依赖：numpy、pandas、scipy。附件可分层或平铺，不依赖 _common.py。

gamma=1 仅作六参数受限拟合；不重选主模型、不重跑完整预算扫描。
Q 剖面在少量预算处显式比较端点和网格定位后的局部极小，不声明全局证明。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
K = 6.0 + 2e-4 * 2048.0
PARAMETER_NAMES = ('E', 'A', 'alpha', 'B', 'beta', 'C', 'gamma')


@dataclass(frozen=True)
class Parameters:
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    C: float
    gamma: float

    def vector(self):
        return np.array([getattr(self, name) for name in PARAMETER_NAMES])


def predict(p, n, d, q):
    return p.E + p.A * np.asarray(n) ** -p.alpha + p.B * np.asarray(d) ** -p.beta + p.C * (1 - np.asarray(q)) ** p.gamma


def quality_cost(q):
    return 5e9 * q ** 4


def fixed_quality(p, budget, q, q0):
    """消元 D 后在 N 方向求解，并保留原 N/D 搜索边界。N、D 输出为 B。"""
    if not 0 <= q0 <= q <= 1:
        raise ValueError('需要 0 <= Q0 <= Q <= 1')
    s = (quality_cost(q) - quality_cost(q0)) / 1e9
    cb = budget / 1e18
    n_lo = max(1e-4, (cb / 1e7 - s) / K)
    n_hi = min(1e6, (cb / 1e-3 - s) / K)
    if n_hi <= n_lo:
        raise ValueError('给定质量下无可行规模区间')

    def data(n):
        return cb / (K * n + s)

    def derivative(log_n):
        n = np.exp(log_n)
        d = data(n)
        return -p.alpha * p.A * n ** -p.alpha + p.beta * p.B * d ** -p.beta * K * n / (K * n + s)

    lo, hi = np.log(n_lo), np.log(n_hi)
    if derivative(lo) >= 0:
        log_n = lo
    elif derivative(hi) <= 0:
        log_n = hi
    else:
        log_n = brentq(derivative, lo, hi, xtol=1e-13)
    n = float(np.exp(log_n))
    d = float(data(n))
    residual = (d * (K * n + s) - cb) / cb
    if not (1e-4 * (1-1e-10) <= n <= 1e6 * (1+1e-10) and 1e-3 * (1-1e-10) <= d <= 1e7 * (1+1e-10)):
        raise RuntimeError('规模边界核验失败')
    if abs(residual) > 1e-10:
        raise RuntimeError('预算核验失败')
    return dict(C_budget=float(budget), N_B=n, D_B=d, Q=float(q), L=float(predict(p, n, d, q)), budget_relative_residual=float(residual))


def quality_profile(p, budget, q0):
    grid = np.unique(np.r_[np.linspace(q0, 1, 121), 1 - 10. ** np.arange(-12., -1.)])
    grid = grid[grid >= q0]
    values = [fixed_quality(p, budget, q, q0)['L'] for q in grid]
    candidates = [fixed_quality(p, budget, q0, q0), fixed_quality(p, budget, 1., q0)]
    for i in range(1, len(grid) - 1):
        if values[i] < values[i-1] and values[i] < values[i+1]:
            result = minimize_scalar(lambda q: fixed_quality(p, budget, q, q0)['L'], bounds=(grid[i-1], grid[i+1]), method='bounded', options={'xatol': 1e-13})
            if not result.success:
                raise RuntimeError('质量剖面局部细化失败')
            candidates.append(fixed_quality(p, budget, float(result.x), q0))
    return min(candidates, key=lambda x: x['L']), candidates


def robust_objective(residual):
    return float(np.sum(.05**2 * (np.sqrt(1 + (residual / .05)**2) - 1)))


def errors(residual):
    return dict(RMSE=float(np.sqrt(np.mean(residual**2))), MAE=float(np.mean(np.abs(residual))))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attachment(data_dir, name):
    matches = sorted(data_dir.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f'{name} 应有且仅有一个文件，实际匹配 {len(matches)} 个')
    return matches[0]


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('MODELING_DATA_DIR'), help='附件根目录；也可设置 MODELING_DATA_DIR')
    parser.add_argument('--output-dir', type=Path, default=HERE/'q23_结果', help='独立补充结果目录，不得覆盖原求解结果')
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error('请用 --data-dir 或 MODELING_DATA_DIR 指定附件目录')
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    if not data_dir.is_dir():
        parser.error('附件目录不存在')
    if output.is_relative_to(data_dir):
        parser.error('补充输出不能写入原附件目录')
    if output.is_relative_to(PROJECT) and not output.is_relative_to(HERE/'q23_结果'):
        parser.error('项目内仅允许写入 求解/审稿补充/q23_结果；其他输出请放项目外')

    inputs = {}
    def read_project(relative, **kwargs):
        path = PROJECT/relative
        inputs['project:'+relative] = sha256(path)
        return pd.read_csv(path, **kwargs)

    def read_attachment(name):
        path = attachment(data_dir, name)
        inputs['attachment:'+path.relative_to(data_dir).as_posix()] = sha256(path)
        return pd.read_csv(path)

    values = read_project('求解/广义标度律参数.csv').set_index('param').value
    p = Parameters(**{name: float(values[name]) for name in PARAMETER_NAMES})
    key_path = PROJECT/'求解/问题一_关键量.json'
    inputs['project:求解/问题一_关键量.json'] = sha256(key_path)
    key = json.loads(key_path.read_text(encoding='utf-8'))
    q0 = float(key['Q0_mix_weighted'])
    qmap = read_project('求解/17域质量映射.csv').set_index('mixture_domain').quality_Q
    mix = read_project('求解/问题一/结果/推荐配比调整.csv')
    mapped = mix.domain.map(qmap)
    if mapped.isna().any():
        raise ValueError('候选配方有未映射的质量领域')
    reference_q = float(mix.reference_mixture @ mapped)
    candidate_q = float(mix.recommended_mixture @ mapped)
    # Q3 沿用原始平均份额计费；KRR 使用和为 1 的闭合参考配比。
    # 分别重建并核验两种口径，不把闭合后的质量静默写回正式 Q0。
    raw_mix = read_attachment('train_mixture_1m.csv')
    mix_columns = [name for name in raw_mix if name.startswith('train_the_pile_')]
    raw_reference = raw_mix[mix_columns].mean()
    raw_reference.index = raw_reference.index.str.removeprefix('train_the_pile_')
    raw_reference = raw_reference.reindex(mix.domain)
    if raw_reference.isna().any() or not np.isfinite(raw_reference).all():
        raise ValueError('原始平均配比与推荐领域不能一一对应')
    raw_mass = float(raw_reference.sum())
    raw_reference_q = float(raw_reference.to_numpy() @ mapped.to_numpy())
    closed_reference = raw_reference.to_numpy()/raw_mass
    if abs(raw_reference_q-q0) > 1e-10:
        raise ValueError('共享外生成本基线与原始平均份额加权值不一致')
    if not np.allclose(mix.reference_mixture, closed_reference, rtol=0, atol=1e-10):
        raise ValueError('核岭参考配比与原始平均份额的闭合结果不一致')
    if abs(reference_q-raw_reference_q/raw_mass) > 1e-10:
        raise ValueError('闭合参考质量口径核验失败')
    if abs(float(mix.recommended_mixture.sum())-1) > 1e-8 or mix.recommended_mixture.min() < -1e-10:
        raise ValueError('候选配比不满足概率单纯形约束')
    calibration = read_project('求解/问题二/结果/配比项校准.csv')
    krr_calibration = read_project('求解/问题一/结果/核岭推荐_目标对照.csv')
    if list(calibration.mixture) != list(krr_calibration.mixture) or not np.allclose(
            calibration[['predicted_weighted_loss', 'M_p']],
            krr_calibration[['predicted_weighted_loss', 'M_p']], rtol=0, atol=1e-12):
        raise ValueError('Q2 配比修正与 Q1 核岭目标对照不一致')
    gains = calibration.loc[~np.isclose(calibration.M_p, 0), 'M_p']
    if len(gains) != 1:
        raise ValueError('需有一个非零核岭候选配比项')
    m = float(gains.iloc[0])
    if m >= 0:
        raise ValueError('本补充的固定收益幅度算例要求现有 M<0')
    sweep = read_project('求解/问题三/结果/预算扫略.csv').sort_values('C')
    saturated = sweep[sweep.Q >= 1-1e-7]
    if saturated.empty:
        raise ValueError('原扫描没有达到上界的有效点')
    sat_budget = float(saturated.iloc[0].C)
    previous_budget = float(sweep[sweep.C < sat_budget].iloc[-1].C)
    b1 = read_attachment('pythia_training_log_existing.csv')
    b6 = read_attachment('supplementary_NQ_experiment.csv')
    n = np.r_[b1.N_params_B, b6.N_params_B]
    d = np.r_[b1.D_tokens_B, b6.D_tokens_B]
    q = np.r_[np.ones(len(b1)), b6.Q_score]
    y = np.r_[b1.val_loss, b6.val_loss]
    residual = predict(p, n, d, q)-y

    scale_rows = []
    for exponent in [.5, 1., 2.]:
        h0, hp = q0**exponent, exponent*q0**(exponent-1)
        benefit = p.C*p.gamma*(1-h0)**(p.gamma-1)*hp
        ncrit = (p.alpha*p.A*(20e9*q0**3)/(K*1e9*benefit))**(1/(p.alpha+1))
        dcrit = (p.beta*p.B/(p.alpha*p.A*ncrit**-p.alpha))**(1/p.beta)
        scale_rows.append(dict(mapping_power=exponent, Q_B_at_A_baseline=h0, derivative_h=hp, marginal_quality_benefit=benefit, local_baseline_release_budget=K*ncrit*dcrit*1e18))

    wide = b6.pivot(index=['N_params_B','D_tokens_B'], columns='Q_score', values='val_loss')
    if .1 not in wide or 1. not in wide or wide[[.1,1.]].isna().any().any():
        raise ValueError('B6组内质量端点不完整')
    contrasts = (wide[.1]-wide[1.]).rename('given_loss_reduction').reset_index()
    contrasts['additive_model_reduction'] = p.C*.9**p.gamma
    contrasts['contrast_residual'] = contrasts.additive_model_reduction-contrasts.given_loss_reduction
    contrast_summary = dict(groups=len(contrasts), minimum=float(contrasts.given_loss_reduction.min()), maximum=float(contrasts.given_loss_reduction.max()), median=float(contrasts.given_loss_reduction.median()), additive_model=float(p.C*.9**p.gamma), spearman_vs_N=float(spearmanr(contrasts.N_params_B, contrasts.given_loss_reduction).statistic), spearman_vs_D=float(spearmanr(contrasts.D_tokens_B, contrasts.given_loss_reduction).statistic))

    def restricted_residual(vector):
        alt = Parameters(*vector, 1.)
        return predict(alt, n, d, q)-y
    fit = least_squares(restricted_residual, p.vector()[:-1], bounds=([0,1e-8,.01,1e-8,.01,1e-8], [5,1e3,2,1e3,2,50]), loss='soft_l1', f_scale=.05, max_nfev=3000)
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError('gamma=1受限拟合未成功')
    alt = Parameters(*[float(x) for x in fit.x], 1.)
    alt_residual = restricted_residual(fit.x)
    fit_rows = [dict(case='原参数', gamma=p.gamma, objective=robust_objective(residual), **errors(residual)), dict(case='gamma固定1受限拟合', gamma=1., objective=robust_objective(alt_residual), **errors(alt_residual))]
    relative_cost = fit_rows[1]['objective']/fit_rows[0]['objective']-1
    endpoint_rows, endpoint_candidates = [], {}
    for name, pp in [('原参数', p), ('gamma固定1受限拟合', alt)]:
        best, candidates = quality_profile(pp, sat_budget, q0)
        endpoint_rows.append(dict(case=name, **best))
        endpoint_candidates[name] = candidates

    profile_rows = []
    for budget in sorted(set([1e19, previous_budget, sat_budget, 1e22])):
        best, _ = quality_profile(p, budget, q0)
        original = sweep.iloc[np.argmin(np.abs(np.log(sweep.C/budget)))]
        if not np.isclose(original.C, budget, rtol=1e-9):
            raise ValueError('抽查预算不在原扫描中')
        gap = best['L']-float(original.L)
        if abs(gap) > 1e-7:
            raise RuntimeError(f'剖面与保存解的损失差需要复核：{gap}')
        profile_rows.append(dict(**best, saved_L=float(original.L), saved_Q=float(original.Q), profile_minus_saved_loss=gap))

    base = fixed_quality(p, 1e22, 1., q0)
    transfer_rows = []
    for fraction in [0.,.25,.5,.75,1.]:
        target = base['L']+fraction*m
        logc = brentq(lambda logc: fixed_quality(p, np.exp(logc), 1., q0)['L']-target, np.log(1e21), np.log(1e25))
        budget = float(np.exp(logc))
        transfer_rows.append(dict(retained_fraction=fraction, M_transferred=fraction*m, target_loss=target, equivalent_budget=budget, budget_multiple=budget/1e22))
    baseline_rows = [dict(case='正式Q3外生成本基线', Q0=q0, **base),
                     dict(case='仅将闭合参考质量作为费用基线的对照', Q0=reference_q, **fixed_quality(p, 1e22, 1., reference_q)),
                     dict(case='仅将核岭候选质量作为费用基线的对照', Q0=candidate_q, **fixed_quality(p, 1e22, 1., candidate_q))]
    official_transfer = read_project('求解/问题三/结果/配比项算力等价.csv')
    official_candidate = official_transfer.loc[official_transfer.M_p.ne(0)]
    if len(official_candidate) != 1:
        raise ValueError('正式Q3配比表须有且仅有一个非零配比项')
    saved = official_candidate.iloc[0]
    official_checks = dict(M=float(saved.M_p), target_loss=float(saved.L), budget_multiple=float(saved.budget_multiplier),
                           independent_target_loss=transfer_rows[-1]['target_loss'], independent_budget_multiple=transfer_rows[-1]['budget_multiple'])
    if abs(float(saved.M_p)-m)>1e-12 or abs(float(saved.L)-transfer_rows[-1]['target_loss'])>1e-7:
        raise ValueError('正式Q3配比损失与独立代回不一致')
    if not np.isclose(saved.budget_multiplier, transfer_rows[-1]['budget_multiple'], rtol=1e-6, atol=1e-8):
        raise ValueError('正式Q3等效预算与独立反解不一致')

    classic = read_project('求解/问题二/结果/经典标度律参数.csv').iloc[0]
    classic_pred = classic.E + classic.A*b1.N_params_B**-classic.alpha + classic.B*b1.D_tokens_B**-classic.beta
    metric_rows = [dict(case='经典全量', rows=len(b1), **errors(np.asarray(classic_pred-b1.val_loss))), dict(case='广义联合', rows=len(y), **errors(residual))]
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    summary = dict(source_revision=revision, scope='独立补充诊断，不替换主模型参数或正式结果；不构成统计置信区间', tool_record=dict(assistant='OpenAI Codex', purpose='审稿补充代码整理与核验', recorded_date='2026-09-27'), created_utc=datetime.now(timezone.utc).isoformat(), parameters=asdict(p), fixed_gamma_parameters=asdict(alt), Q0_reference=q0, Q0_candidate=candidate_q, quality_scale_counterexample=scale_rows, B6_contrast=contrast_summary, gamma_restricted=dict(success=bool(fit.success), nfev=int(fit.nfev), objective_relative_change=relative_cost, fit_rows=fit_rows, endpoint_budget=sat_budget, endpoint_profiles=endpoint_rows, endpoint_candidates=endpoint_candidates), saved_solution_checks=profile_rows, M_amplitude=transfer_rows, baseline_sensitivity=baseline_rows, original_model_metrics=metric_rows, input_sha256=inputs, script_sha256=sha256(Path(__file__)), versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))

    summary['revision_note'] = 'source_revision 是读取时的 Git HEAD；本补充脚本可尚未提交，其精确身份由 script_sha256 标识。'
    summary['quality_baselines'] = dict(Q0_official_exogenous=q0, raw_reference_mass=raw_mass,
                                       Q_raw_reference=raw_reference_q, Q_closed_reference=reference_q,
                                       Q_krr_candidate=candidate_q, closed_minus_official=reference_q-q0,
                                       policy='正式Q3保持共享外生Q0；闭合参考质量与候选质量仅作另列成本基线对照')
    summary['M_source'] = 'Q1 核岭推荐_目标对照.csv，经 Q2 配比项校准.csv 传递'
    summary['official_transfer_checks'] = official_checks
    # 输入在本次诊断前后保持不变；程序只在独立输出目录写入自己的文件。
    for spec, digest in inputs.items():
        kind, relative = spec.split(':', 1)
        path = (PROJECT if kind=='project' else data_dir)/relative
        if sha256(path) != digest:
            raise RuntimeError(f'运行期间输入发生变化：{spec}')
    output.mkdir(parents=True, exist_ok=True)
    frames = {'质量尺度反例.csv':pd.DataFrame(scale_rows), 'B6组内质量改善.csv':contrasts, 'gamma固定1_参数.csv':pd.DataFrame({'param':PARAMETER_NAMES, 'original':p.vector(), 'restricted':alt.vector()}), 'gamma固定1_拟合对照.csv':pd.DataFrame(fit_rows), 'gamma固定1_临界预算对照.csv':pd.DataFrame(endpoint_rows), '原解剖面抽查.csv':pd.DataFrame(profile_rows), '配比收益幅度敏感性.csv':pd.DataFrame(transfer_rows), '配比质量基线对照.csv':pd.DataFrame(baseline_rows), '原模型误差复核.csv':pd.DataFrame(metric_rows)}
    for name, frame in frames.items():
        frame.to_csv(output/name, index=False, encoding='utf-8-sig')
    (output/'q23_核验.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'README.md').write_text('''# Q2/Q3 审稿补充结果

运行 `python 求解/审稿补充/q23_尺度与结构敏感性.py --data-dir <附件目录>`。
可添加 `--output-dir <项目外独立目录>`；在项目内仅允许写本目录。

- 质量尺度表固定A评分轴上的题设成本，改变损失轴转换；给出基线一阶释放点，不宣称全局转移阈值。
- B6表是相同N、D内Q从0.1提高到1的给定损失差，不是新增真实训练实验。
- gamma=1仅受限重估其他6个参数，用相同soft-L1目标比较，不替代原参数，也不构成置信区间。
- 剖面抽查只核对少量预算的端点与局部候选，不能证明所有预算全局最优。
- 配比幅度表在Q=1、现有费用关系下反解预算，幅度比例为假设；不代表实测节约。
- 配比基线表区分正式Q3外生Q0、核岭闭合参考质量、核岭候选质量；后两项仅改变费用基线作对照，正式Q3不变。
- 原模型误差表只用保存参数代回数据，无额外主模型拟合。

输入及脚本SHA-256、版本、求解状态和全精度数值见q23_核验.json。原模型、附件、参数和原结果不被写入。
''', encoding='utf-8')
    print(json.dumps(dict(output=str(output), B6_groups=len(contrasts), restricted_objective_increase_percent=100*relative_cost, Q_at_original_saturation=endpoint_rows[1]['Q'], profile_checks=len(profile_rows), original_metrics=metric_rows, input_files_unchanged=len(inputs)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
