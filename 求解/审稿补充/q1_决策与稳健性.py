# -*- coding: utf-8 -*-
"""问题一的独立诊断；不覆盖原评分、拟合、推荐或跨问接口。

用法：python 求解/审稿补充/q1_决策与稳健性.py --data-dir <real_attachments>
可用 --section decision 只复核决策，或 --section quality 只复核标尺。
固定既有超参数，不重新搜索核岭；四档岭参数沿用原敏感性区间。
线性对照读取“线性有界配比对照.csv”；正式配方读取“推荐配比调整.csv”。
诊断不产生真实语言模型训练结果。实现与说明经 OpenAI Codex 辅助。
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import lzma
import platform

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / '求解/问题一/结果'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pure_definitions(relative, functions, assignments=()):
    """只载入白名单纯函数/常量，绝不执行原求解脚本的顶层流程。"""
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in functions
    ) or (isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in assignments for target in node.targets))]
    namespace = {'np': np, 'cdist': cdist, 'solve': solve}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<q1-pure-definitions>', 'exec'), namespace)
    return namespace


def write_json(out, name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def decision_diagnostics(data, out):
    folder = data / 'A_data_value/regmix_tables'
    mix = pd.read_csv(folder / 'train_mixture_1m.csv').sort_values('index')
    loss = pd.read_csv(folder / 'train_pile_loss_1m.csv').sort_values('index')
    if mix['index'].duplicated().any() or not np.array_equal(mix['index'], loss['index']):
        raise ValueError('训练配方与损失的 index 不匹配或重复')
    mc = [c for c in mix if c.startswith('train_the_pile_')]
    lc = [c for c in loss if c.endswith('_val_loss')]
    domains = [c.removeprefix('train_the_pile_') for c in mc]
    targets = [c.removeprefix('metric/the_pile_').removesuffix('_val_loss') for c in lc]
    x, y = mix[mc].to_numpy(float), loss[lc].to_numpy(float)
    ref = x.mean(0)
    saved_recipe = pd.read_csv(RESULTS / '线性有界配比对照.csv').set_index('domain').loc[domains]
    rec = saved_recipe.recommended_mixture.to_numpy(float)
    np.testing.assert_allclose(ref, saved_recipe.reference_mixture, atol=1e-12, rtol=0)
    weights = ref[[domains.index(d) for d in targets]].copy()
    weights /= weights.sum()
    a0 = pd.read_csv(RESULTS / '混合系数矩阵.csv').set_index('training_domain').loc[domains, targets].to_numpy(float)
    b0 = pd.read_csv(RESULTS / '截距.csv').set_index('loss_domain').loc[targets, 'intercept'].to_numpy(float)
    ac0 = a0 - a0.mean(0)
    fixed = np.array([d in {'nih_exporter', 'enron_emails', 'europarl', 'philpapers'} for d in domains])
    bounds = [(ref[i], ref[i]) if fixed[i] else (0, 2.5 * ref[i]) for i in range(len(domains))]

    sensitivity, recipes = [], []
    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        design = np.column_stack([x, np.ones(len(x))])
        penalty = lam * np.eye(design.shape[1])
        penalty[-1, -1] = 0
        fit = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        a, b = fit[:-1], fit[-1]
        centered = a - a.mean(0)
        opt = linprog(a @ weights, A_eq=np.ones((1, len(domains))), b_eq=[1], bounds=bounds, method='highs')
        if not opt.success:
            raise RuntimeError(opt.message)
        if lam == 1e-3:
            np.testing.assert_allclose(a, a0, atol=1e-10, rtol=0)
            np.testing.assert_allclose(b, b0, atol=1e-10, rtol=0)
            np.testing.assert_allclose(opt.x, rec, atol=1e-9, rtol=0)
        sensitivity.append({
            'lambda': lam,
            'centered_relative_change': float(np.linalg.norm(centered-ac0) / np.linalg.norm(ac0)),
            'centered_sign_flips': int(np.sum(centered * ac0 < 0)),
            'recipe_L1_change': float(np.abs(opt.x-rec).sum()),
            'candidate_regret_under_main_model': float((opt.x-rec) @ a0 @ weights),
            'own_model_reference_minus_candidate': float((ref-opt.x) @ a @ weights),
        })
        recipes.extend({'lambda': lam, 'domain': d, 'mixture': float(v)} for d, v in zip(domains, opt.x))

    table = pd.read_csv(RESULTS / '非线性代理对照_预测明细.csv')
    weighted = []
    for (scope, model), group in table.groupby(['scope', 'model']):
        yp_frame = group.pivot(index='index', columns='loss_domain', values='predicted')[targets]
        yo_frame = group.pivot(index='index', columns='loss_domain', values='observed')[targets]
        yp, yo = yp_frame.to_numpy() @ weights, yo_frame.to_numpy() @ weights
        best = int(np.argmin(yp))
        weighted.append({
            'scope': scope, 'model': model, 'n_recipes': len(yp),
            'weighted_RMSE': float(np.sqrt(np.mean((yp-yo)**2))),
            'weighted_Spearman': float(spearmanr(yp, yo).statistic),
            'predicted_best_index': int(yp_frame.index[best]),
            'observed_rank_of_predicted_best': float(pd.Series(yo).rank(method='average').iloc[best]),
            'observed_regret_of_predicted_best': float(yo[best]-yo.min()),
        })

    krr = pure_definitions('求解/问题一/非线性代理对照.py', {'kernel', 'centered_kernel', 'cross_center', 'krr_predict'})['krr_predict']
    config = json.loads((RESULTS / '非线性代理对照_复现清单.json').read_text(encoding='utf-8-sig'))
    gamma, krr_lambda = float(config['selected_gamma']), float(config['selected_lambda'])
    px = x / x.sum(1, keepdims=True)
    tx_frame = pd.read_csv(folder / 'test_mixture_1m.csv').sort_values('index')
    tx = tx_frame[mc].to_numpy(float)
    test_pred = krr(px, y, tx / tx.sum(1, keepdims=True), gamma, krr_lambda)
    saved_test = table[(table.scope == '1m') & (table.model == 'sqrt_krr')].pivot(index='index', columns='loss_domain', values='predicted').loc[tx_frame['index'], targets].to_numpy()
    max_error = float(np.max(np.abs(test_pred-saved_test)))
    np.testing.assert_allclose(test_pred, saved_test, atol=1e-9, rtol=0)
    kp = krr(px, y, np.vstack([ref/ref.sum(), rec]), gamma, krr_lambda) @ weights
    lp = (np.vstack([ref, rec]) @ a0 + b0) @ weights
    candidate = [dict(model=model, reference=float(v[0]), candidate=float(v[1]), reference_minus_candidate=float(v[0]-v[1])) for model, v in [('linear_ridge', lp), ('sqrt_krr', kp)]]
    # 给出训练支持域检查；不把凸包外自动解释成预测无效。
    n, p = px.shape
    projection = linprog(np.r_[np.zeros(n), np.ones(2*p)],
        A_eq=np.r_[np.c_[px.T, np.eye(p), -np.eye(p)], np.r_[np.ones(n), np.zeros(2*p)][None, :]],
        b_eq=np.r_[rec, 1], bounds=(0, None), method='highs')
    if not projection.success:
        raise RuntimeError(projection.message)
    formal = pd.read_csv(RESULTS / '推荐配比调整.csv').set_index('domain').loc[domains]
    formal_p = formal.recommended_mixture.to_numpy(float)
    formal_ref = formal.reference_mixture.to_numpy(float)
    np.testing.assert_allclose(formal_ref, ref/ref.sum(), atol=1e-12, rtol=0)
    certificate = pd.read_csv(RESULTS / '核岭推荐_训练凸组合.csv').set_index('index')
    z = certificate.loc[mix['index'], 'convex_weight'].to_numpy(float)
    reconstruction_error = float(np.max(np.abs(z @ px-formal_p)))
    residual = max(float(abs(z.sum()-1)), float(max(0, -z.min())),
                   float(np.max(np.abs(formal_p[fixed]-formal_ref[fixed]))),
                   float(np.maximum(formal_p[~fixed]-2.5*formal_ref[~fixed], 0).max()))
    assert reconstruction_error < 1e-10 and residual < 1e-8
    formal_scores = krr(px, y, np.vstack([formal_ref, formal_p]), gamma, krr_lambda) @ weights
    saved_targets = pd.read_csv(RESULTS / '核岭推荐_目标对照.csv').set_index('mixture')
    np.testing.assert_allclose(formal_scores, saved_targets.loc[['参考配比', '问题一有界推荐'], 'predicted_weighted_loss'], atol=1e-9, rtol=0)
    starts = pd.read_csv(RESULTS / '核岭推荐_多初值.csv')
    formal_check = {
        'recipe_role': 'current_kernel_recommendation_with_training_convex_hull_constraint',
        'reference': float(formal_scores[0]), 'candidate': float(formal_scores[1]),
        'M_p': float(formal_scores[1]-formal_scores[0]),
        'certificate_reconstruction_max_error': reconstruction_error,
        'certificate_constraint_residual': residual,
        'active_training_rows': int(np.sum(z > 1e-8)),
        'multistart_objective_span': float(starts.objective.max()-starts.objective.min()),
        'saved_multistart_all_success': bool(starts.success.all()),
        'new_vs_old_linear_recipe_L1': float(np.abs(formal_p-rec).sum()),
        'interpretation': 'Saved certificate and fixed-model prediction verified; no optimization rerun, global optimality or real-training claim.',
    }
    write_json(out, 'q1_核岭正式配方核验.json', formal_check)
    summary = {
        'status': 'independent_diagnostic_not_new_training',
        'linear_candidate_source': '线性有界配比对照.csv',
        'formal_candidate_source': '推荐配比调整.csv',
        'weights_definition': 'A4 raw mean shares restricted to the 13 validation domains, then normalized',
        'target_weights': dict(zip(targets, weights.tolist())),
        'reference_raw_sum': float(ref.sum()),
        'krr_reference_rule': 'row-closed reference; linear baseline uses original raw shares',
        'fixed_krr_theta': gamma, 'fixed_krr_lambda': krr_lambda,
        'saved_1m_krr_max_abs_error': max_error,
        'candidate_training_convex_hull_L1_distance': float(projection.fun),
        'ridge_sensitivity': sensitivity, 'candidate_model_comparison': candidate,
        'weighted_recipe_diagnostics': weighted,
        'formal_kernel_candidate_check': formal_check,
    }
    pd.DataFrame(weighted).to_csv(out / 'q1_目标一致检验.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(candidate).to_csv(out / 'q1_线性候选双代理对照.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(out / 'q1_正则配置敏感性.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(recipes).to_csv(out / 'q1_正则条件候选配比.csv', index=False, encoding='utf-8-sig')
    write_json(out, 'q1_决策诊断.json', summary)
    return summary


def conflict_diagnostics(out):
    grouped = pd.read_csv(RESULTS / '冲突样本裁决汇总.csv')
    rows = []
    for domain, group in grouped.groupby('domain'):
        flagged = group[group.conflict_flag == True]
        count = lambda label: int(flagged.loc[flagged.decision == label, 'n'].sum())
        rows.append({'domain': domain, 'n': int(group.n.sum()), 'flagged': int(flagged.n.sum()),
                     'flagged_fraction': float(flagged.n.sum()/group.n.sum()),
                     'flagged_below_median': count('低于域中位数'),
                     'flagged_equal_median': count('等于域中位数'),
                     'flagged_above_median': count('高于域中位数')})
    pd.DataFrame(rows).to_csv(out / 'q1_冲突标签定位.csv', index=False, encoding='utf-8-sig')
    return rows


def quality_diagnostics(data, out):
    functions = pure_definitions('求解/问题一/问题一.py', {'_scalarize', '_softmax_expected'}, {'QUALITY_COLS', 'TARGET_COLS', 'NEGATIVE_COLS'})
    cols = functions['QUALITY_COLS']
    raw = data / 'A_data_value'
    paths = [(raw / 'slimpajama_quality_signal_sample.jsonl.xz', 'A1', None)]
    paths += [(p, 'A2' if p.name.startswith('arxiv') else 'A3', 'arxiv' if p.name.startswith('arxiv') else 'github') for p in sorted((raw / 'slimpajama_quality_extended').glob('*.xz'))]
    rows, qr, domains, sources, ids = [], [], [], [], []
    for path, source, hint in paths:
        with lzma.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                rows.append([functions['_scalarize'](c, item.get(c)) for c in cols])
                qr.append(item['qurater'])
                domains.append(item.get('_source_domain', hint))
                sources.append(source)
                ids.append(item['id'])
        print(f'读取 {path.name}: 累计 {len(rows)} 条', flush=True)
    x, qr = np.asarray(rows, float), np.asarray(qr, float)
    domain, source = np.asarray(domains), np.asarray(sources)
    a1 = source == 'A1'
    qi = cols.index('qurater')

    def score(mask):
        q = qr.copy()
        q = np.where(np.isfinite(q), q, np.nanmedian(np.where(np.isfinite(q[mask]), q[mask], np.nan), axis=0))
        lo, hi = np.quantile(q[mask], [.01, .99], axis=0)
        values = x.copy()
        values[:, qi] = np.clip((q-lo)/np.maximum(hi-lo, 1e-12), 0, 1).mean(1)
        med = np.nanmedian(np.where(np.isfinite(values[mask]), values[mask], np.nan), axis=0)
        values = np.where(np.isfinite(values), values, med)
        centers = {}
        for name in functions['TARGET_COLS']:
            j = cols.index(name)
            z = np.log1p(np.maximum(values[:, j], 0))
            m = np.median(z[mask])
            mad = np.median(np.abs(z[mask]-m)) + 1e-9
            centers[name] = float(np.expm1(m))
            values[:, j] = np.exp(-np.abs(z-m)/(2.5*mad))
        lo, hi = np.quantile(values[mask], [.01, .99], axis=0)
        z = np.clip((values-lo)/np.maximum(hi-lo, 1e-12), 0, 1)
        for name in functions['NEGATIVE_COLS']:
            z[:, cols.index(name)] = 1-z[:, cols.index(name)]
        selected = z[mask]
        p = (selected+1e-12)/(selected.sum(0)+1e-12)
        entropy = -(p*np.log(p)).sum(0)/np.log(mask.sum())
        w = np.maximum(1-entropy, 0)
        w /= w.sum()
        return z @ w, w, centers

    full, wf, cf = score(np.ones(len(x), dtype=bool))
    alternate, wa, ca = score(a1)
    saved = pd.read_csv(RESULTS / '样本级质量评分.csv')
    if list(saved.id) != ids or not np.array_equal(saved.source.to_numpy(), source):
        raise ValueError('原始记录次序与保存评分不对应')
    error = float(np.max(np.abs(full-saved.quality_Q.to_numpy())))
    np.testing.assert_allclose(full, saved.quality_Q, atol=2e-10, rtol=0)
    means = pd.DataFrame({'domain': domain, 'pooled_calibration': full, 'A1_calibration': alternate}).groupby('domain').mean().reset_index()
    summary = {
        'status': 'calibration_sensitivity_not_replacement', 'n_records': len(x), 'n_A1': int(a1.sum()),
        'pooled_score_reconstruction_max_error': error,
        'length_centers_pooled': cf, 'length_centers_A1': ca,
        'github_fraction_pooled': float(np.mean(domain == 'github')),
        'github_fraction_A1': float(np.mean(domain[a1] == 'github')),
        'domain_rank_spearman': float(spearmanr(means.pooled_calibration, means.A1_calibration).statistic),
        'domain_means': means.to_dict('records'),
        'quality_global_mean_pooled': float(full.mean()), 'quality_global_mean_A1_calibrated': float(alternate.mean()),
        'raw_inputs': [{'path': p.relative_to(data).as_posix(), 'sha256': digest(p)} for p, _, _ in paths],
    }
    means.to_csv(out / 'q1_A1标尺域均分对照.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'indicator': cols, 'pooled_weight': wf, 'A1_calibrated_weight': wa}).to_csv(out / 'q1_A1标尺权重对照.csv', index=False, encoding='utf-8-sig')
    write_json(out, 'q1_A1标尺诊断.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='包含 A_data_value 的原始附件根目录')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'q1_结果')
    parser.add_argument('--section', choices=['all', 'decision', 'quality'], default='all')
    args = parser.parse_args()
    data, out = args.data_dir.resolve(), args.output_dir.resolve()
    if not (data / 'A_data_value/regmix_tables').is_dir():
        raise FileNotFoundError('附件根目录缺少 A_data_value/regmix_tables')
    if out == RESULTS.resolve() or RESULTS.resolve() in out.parents or out == data or data in out.parents:
        raise ValueError('诊断结果不能写入原结果目录或原始附件')
    out.mkdir(parents=True, exist_ok=True)
    watched = list(RESULTS.glob('*.csv')) + [ROOT / '求解/问题一_关键量.json']
    before = {p: digest(p) for p in watched}
    if args.section in {'all', 'decision'}:
        decision_diagnostics(data, out)
    if args.section in {'all', 'quality'}:
        quality_diagnostics(data, out)
    conflict_diagnostics(out)
    if before != {p: digest(p) for p in watched}:
        raise RuntimeError('原结果或跨问接口在诊断期间发生变化')
    manifest = {
        'section': args.section, 'python': platform.python_version(),
        'description': 'Independent diagnostics; original result hashes unchanged. No language model training.',
        'script_sha256': digest(__file__),
        'method_sources': {p.relative_to(ROOT).as_posix(): digest(p) for p in
                           [ROOT / '求解/问题一/问题一.py', ROOT / '求解/问题一/非线性代理对照.py',
                            ROOT / '求解/问题一/核岭支持域推荐.py']},
        'protected_inputs': {p.relative_to(ROOT).as_posix(): h for p, h in before.items()},
        'regmix_inputs': {p.relative_to(data).as_posix(): digest(p) for p in sorted((data / 'A_data_value/regmix_tables').glob('*.csv'))},
        'outputs': {p.name: digest(p) for p in sorted(out.glob('q1_*')) if p.is_file() and p.name != 'q1_复现清单.json'},
    }
    write_json(out, 'q1_复现清单.json', manifest)
    print('独立诊断完成；原结果、推荐与跨问接口的 SHA256 均未改变。')


if __name__ == '__main__':
    main()
