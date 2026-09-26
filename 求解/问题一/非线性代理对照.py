# -*- coding: utf-8 -*-
# 本程序的实现、验证与说明由 OpenAI Codex（GPT-6）辅助完成。
"""同尺度配比预测：既有线性岭基线与平方根嵌入核岭。

python 求解/问题一/非线性代理对照.py --data-dir <附件根目录>
只写“非线性代理对照_*”结果，不写任何跨问接口。绘图另由重绘图表.py执行。
核岭采用闭合配方、训练核中心化、未惩罚截距；全部超参数只用训练五折选择。
CV最小值是调参得分，不是嵌套CV泛化估计；既有检验集不是新的盲测集。
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import importlib.metadata

import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
SEED = 20260926
LAMBDAS = 10.0 ** np.arange(-6, 3)
GAMMAS = 2.0 ** np.arange(-6, 3)


def find_data(path=None):
    candidates = [Path(path)] if path else [p / 'real_attachments' for p in HERE.parents]
    for p in candidates:
        if (p / 'A_data_value/regmix_tables').is_dir():
            return p.resolve()
    raise FileNotFoundError('请用 --data-dir 指定包含 A_data_value 的附件根目录')


def load_pair(root, split, scale):
    folder = root / 'A_data_value/regmix_tables'
    paths = [folder / f'{split}_mixture_{scale}.csv',
             folder / f'{split}_pile_loss_{scale}.csv']
    mix, loss = [pd.read_csv(p).sort_values('index').reset_index(drop=True) for p in paths]
    if mix['index'].duplicated().any() or loss['index'].duplicated().any():
        raise ValueError('配方或损失存在重复 index')
    if not np.array_equal(mix['index'].values, loss['index'].values):
        raise ValueError('配方与损失 index 不匹配')
    mc = [c for c in mix if c.startswith('train_the_pile_')]
    lc = [c for c in loss if c.endswith('_val_loss')]
    x, y = mix[mc].to_numpy(float), loss[lc].to_numpy(float)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and (x >= 0).all()
            and (x.sum(1) > 0).all()):
        raise ValueError('非有限值、负配比或零行和')
    domains = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in lc]
    return x, y, mix['index'].values, mc, domains, paths


def closed(x):
    return x / x.sum(axis=1, keepdims=True)


def ridge_predict(x, y, v, alpha=1e-3):
    # 与问题一.py完全相同：17个原始份额，截距不受惩罚。
    a = np.column_stack([x, np.ones(len(x))])
    penalty = alpha * np.eye(a.shape[1])
    penalty[-1, -1] = 0
    w = np.linalg.solve(a.T @ a + penalty, a.T @ y)
    return np.column_stack([v, np.ones(len(v))]) @ w


def kernel(x, v, gamma):
    return np.exp(-gamma * cdist(np.sqrt(x), np.sqrt(v), 'sqeuclidean'))


def centered_kernel(k):
    means, grand = k.mean(axis=0), float(k.mean())
    return k - means[None, :] - means[:, None] + grand, means, grand


def cross_center(k, means, grand):
    return k - k.mean(axis=1, keepdims=True) - means[None, :] + grand


def krr_predict(x, y, v, gamma, lam):
    kc, means, grand = centered_kernel(kernel(x, x, gamma))
    ym = y.mean(0)
    coef = solve(kc + lam * np.eye(len(x)), y - ym, assume_a='pos')
    return cross_center(kernel(v, x, gamma), means, grand) @ coef + ym


def metrics(y, pred):
    err = y - pred
    r2 = 1 - (err**2).sum(0) / np.maximum(((y - y.mean(0))**2).sum(0), 1e-12)
    rho = np.array([spearmanr(y[:, j], pred[:, j]).statistic for j in range(y.shape[1])])
    rmse = np.sqrt((err**2).mean(0))
    mae = np.abs(err).mean(0)
    overall = dict(R2_mean=float(r2.mean()), Spearman_domain_mean=float(rho.mean()),
                   RMSE_pooled=float(np.sqrt((err**2).mean())),
                   RMSE_domain_mean=float(rmse.mean()), MAE=float(mae.mean()),
                   RMSE_recipe_mean=float(np.sqrt(((y.mean(1) - pred.mean(1))**2).mean())),
                   Spearman_recipe_mean=float(spearmanr(y.mean(1), pred.mean(1)).statistic))
    return overall, dict(R2=r2, Spearman=rho, RMSE=rmse, MAE=mae)


def cv_search(xraw, y, folds, gammas=GAMMAS, lambdas=LAMBDAS):
    x = closed(xraw)
    oof_ridge = np.empty_like(y)
    squared = np.zeros((len(gammas), len(lambdas)))
    fold_rows = []
    for fold, (tr, va) in enumerate(folds, 1):
        oof_ridge[va] = ridge_predict(xraw[tr], y[tr], xraw[va])
        for g, gamma in enumerate(gammas):
            kc, means, grand = centered_kernel(kernel(x[tr], x[tr], gamma))
            eig, vectors = eigh(kc)
            if eig.min() < -1e-9:
                raise ValueError('中心化核出现显著负特征值')
            eig = np.maximum(eig, 0)
            ym = y[tr].mean(0)
            rotated = vectors.T @ (y[tr] - ym)
            projection = cross_center(kernel(x[va], x[tr], gamma), means, grand) @ vectors
            for a, lam in enumerate(lambdas):
                pred = projection @ (rotated / (eig[:, None] + lam)) + ym
                sse = float(((y[va] - pred)**2).sum())
                squared[g, a] += sse
                fold_rows.append(dict(fold=fold, n_valid=len(va), gamma=gamma, lambda_=lam,
                                      SSE=sse, RMSE=np.sqrt(sse / y[va].size)))
        print(f'五折选择：{fold}/{len(folds)}', flush=True)
    scores = np.sqrt(squared / y.size)
    g, a = np.unravel_index(scores.argmin(), scores.shape)
    gamma, lam = float(gammas[g]), float(lambdas[a])
    oof_krr = np.empty_like(y)
    for tr, va in folds:
        oof_krr[va] = krr_predict(x[tr], y[tr], x[va], gamma, lam)
    grid = [dict(gamma=float(gammas[g]), lambda_=float(lambdas[a]), CV_RMSE=float(scores[g, a]))
            for g in range(len(gammas)) for a in range(len(lambdas))]
    return gamma, lam, oof_ridge, oof_krr, pd.DataFrame(grid), pd.DataFrame(fold_rows)


def self_check(data_root):
    # 与增广核方程独立比较，检验训练/预测中心化及不惩罚截距。
    raw, y, _, _, _, _ = load_pair(data_root, 'train', '1m')
    x = closed(raw[:32]); v = closed(raw[32:40]); y = y[:32]
    gamma, lam = .125, .01
    k = kernel(x, x, gamma)
    augmented = np.block([[k + lam*np.eye(len(x)), np.ones((len(x), 1))],
                          [np.ones((1, len(x))), np.zeros((1, 1))]])
    rhs = np.vstack([y, np.zeros((1, y.shape[1]))])
    solution = np.linalg.solve(augmented, rhs)
    reference = kernel(v, x, gamma) @ solution[:-1] + solution[-1]
    actual = krr_predict(x, y, v, gamma, lam)
    np.testing.assert_allclose(actual, reference, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(krr_predict(x, y+7, v, gamma, lam), actual+7, atol=1e-9)
    folds = list(KFold(2, shuffle=True, random_state=SEED).split(raw[:40]))
    cv_search(raw[:40], load_pair(data_root, 'train', '1m')[1][:40], folds,
              gammas=np.array([.125]), lambdas=np.array([.01]))
    print(f'最小核验通过；增广方程最大偏差 {abs(actual-reference).max():.3g}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--output-dir', type=Path, default=HERE/'结果')
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args(); root = find_data(args.data_dir)
    if args.self_check:
        self_check(root); return
    start = time.monotonic()
    raw, y, indices, columns, domains, inputs = load_pair(root, 'train', '1m')
    x = closed(raw)
    if len(np.unique(x, axis=0)) != len(x):
        raise ValueError('重复配方需采用分组交叉验证')
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(x))
    gamma, lam, cv_r, cv_k, grid, foldscores = cv_search(raw, y, folds)
    fold_ids = np.empty(len(y), int)
    for fold, (_, va) in enumerate(folds, 1): fold_ids[va] = fold
    summary, detail, predictions = [], [], []

    def record(scope, idx, obs, preds):
        for model, pred in preds.items():
            overall, per_domain = metrics(obs, pred)
            summary.append(dict(scope=scope, model=model, n=len(obs), **overall))
            for j, domain in enumerate(domains):
                detail.append(dict(scope=scope, model=model, loss_domain=domain,
                                   **{key: float(val[j]) for key, val in per_domain.items()}))
                predictions.append(pd.DataFrame(dict(scope=scope, model=model, index=idx,
                    loss_domain=domain, observed=obs[:, j], predicted=pred[:, j])))
    record('train_oof', indices, y, dict(ridge=cv_r, sqrt_krr=cv_k))
    profiles = [dict(scope='train', n=len(raw), min_share=float(raw.min()),
                     max_row_sum_error=float(abs(raw.sum(1)-1).max()))]
    for scale in ['1m', '60m', '1B']:
        test, obs, idx, mc, ds, paths = load_pair(root, 'test', scale)
        if mc != columns or ds != domains: raise ValueError('跨文件字段顺序不一致')
        inputs += paths
        record(scale, idx, obs, dict(ridge=ridge_predict(raw, y, test),
                                    sqrt_krr=krr_predict(x, y, closed(test), gamma, lam)))
        profiles.append(dict(scope=scale, n=len(test), min_share=float(test.min()),
                             max_row_sum_error=float(abs(test.sum(1)-1).max())))
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    tables = dict(汇总=pd.DataFrame(summary), 逐域=pd.DataFrame(detail),
                  预测明细=pd.concat(predictions, ignore_index=True), 参数搜索=grid,
                  五折=foldscores, 折分配=pd.DataFrame(dict(index=indices, fold=fold_ids)))
    for name, table in tables.items():
        table.to_csv(out/f'非线性代理对照_{name}.csv', index=False, encoding='utf-8-sig')
    manifest = dict(seed=SEED, n_train=len(y), n_domains=len(domains), selected_gamma=gamma,
                    selected_lambda=lam, baseline_lambda=.001, profiles=profiles,
                    lambda_grid=LAMBDAS.tolist(), gamma_grid=GAMMAS.tolist(),
                    selection='minimum pooled out-of-fold RMSE; not nested CV',
                    preprocessing='KRR: row closure, sqrt, fold-local kernel/output centering; ridge: original shares',
                    test_status='previously inspected existing test sets; not a fresh blind holdout',
                    python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
                    inputs={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    command='python 求解/问题一/非线性代理对照.py --data-dir <附件根目录>',
                    elapsed_seconds=time.monotonic()-start,
                    shared_outputs_modified=False)
    (out/'非线性代理对照_复现清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'选择 gamma={gamma}, lambda={lam}')
    print(tables['汇总'].to_string(index=False))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    main()
