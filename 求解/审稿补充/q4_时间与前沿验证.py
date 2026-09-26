"""问题四的事后时间诊断，不替换论文原模型或原结果。

用法：python q4_时间与前沿验证.py --data /path/to/real_attachments
仅读取附件和问题四已有结果；默认写入本目录的 q4_结果。
改变前沿点权重/时间窗口仅用于敏感性检查，不是新选定的预测方案。
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
PAPER_RESULT = HERE.parent / '问题四' / '结果'
TASKS = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
LICENSES = {'apache-2.0', 'mit', 'gpl-3.0', 'cc-by-4.0', 'mpl-2.0',
            'bsd-3-clause', 'llama2', 'llama3', 'llama3.1', 'llama3.2',
            'llama3.3', 'gemma', 'cc-by-sa-4.0', 'openrail'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_hashes():
    return {str(p.relative_to(PAPER_RESULT)): digest(p)
            for p in PAPER_RESULT.rglob('*') if p.is_file()}


def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def fit_curve(t, y):
    # 沿用原拟合的函数、初值、参数界限；未重新搜索超参数。
    p, _ = curve_fit(logistic, t, y, p0=[55, .8, 3.5],
                     bounds=([20, .01, .5], [100, 5, 8]), maxfev=40000)
    residual = y - logistic(t, *p)
    r2 = 1 - np.sum(residual**2) / np.sum((y - y.mean())**2)
    return p, float(r2)


def ols(df, time_column='year_t', include_type=False):
    X = np.column_stack([np.ones(len(df)), df.logP, df[time_column]])
    if include_type:
        X = np.column_stack([X, pd.get_dummies(df.type_group, drop_first=True, dtype=float)])
    beta = np.linalg.lstsq(X, df.avg, rcond=None)[0]
    fitted = X @ beta
    return {'n': len(df), 'b_P': beta[1], 'time_coefficient': beta[2],
            'R2': 1 - np.sum((df.avg-fitted)**2)/np.sum((df.avg-df.avg.mean())**2)}


def prepare(data):
    src = data / 'C_efficiency_evolution'
    lb = pd.read_csv(src / 'leaderboard_cleaned.csv')
    lb['P'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
    lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
    lb['avg'] = pd.to_numeric(lb['Average \u2b06\ufe0f'], errors='coerce')
    lb[TASKS] = lb[TASKS].apply(pd.to_numeric, errors='coerce')
    good = (lb[TASKS].notna().all(axis=1) & lb.date.notna() & lb.P.notna()
            & lb['Hub License'].astype(str).str.strip().str.lower().isin(LICENSES))
    core = lb[good].copy()
    if not (core.P.gt(0).all() and core.avg.notna().all()):
        raise ValueError('C1 出现不满足原回归定义的参数或均分，须先核对原样本口径。')
    core['Year'] = core.date.dt.year
    core['year_t'] = core.Year - 2019
    core['continuous_t'] = (core.date-pd.Timestamp('2019-01-01')).dt.total_seconds()/86400/365.25
    core['logP'] = np.log10(core.P)
    pre = ['\U0001f7e2 pretrained', '\U0001f7e9 continuously pretrained']
    chat = ['\U0001f4ac chat models (RLHF, DPO, IFT, ...)', '\U0001f536 fine-tuned on domain-specific datasets']
    core['type_group'] = np.where(core.Type.isin(pre), 'pretrained',
                                   np.where(core.Type.isin(chat), 'chat/finetuned', '其他'))
    core['month'] = core.date.dt.to_period('M')
    monthly = []
    for month, g in core.groupby('month', sort=True):
        if month < pd.Period('2024-01'):
            continue
        n_top = max(1, int(len(g)*.01))
        monthly.append({'source': 'C1', 'period': str(month), 'n_records': len(g),
                        'n_top': n_top, 't': month.year-2019+(month.month-.5)/12,
                        'raw_frontier': float(g.avg.nlargest(n_top).mean())})
    mon = pd.DataFrame(monthly)
    mon['cumulative_frontier'] = mon.raw_frontier.cummax()

    ts = pd.read_csv(src / 'leaderboard_extended_timeseries.csv')
    ts['avg'] = pd.to_numeric(ts.Average, errors='coerce')
    ts['P'] = pd.to_numeric(ts.Params_B, errors='coerce')
    six = ['IFEval', 'BBH', 'MATH_Lvl5', 'GPQA', 'MUSR', 'MMLU_PRO']
    ts[six] = ts[six].apply(pd.to_numeric, errors='coerce')
    ts['gap'] = (ts.avg-ts[six].mean(axis=1)).abs()
    valid = ts[(ts.gap <= 5) & ts.P.notna() & ts.avg.notna() & ts.P.gt(0)].copy()
    valid = valid[np.isfinite(np.log10(valid.P)) & np.isfinite(valid.avg)]
    annual = valid.groupby('Year').agg(raw_frontier=('avg', 'max'), n_records=('avg', 'size')).reset_index()
    annual['cumulative_frontier'] = annual.raw_frontier.cummax()
    hist = annual[annual.Year <= 2023].copy()
    hist['source'] = 'C3'
    hist['period'] = hist.Year.astype(int).astype(str)
    hist['n_top'] = 1
    hist['t'] = hist.Year-2019+.5
    columns = ['source', 'period', 'n_records', 'n_top', 't', 'raw_frontier', 'cumulative_frontier']
    frontier = pd.concat([hist[columns], mon[columns]], ignore_index=True).sort_values('t')
    frontier['cumulative_frontier'] = frontier.cumulative_frontier.cummax()
    return core, valid, frontier.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=os.environ.get('MODELING_DATA_DIR'))
    ap.add_argument('--output', type=Path, default=HERE/'q4_结果')
    args = ap.parse_args()
    if args.data is None:
        ap.error('请用 --data 或 MODELING_DATA_DIR 指定 real_attachments。')
    data, output = Path(args.data).resolve(), args.output.resolve()
    if output.is_relative_to(data) or output.is_relative_to(PAPER_RESULT.resolve()):
        ap.error('输出目录不能位于原始附件或问题四原结果目录内。')
    before = result_hashes()
    core, c3, frontier = prepare(data)
    t = frontier.t.to_numpy(float)
    y = frontier.cumulative_frontier.to_numpy(float)
    if len(t) != 14 or not np.isclose(t[-1], 2025-2019+(3-.5)/12):
        raise ValueError('本诊断针对截至2025年3月的14点序列；数据窗口变化后须重新确定留出期与表头。')
    future = np.array([t[-1]+1, t[-1]+2])
    original, original_r2 = fit_curve(t, y)
    saved = pd.read_csv(PAPER_RESULT/'logistic参数.csv').iloc[0]
    saved_forecast = pd.read_csv(PAPER_RESULT/'前沿预测.csv')
    if not np.allclose(original, saved[['K','r','t0(2019起)']].astype(float), rtol=2e-5, atol=1e-5):
        raise ValueError('重建的原曲线与论文结果不一致，停止生成诊断。')
    if not np.allclose(logistic(future,*original), saved_forecast.frontier_pred, rtol=2e-5, atol=1e-5):
        raise ValueError('重建的预测时点/原预测值不一致。')

    regressions = []
    for name, frame, time_col, typed in [
        ('年份标签（原规格）',core,'year_t',False),
        ('按名称保留首条',core.drop_duplicates('Model'),'year_t',False),
        ('按名称保留末条',core.drop_duplicates('Model',keep='last'),'year_t',False),
        ('年份标签并控制类型',core,'year_t',True),
        ('实际提交日连续年数',core,'continuous_t',False),
    ]:
        regressions.append({'diagnostic':name,**ols(frame,time_col,typed)})

    variants = [
        ('原14点',np.ones(len(t),bool)),
        ('仅保留C1月度点',frontier.source.eq('C1').to_numpy()),
        ('去掉末3个月',np.arange(len(t))<len(t)-3),
        ('相同累计值仅留首次',np.r_[True,np.diff(y)!=0]),
    ]
    windows = []
    for name, mask in variants:
        p, r2 = fit_curve(t[mask],y[mask])
        pred = logistic(future,*p)
        bound = ('K上界' if np.isclose(p[0],100,atol=1e-6) else
                 'r上界' if np.isclose(p[1],5,atol=1e-6) else '无')
        windows.append({'diagnostic':name,'n':int(mask.sum()),'K':p[0],'r':p[1],'t0':p[2],
                        'training_R2':r2,'value_2026_03':pred[0],'value_2027_03':pred[1],
                        'active_bound':bound})

    cutoff = 2024-2019+(9-.5)/12
    train = t <= cutoff+1e-9
    p, _ = fit_curve(t[train],y[train])
    hold = frontier.loc[~train,['period','t','cumulative_frontier']].copy()
    hold['logistic_prediction'] = logistic(t[~train],*p)
    hold['last_peak_prediction'] = y[train][-1]
    metrics = []
    for name,col in [('logistic条件曲线','logistic_prediction'),('保持最后累计峰值','last_peak_prediction')]:
        err = hold[col]-hold.cumulative_frontier
        metrics.append({'method':name,'training_cutoff':'2024-09','holdout':'2024-10至2025-03',
                        'n':len(hold),'RMSE':float(np.sqrt(np.mean(err**2))),'MAE':float(np.mean(np.abs(err)))})

    mean_dates = core.groupby('Year').date.mean()
    summary = {
        'purpose':'事后诊断，不替代论文原模型、原结果或点预测；单次留出窗口恰为累计平台',
        'n_C1':len(core),'n_C3_valid':len(c3),'unique_exact_names':int(core.Model.nunique()),
        'names_in_multiple_years':int((core.groupby('Model').Year.nunique()>1).sum()),
        'mean_submission_dates':{str(k):str(v) for k,v in mean_dates.items()},
        'mean_date_gap_years':float(core.groupby('Year').continuous_t.mean().diff().iloc[-1]),
        'origin_observed_cumulative':float(y[-1]),'origin_logistic_fitted':float(logistic(t[-1],*original)),
        'original_R2':original_r2,'original_result_verified':True,
        'curve_inputs':'C3年度累计最高分与C1月度前1%均值累计最大值；未作历史量表校准',
        'diagnostic_limits':'删点改变权重和时间窗口；单次留出为事后检查，不能用于宣称基线普遍优越',
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
        'original_result_sha256':before,
        'source_hashes':{name:digest(data/'C_efficiency_evolution'/name) for name in
                         ['leaderboard_cleaned.csv','leaderboard_extended_timeseries.csv']},
    }
    output.mkdir(parents=True,exist_ok=True)
    for name,df in [('q4_横截面诊断',pd.DataFrame(regressions)),('q4_窗口敏感性',pd.DataFrame(windows)),
                    ('q4_时间留出',pd.DataFrame(metrics)),('q4_时间留出明细',hold),('q4_前沿诊断序列',frontier)]:
        df.to_csv(output/(name+'.csv'),index=False,encoding='utf-8-sig')
    if before != result_hashes():
        raise RuntimeError('问题四原结果文件发生变化。')
    summary['original_results_unchanged'] = True
    (output/'q4_诊断摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(regressions).to_string(index=False))
    print(pd.DataFrame(windows).to_string(index=False))
    print(pd.DataFrame(metrics).to_string(index=False))
    print('诊断完成；原曲线一致，原结果文件未改动。')


if __name__ == '__main__':
    main()
