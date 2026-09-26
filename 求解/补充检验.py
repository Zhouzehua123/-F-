# 本补充检验程序使用 OpenAI Codex 辅助编写与核对。
# 用途：检验保存结果的决策含义、跨问假设和时间外推；不改写正式求解结果。
"""运行：python 求解/补充检验.py --data-dir <附件根目录>
输出写入复现记录/补充检验；核岭超参数沿用保存值，不重新搜索。
两组检验依次执行，后一组读取前一组重建的前沿序列。
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import linprog, curve_fit, least_squares, brentq, minimize_scalar, minimize
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

def check_q1_q4(REPO, DATA, OUT):
    report = {}
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in (REPO/'求解').rglob('*') if p.is_file() and p.suffix in ('.csv','.json')}

    # Q1: evaluate the exact saved recommendation objective, without hyperparameter search.
    a_dir = DATA/'A_data_value/regmix_tables'
    mix = pd.read_csv(a_dir/'train_mixture_1m.csv').sort_values('index')
    loss = pd.read_csv(a_dir/'train_pile_loss_1m.csv').sort_values('index')
    assert np.array_equal(mix['index'], loss['index'])
    mc = [c for c in mix if c.startswith('train_the_pile_')]
    lc = [c for c in loss if c.endswith('_val_loss')]
    domains = [c.removeprefix('train_the_pile_') for c in mc]
    ld = [c.removeprefix('metric/the_pile_').removesuffix('_val_loss') for c in lc]
    xraw = mix[mc].to_numpy(float); x = xraw/xraw.sum(1,keepdims=True); y = loss[lc].to_numpy(float)
    rec = pd.read_csv(REPO/'求解/问题一/结果/线性有界配比对照.csv').set_index('domain').loc[domains]
    ref = rec.reference_mixture.to_numpy(); proposed = rec.recommended_mixture.to_numpy()
    assert np.allclose(ref,xraw.mean(0),atol=1e-14)
    weights = ref[[domains.index(d) for d in ld]]; weights /= weights.sum()
    saved = pd.read_csv(REPO/'求解/问题一/结果/非线性代理对照_预测明细.csv')
    ranking = []
    for (scope,model), rows in saved.groupby(['scope','model']):
        obs = rows.pivot(index='index',columns='loss_domain',values='observed')[ld]
        pred = rows.pivot(index='index',columns='loss_domain',values='predicted')[ld]
        ov, pv = obs.to_numpy()@weights,pred.to_numpy()@weights
        pick = int(np.argmin(pv)); ranks=pd.Series(ov).rank(method='min').to_numpy()
        ranking.append(dict(scope=scope,model=model,n=len(ov),spearman=float(spearmanr(ov,pv).statistic),
                            selected_index=int(obs.index[pick]),selected_observed_rank=int(ranks[pick]),
                            selected_observed=float(ov[pick]),best_observed=float(ov.min()),
                            selection_regret=float(ov[pick]-ov.min())))
    pd.DataFrame(ranking).to_csv(OUT/'q1_same_objective_ranking.csv',index=False,encoding='utf-8-sig')
    manifest=json.loads((REPO/'求解/问题一/结果/非线性代理对照_复现清单.json').read_text('utf-8-sig'))
    g,l=manifest['selected_gamma'],manifest['selected_lambda']
    kernel=lambda u,v:np.exp(-g*cdist(np.sqrt(u),np.sqrt(v),'sqeuclidean'))
    k=kernel(x,x);means=k.mean(0);grand=k.mean();kc=k-means[None,:]-means[:,None]+grand
    coef=solve(kc+l*np.eye(len(x)),y-y.mean(0),assume_a='pos')
    def kpredict(v):
        v=v/v.sum(1,keepdims=True); kv=kernel(v,x)
        return (kv-kv.mean(1,keepdims=True)-means[None,:]+grand)@coef+y.mean(0)
    tm=pd.read_csv(a_dir/'test_mixture_1m.csv').sort_values('index')
    check=kpredict(tm[mc].to_numpy(float))
    saved_k=saved[(saved.scope=='1m')&(saved.model=='sqrt_krr')].pivot(index='index',columns='loss_domain',values='predicted')[ld]
    assert np.array_equal(tm['index'],saved_k.index)
    rebuild_error=float(np.max(np.abs(check-saved_k.to_numpy())))
    A=pd.read_csv(REPO/'求解/问题一/结果/混合系数矩阵.csv').set_index('training_domain').loc[domains,ld].to_numpy()
    b=pd.read_csv(REPO/'求解/问题一/结果/截距.csv').set_index('loss_domain').loc[ld,'intercept'].to_numpy()
    targets=np.stack([ref,proposed]); linear=(targets@A+b)@weights; nonlinear=kpredict(targets)@weights
    # Convex combination feasibility, then nearest Linf distance as a numerical certificate.
    hull={}
    for name,target in [('reference_closed',ref/ref.sum()),('recommended',proposed)]:
        eq=np.vstack([x.T,np.ones(len(x))]);rhs=np.r_[target,1]
        fit=linprog(np.zeros(len(x)),A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs')
        upper=np.vstack([np.column_stack([x.T,-np.ones(len(domains))]),np.column_stack([-x.T,-np.ones(len(domains))])])
        dist=linprog(np.r_[np.zeros(len(x)),1.],A_ub=upper,b_ub=np.r_[target,-target],
                     A_eq=np.array([np.r_[np.ones(len(x)),0.]]),b_eq=[1.],bounds=(0,None),method='highs')
        hull[name]=dict(feasible=bool(fit.success),status=int(fit.status),minimum_linf_distance=float(dist.fun))
    report['q1']=dict(weights=dict(zip(ld,weights)),same_objective=ranking,krr_rebuild_max_abs_error=rebuild_error,
                      reference_and_recommended=dict(linear=linear.tolist(),sqrt_krr=nonlinear.tolist(),
                      linear_change=float(linear[1]-linear[0]),krr_change=float(nonlinear[1]-nonlinear[0])),hull=hull)
    # 同一上限与固定未观测领域下，检查正则改变对系数和配方的影响。
    measured = [domains.index(d) for d in ld]
    fixed_idx = [i for i in range(len(domains)) if i not in measured]
    def fit_linear(alpha):
        xc=xraw-xraw.mean(axis=0);yc=y-y.mean(axis=0)
        co=solve(xc.T@xc+alpha*np.eye(xraw.shape[1]),xc.T@yc,assume_a='pos')
        return co,y.mean(axis=0)-xraw.mean(axis=0)@co
    def recommend(co, intercept):
        def build(z):
            v=ref.copy();v[measured]=z
            return v/v.sum()
        # 等式约束使配方和为 1，目标为线性；用 LP 避免近乎平坦目标的提前停止。
        sol=linprog((co@weights)[measured],A_eq=np.ones((1,len(measured))),
                    b_eq=[1-ref[fixed_idx].sum()],bounds=[(0,2.5*ref[i]) for i in measured],method='highs')
        if not sol.success:raise RuntimeError(sol.message)
        mixture=build(sol.x)
        return mixture,float((mixture@co+intercept)@weights)
    co0,b0=fit_linear(.001);co1,b1=fit_linear(.1)
    pp0,v0=recommend(co0,b0);pp1,v1=recommend(co1,b1)
    cc0=co0-co0.mean(axis=0);cc1=co1-co1.mean(axis=0)
    report['q1']['regularization']=dict(centered_relative_change=float(np.linalg.norm(cc1-cc0)/np.linalg.norm(cc0)),
        sign_flips=int(np.sum(np.sign(cc1)!=np.sign(cc0))),domains=domains,
        mixture_lambda_001=pp0.tolist(),mixture_lambda_01=pp1.tolist(),
        objective_lambda_001=v0,objective_lambda_01=v1,
        both_recipes_under_original_model=[float((v@co0+b0)@weights) for v in [pp0,pp1]])
    full=pd.read_csv(REPO/'求解/问题一/结果/质量域评分_全量.csv')
    sample=pd.read_csv(REPO/'求解/问题一/结果/质量域评分_抽样与扩展.csv').query("source == 'A1'")
    report['q1']['github_shares']=dict(full=float(full.loc[full.domain.eq('github'),'n'].sum()/full.n.sum()),
                                      A1=float(sample.loc[sample.domain.eq('github'),'n'].sum()/sample.n.sum()))

    # Q4: reconstruct filters, time encoding and frontier from raw tables, no solver imports.
    cdir=DATA/'C_efficiency_evolution';lb=pd.read_csv(cdir/'leaderboard_cleaned.csv')
    lb['P']=pd.to_numeric(lb['#Params (B)'],errors='coerce');lb['date']=pd.to_datetime(lb['Submission Date'],errors='coerce')
    lb['Year']=lb.date.dt.year;lb['avg']=pd.to_numeric(lb['Average \u2b06\ufe0f'],errors='coerce')
    tasks=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']
    lb[tasks]=lb[tasks].apply(pd.to_numeric,errors='coerce')
    licenses={'apache-2.0','mit','gpl-3.0','cc-by-4.0','mpl-2.0','bsd-3-clause','llama2','llama3','llama3.1','llama3.2','llama3.3','gemma','cc-by-sa-4.0','openrail'}
    lb=lb[lb[tasks].notna().all(axis=1)&lb.date.notna()&lb.P.notna()&lb['Hub License'].astype(str).str.strip().str.lower().isin(licenses)].copy()
    lb['logP']=np.log10(lb.P);lb['t']=lb.Year-2019;lb=lb[lb.t.between(0,12)].copy()
    sample=lb.groupby('Model').Year.nunique();mean_dates=lb.groupby('Year').date.mean()
    def regress(t):
        X=np.column_stack([np.ones(len(lb)),lb.logP,t]);coef=np.linalg.lstsq(X,lb.avg,rcond=None)[0]
        return dict(intercept=float(coef[0]),scale=float(coef[1]),time=float(coef[2]),r2=float(1-((lb.avg-X@coef)**2).sum()/((lb.avg-lb.avg.mean())**2).sum()))
    annual=regress(lb.t);continuous=regress((lb.date-pd.Timestamp('2019-01-01')).dt.total_seconds()/(365.25*86400))
    ts=pd.read_csv(cdir/'leaderboard_extended_timeseries.csv')
    ts['avg']=pd.to_numeric(ts.Average,errors='coerce');ts['P']=pd.to_numeric(ts.Params_B,errors='coerce')
    six=['IFEval','BBH','MATH_Lvl5','GPQA','MUSR','MMLU_PRO'];ts[six]=ts[six].apply(pd.to_numeric,errors='coerce')
    ts=ts[((ts.avg-ts[six].mean(axis=1)).abs()<=5)&ts.P.notna()&ts.avg.notna()&(ts.P>0)].copy()
    fy=ts.groupby('Year').agg(avg=('avg','max'),P=('P','max')).reset_index();fy['cum_avg']=fy.avg.cummax()
    monthly=lb.groupby(lb.date.dt.to_period('M')).avg.apply(lambda z:z.nlargest(max(1,int(len(z)*.01))).mean())
    monthly=monthly[monthly.index>=pd.Period('2024-01')]
    mt=np.array([(p.year-2019)+(p.month-.5)/12 for p in monthly.index]);hist=fy[fy.Year<=2023]
    tf=np.r_[hist.Year.to_numpy()-2019+.5,mt];sf=np.r_[hist.cum_avg.to_numpy(),np.maximum.accumulate(monthly.to_numpy())]
    idx=np.argsort(tf);tf=tf[idx];sf=np.maximum.accumulate(sf[idx]);future=np.array([mt.max()+1,mt.max()+2])
    def logistic(t,K,r,t0):return K/(1+np.exp(-r*(np.asarray(t)-t0)))
    def lfit(t,s):return curve_fit(logistic,t,s,p0=[55,.8,3.5],bounds=([20,.01,.5],[100,5,8]),maxfev=40000)[0]
    full=lfit(tf,sf);short=lfit(tf[:-3],sf[:-3])
    split=2024-2019+(9-.5)/12;train=tf<=split;test=~train;hold=lfit(tf[train],sf[train])
    rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    pd.DataFrame(dict(t=tf,score=sf,held_out=test)).to_csv(OUT/'q4_reconstructed_frontier.csv',index=False,encoding='utf-8-sig')
    annual_models=[]
    for year,group in ts.groupby('Year'):
        peak=group.loc[group.avg.idxmax()];big=group.loc[group.P.idxmax()]
        annual_models.append(dict(year=int(year),score_peak_model=peak.Model,score_peak=float(peak.avg),score_peak_parameters=float(peak.P),max_size_model=big.Model,max_size=float(big.P),max_size_model_score=float(big.avg)))
    report['q4']=dict(records=len(lb),unique_model_names=int(lb.Model.nunique()),names_across_years=int((sample>1).sum()),
        year_mean_dates={str(k):str(v) for k,v in mean_dates.items()},mean_date_gap_years=float((mean_dates.iloc[-1]-mean_dates.iloc[0]).total_seconds()/(365.25*86400)),
        annual_label_regression=annual,continuous_date_regression=continuous,annual_frontier_subjects=annual_models,
        frontier=dict(n=len(tf),historical_n=len(hist),monthly_n=len(mt),peak_repeated_points=int((sf==sf.max()).sum()),parameters=full.tolist(),prediction=logistic(future,*full).tolist(),
        truncated_last_3_parameters=short.tolist(),truncated_last_3_prediction=logistic(future,*short).tolist(),
        holdout_cutoff='2024-09',holdout_n=int(test.sum()),holdout_logistic_parameters=hold.tolist(),
        holdout_logistic_rmse=rmse(logistic(tf[test],*hold),sf[test]),holdout_persistence_rmse=rmse(np.full(test.sum(),sf[train][-1]),sf[test])))
    report['official_results_unchanged']=all(Path(p).exists() and digest(Path(p))==v for p,v in before.items())
    assert report['official_results_unchanged']
    (OUT/'independent_q1_q4_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def check_q2_q3(REPO, DATA, OUT):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):sha(p) for p in (REPO/'求解').rglob('*') if p.is_file() and p.suffix in ('.csv','.json')}
    bdir=DATA/'B_scaling_laws';b1=pd.read_csv(bdir/'pythia_training_log_existing.csv');b6=pd.read_csv(bdir/'supplementary_NQ_experiment.csv')
    p0=pd.read_csv(REPO/'求解/问题二/结果/经典标度律参数.csv').iloc[0][['E','A','alpha','B','beta']].to_numpy(float)
    p=pd.read_csv(REPO/'求解/广义标度律参数.csv').set_index('param').loc[['E','A','alpha','B','beta','C','gamma'],'value'].to_numpy(float)
    classic=lambda p,N,D:p[0]+p[1]*N**(-p[2])+p[3]*D**(-p[4])
    general=lambda p,N,D,Q:classic(p[:5],N,D)+p[5]*np.maximum(1-Q,0)**p[6]
    def metrics(y,yh):
        e=np.asarray(yh)-np.asarray(y)
        return dict(n=len(e),R2=float(1-(e*e).sum()/((y-np.mean(y))**2).sum()),RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(np.mean(abs(e))),MAPE_percent=float(np.mean(abs(e)/abs(y))*100))
    N=np.r_[b1.N_params_B,b6.N_params_B];D=np.r_[b1.D_tokens_B,b6.D_tokens_B];Q=np.r_[np.ones(len(b1)),b6.Q_score];Y=np.r_[b1.val_loss,b6.val_loss]
    errors=dict(classic_B1=metrics(b1.val_loss.to_numpy(),classic(p0,b1.N_params_B.to_numpy(),b1.D_tokens_B.to_numpy())),general_B1_B6=metrics(Y,general(p,N,D,Q)))
    groups=[]
    for (n,d),g in b6.groupby(['N_params_B','D_tokens_B']):
        low=g[np.isclose(g.Q_score,.1)];high=g[np.isclose(g.Q_score,1)]
        assert len(low)==len(high)==1
        groups.append(dict(N=n,D=d,delta=float(low.val_loss.iloc[0]-high.val_loss.iloc[0])))
    groups=pd.DataFrame(groups);groups.to_csv(OUT/'q2_fixed_ND_quality_gain.csv',index=False,encoding='utf-8-sig')
    gain=dict(groups=len(groups),min=float(groups.delta.min()),max=float(groups.delta.max()),spearman_with_N=float(spearmanr(groups.N,groups.delta).statistic),additive_model_gain=float(p[5]*(.9**p[6])))
    # Fixed gamma=1 retains identical data, loss, bounds, and six remaining parameters.
    fixed=least_squares(lambda v:general(np.r_[v,1.],N,D,Q)-Y,p[:6],bounds=([0,1e-8,.01,1e-8,.01,1e-8],[5,1e3,2,1e3,2,50]),loss='soft_l1',f_scale=.05,max_nfev=80000)
    pg=np.r_[fixed.x,1.]
    soft=lambda e:float(.05**2*np.sum(np.sqrt(1+(e/.05)**2)-1))
    base_cost=soft(general(p,N,D,Q)-Y);fixed_cost=soft(general(pg,N,D,Q)-Y)
    Q0=json.loads((REPO/'求解/问题一_关键量.json').read_text('utf-8-sig'))['Q0_mix_weighted'];k=6+2e-4*2048
    def optimum(budget,par):
        E,A,a,B,b,C,g=par;bc=budget/1e18
        def atq(q):
            h=5*(q**4-Q0**4)
            def deriv(logn):
                n=np.exp(logn)
                return -a*A*n**(-a-1)+b*B*bc**(-b)*k*(k*n+h)**(b-1)
            ln=brentq(deriv,np.log(1e-4),np.log(1e6),xtol=1e-13)
            n=np.exp(ln);d=bc/(k*n+h)
            return dict(Q=float(q),N_B=float(n),D_B=float(d),loss=float(E+A*n**(-a)+B*d**(-b)+C*(1-q)**g))
        grid=np.linspace(Q0,1,61);vals=[atq(q) for q in grid];cand=[vals[0],vals[-1]]
        for i in range(1,len(vals)-1):
            if vals[i]['loss']<=vals[i-1]['loss'] and vals[i]['loss']<=vals[i+1]['loss']:
                sol=minimize_scalar(lambda q:atq(q)['loss'],bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-13})
                cand.append(atq(sol.x))
        return min(cand,key=lambda x:x['loss'])
    sweep=pd.read_csv(REPO/'求解/问题三/结果/预算扫略.csv');sat=sweep[sweep.Q>=1-1e-8].iloc[0];budget=float(sat.C)
    base_sat=optimum(budget,p);fixed_sat=optimum(budget,pg)
    base22=optimum(1e22,p)
    mix=pd.read_csv(REPO/'求解/问题三/结果/配比项算力等价.csv');M=float(mix.loc[mix.mixture=='问题一有界推荐','M_p'].iloc[0])
    transfer=[]
    for fraction in [0,.25,.5,.75,1]:
        target=base22['loss']+fraction*M
        root=brentq(lambda logc:optimum(np.exp(logc),p)['loss']-target,np.log(1e21),np.log(1e25),xtol=1e-11)
        transfer.append(dict(retention=fraction,target_loss=target,equivalent_budget=float(np.exp(root)),multiplier=float(np.exp(root)/1e22)))
    # C3 sample count uses the exact documented six-score consistency filter.
    ts=pd.read_csv(DATA/'C_efficiency_evolution/leaderboard_extended_timeseries.csv');cols=['IFEval','BBH','MATH_Lvl5','GPQA','MUSR','MMLU_PRO']
    ts['avg']=pd.to_numeric(ts.Average,errors='coerce');ts['P']=pd.to_numeric(ts.Params_B,errors='coerce');ts[cols]=ts[cols].apply(pd.to_numeric,errors='coerce')
    mask=((ts.avg-ts[cols].mean(axis=1)).abs()<=5)&ts.P.notna()&ts.avg.notna()&(ts.P>0)
    c3=ts[mask].copy();X=np.column_stack([np.ones(len(c3)),np.log10(c3.P),c3.Year-2019]);c=np.linalg.lstsq(X,c3.avg,rcond=None)[0]
    c3_summary=dict(raw_n=len(ts),retained_n=len(c3),excluded_n=int((~mask).sum()),R2=float(1-((c3.avg-X@c)**2).sum()/((c3.avg-c3.avg.mean())**2).sum()))
    # Three cutoffs all evaluate their subsequent months through the same last month.
    front=pd.read_csv(OUT/'q4_reconstructed_frontier.csv');t=front.t.to_numpy();s=front.score.to_numpy()
    logistic=lambda t,K,r,t0:K/(1+np.exp(-r*(np.asarray(t)-t0)))
    cuts=[]
    for month in [7,8,9]:
        cutoff=5+(month-.5)/12;tr=t<=cutoff+1e-12;te=~tr
        theta=curve_fit(logistic,t[tr],s[tr],p0=[55,.8,3.5],bounds=([20,.01,.5],[100,5,8]),maxfev=40000)[0]
        pred=logistic(t[te],*theta);baseline=np.full(te.sum(),s[tr][-1]);rmse=lambda p:float(np.sqrt(np.mean((p-s[te])**2)))
        cuts.append(dict(cutoff=f'2024-{month:02}',train_n=int(tr.sum()),test_n=int(te.sum()),test_end='2025-03',parameters=theta.tolist(),
                         logistic_RMSE=rmse(pred),persistence_RMSE=rmse(baseline),test_time=t[te].tolist(),test_observed=s[te].tolist(),logistic_prediction=pred.tolist(),persistence_prediction=baseline.tolist()))
    result=dict(error_metrics=errors,B6_quality_gain=gain,gamma_fixed_one=dict(parameters=pg.tolist(),original_cost=base_cost,fixed_cost=fixed_cost,
        cost_increase_percent=(fixed_cost/base_cost-1)*100,original_first_saturation_budget=budget,original_saved_Q=float(sat.Q),original_recomputed=base_sat,fixed_gamma_recomputed=fixed_sat),
        transfer_sensitivity=transfer,C3=c3_summary,Q4_time_holdouts=cuts,official_results_unchanged=all(Path(f).exists() and sha(Path(f))==v for f,v in before.items()))
    # 保持 A 侧费用不变，只改变未知的 A -> B 质量映射；不是纯坐标替换。
    E,A,a,B,b,C,g=p
    mapping=[]
    for power in [.5,1.,2.]:
        def local_derivative(logbudget):
            bc=np.exp(logbudget)/1e18
            n=(a*A/(b*B)*(bc/k)**b)**(1/(a+b));d=bc/(k*n)
            return b*B*d**(-b)*20*Q0**3/(k*n)-C*g*(1-Q0**power)**(g-1)*power*Q0**(power-1)
        boundary=np.exp(brentq(local_derivative,np.log(1e17),np.log(1e22)))
        mapping.append(dict(power=power,local_start_budget=float(boundary)))
    result['quality_mapping_local_sensitivity']=mapping
    assert result['official_results_unchanged']
    (OUT/'independent_q23_and_timecuts_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=repo/'复现记录/补充检验')
    args = ap.parse_args()
    data = args.data_dir.resolve()
    out = args.out.resolve()
    if out.is_relative_to(repo/'求解') or out.is_relative_to(data):
        raise ValueError('诊断输出目录不得位于正式求解或附件目录内')
    out.mkdir(parents=True, exist_ok=True)
    first = check_q1_q4(repo, data, out)
    second = check_q2_q3(repo, data, out)
    print(json.dumps({'output': str(out), 'q1_krr_rebuild_error': first['q1']['krr_rebuild_max_abs_error'],
                     'C3_retained_n': second['C3']['retained_n'], 'official_results_unchanged':
                     first['official_results_unchanged'] and second['official_results_unchanged']},
                    ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
