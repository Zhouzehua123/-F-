# 本程序使用 OpenAI Codex 辅助实现与数值核对。
# 用途：沿用已选核岭，以训练配方凸组合限制推荐范围；不重新搜索超参数。
"""python 求解/问题一/核岭支持域推荐.py --data-dir <附件根目录>
推荐写入原结果目录；--output-dir 可用于隔离核验。
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import linprog, minimize
from 非线性代理对照 import find_data, load_pair, closed, kernel, krr_predict

HERE=Path(__file__).resolve().parent


def recommend(data, result_dir, output_dir):
    raw,y,indices,columns,loss_domains,_=load_pair(data,'train','1m')
    domains=[s.removeprefix('train_the_pile_') for s in columns]
    x=closed(raw);ref=raw.mean(axis=0);ref=ref/ref.sum()
    measured=[domains.index(d) for d in loss_domains]
    fixed=[i for i in range(len(domains)) if i not in measured]
    weights=ref[measured];weights=weights/weights.sum()
    config=json.loads((result_dir/'非线性代理对照_复现清单.json').read_text('utf-8-sig'))
    gamma,lam=config['selected_gamma'],config['selected_lambda']
    k=kernel(x,x,gamma);means=k.mean(axis=0);grand=k.mean()
    kc=k-means[None,:]-means[:,None]+grand
    beta=solve(kc+lam*np.eye(len(x)),y-y.mean(axis=0),assume_a='pos')@weights
    offset=float(y.mean(axis=0)@weights-means@beta+grand*beta.sum())
    beta=beta-beta.mean()
    sx=np.sqrt(x)
    def value_gradient(z):
        p=z@x;sp=np.sqrt(np.maximum(p,1e-30))
        kval=np.exp(-gamma*np.sum((sx-sp)**2,axis=1))
        v=kval*beta
        gp=-gamma*(v.sum()-(v@sx)/sp)
        return float(v.sum()+offset),x@gp
    # 全部约束在凸组合权重上为线性；四个无验证域固定闭合参考占比。
    eq=np.vstack([np.ones(len(x)),x[:,fixed].T]);rhs=np.r_[1.,ref[fixed]]
    cap=2.5*ref[measured];upper=x[:,measured].T
    init=linprog(np.zeros(len(x)),A_eq=eq,b_eq=rhs,A_ub=upper,b_ub=cap,bounds=(0,None),method='highs')
    if not init.success:raise RuntimeError('训练支持域与配比约束无共同可行点')
    starts=[init.x]
    rng=np.random.default_rng(20260927)
    for _ in range(3):
        vertex=linprog(rng.normal(size=len(x)),A_eq=eq,b_eq=rhs,A_ub=upper,b_ub=cap,
                       bounds=(0,None),method='highs')
        if vertex.success:starts.append(vertex.x)
    fits=[]
    for j,start in enumerate(starts):
        z=start.copy();vertices=[z.copy()];combination=np.ones(1)
        # 全校正条件梯度：在少量可行顶点的凸组合上求解，约束始终可行。
        for iteration in range(150):
            value,gradient=value_gradient(z)
            vertex=linprog(gradient,A_eq=eq,b_eq=rhs,A_ub=upper,b_ub=cap,
                           bounds=(0,None),method='highs')
            if not vertex.success:raise RuntimeError(vertex.message)
            direction=vertex.x-z;gap=float(-gradient@direction)
            if gap<1e-6:break
            vertices.append(vertex.x);basis=np.stack(vertices)
            def reduced(c):
                val,grad=value_gradient(c@basis)
                return val,basis@grad
            fit=minimize(reduced,np.r_[combination,0.],jac=True,method='SLSQP',
                         bounds=[(0,1)]*len(vertices),constraints={'type':'eq','fun':lambda c:c.sum()-1.,'jac':lambda c:np.ones_like(c)},
                         options={'maxiter':1000,'ftol':1e-13})
            combination=np.maximum(fit.x,0.);combination/=combination.sum()
            keep=combination>1e-10;combination=combination[keep];combination/=combination.sum()
            vertices=[v for v,yes in zip(vertices,keep) if yes];z=combination@np.stack(vertices)
        residual=max(float(np.max(abs(eq@z-rhs))),float(np.maximum(upper@z-cap,0).max()),float(max(0,-z.min())))
        fits.append(dict(start=j,success=bool(gap<1e-6),objective=value_gradient(z)[0],
                         constraint_residual=residual,iterations=iteration+1,stationarity_gap=gap,weights=z))
        print({k:v for k,v in fits[-1].items() if k!='weights'},flush=True)
    valid=[f for f in fits if f['success'] and f['constraint_residual']<1e-7]
    if not valid:raise RuntimeError('多初值推荐求解未通过可行性检查')
    best=min(valid,key=lambda f:f['objective']);z=best['weights'];p=z@x
    # 与正式核岭预测函数交叉核验目标值，并保存训练凸组合证书。
    scores=krr_predict(x,y,np.stack([ref/ref.sum(),p]),gamma,lam)@weights
    assert abs(scores[1]-best['objective'])<1e-7
    output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(domain=domains,reference_mixture=ref,recommended_mixture=p,adjustment=p-ref)).sort_values('adjustment',ascending=False).to_csv(output_dir/'推荐配比调整.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dict(index=indices,convex_weight=z)).to_csv(output_dir/'核岭推荐_训练凸组合.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{k:v for k,v in f.items() if k!='weights'} for f in fits]).to_csv(output_dir/'核岭推荐_多初值.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(mixture=name,predicted_weighted_loss=float(score),M_p=float(score-scores[0])) for name,score in zip(['参考配比','问题一有界推荐'],scores)]).to_csv(output_dir/'核岭推荐_目标对照.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(reference=float(scores[0]),recommended=float(scores[1]),change=float(scores[1]-scores[0]),active_training_rows=int(np.sum(z>1e-8)),runs=[{k:v for k,v in f.items() if k!='weights'} for f in fits]),ensure_ascii=False,indent=2))
    return p,scores


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--output-dir',type=Path,default=HERE/'结果')
    args=ap.parse_args()
    recommend(find_data(args.data_dir),HERE/'结果',args.output_dir)


if __name__=='__main__':main()
