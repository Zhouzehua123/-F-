# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6-Luna；机构：OpenAI；版本发布日期：2026-07-09。
# 本程序由 OpenAI Codex（GPT-5.6-Luna）辅助实现；仅从保存的验证结果绘图。
"""核岭增强的证据图：逐域精度及逐配方平均损失；不重新拟合。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from 展示风格公共 import PURPLE, TEAL, configure
from 绘图数据 import DOMAIN_NAMES
from 出版导出 import export_figure


def plot_nonlinear(data, out):
    table = data.nonlinear_domains
    table = table[table.scope.eq('1m')]
    a = table[table.model.eq('ridge')].set_index('loss_domain')
    b = table[table.model.eq('sqrt_krr')].set_index('loss_domain').loc[a.index]
    pred = data.nonlinear_predictions
    pred = pred[pred.scope.eq('1m') & pred.model.eq('sqrt_krr')]
    p = pred.groupby('index')[['observed','predicted']].mean()
    assert len(p) == 256 and len(a) == 13 and len(pred) == 3328
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(5.92, 3.65), gridspec_kw={'width_ratios':[1.1,1]})
    ax = axes[0]; y = np.arange(len(a))
    for i in y:
        ax.plot([a.R2.iloc[i], b.R2.iloc[i]], [i,i], color='#CAD1D9', lw=1.4)
    ax.scatter(a.R2,y,s=24,marker='o',color=PURPLE,label='线性岭')
    ax.scatter(b.R2,y,s=29,marker='D',color=TEAL,label='平方根核岭')
    ax.set(yticks=y, yticklabels=[DOMAIN_NAMES[d] for d in a.index], xlim=(0,1.025),
           xlabel='逐域 R²', title='(a) 同尺度预测精度')
    ax.invert_yaxis(); ax.grid(axis='x'); ax.tick_params(labelsize=8)
    ax.legend(loc='lower left', bbox_to_anchor=(-.04,-.34), ncol=2, fontsize=8,
              handletextpad=.3, columnspacing=.9)
    ax=axes[1]
    ax.scatter(p.observed,p.predicted,s=9,alpha=.6,color=TEAL,edgecolors='none')
    lim=[min(p.min())-.08,max(p.max())+.08]
    ax.plot(lim,lim,color='#79818C',ls='--',lw=.9)
    ax.set(xlim=lim,ylim=lim,xlabel='观测平均损失',ylabel='预测平均损失',
           title='(b) 256 组检验配方')
    ax.set_aspect('equal', adjustable='box');ax.grid()
    rmse=np.sqrt(np.mean((p.observed-p.predicted)**2))
    ax.text(.04,.95,f'配方均值 RMSE = {rmse:.4f}',transform=ax.transAxes,va='top',fontsize=8)
    for ax in axes:
        ax.title.set_fontsize(9); ax.xaxis.label.set_fontsize(9);ax.yaxis.label.set_fontsize(9)
    fig.subplots_adjust(left=.14,right=.99,bottom=.22,top=.91,wspace=.42)
    export_figure(fig,str(Path(out)/'图9_非线性代理对照'),formats=['png','svg'],
                  dpi=400,size_inches=(5.92,3.65),tight=False)
    # 统一矢量文本行尾并去除导出时钟，保持版本对照稳定。
    svg = Path(out)/'图9_非线性代理对照.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()
                             if '<dc:date>' not in line)+'\n', encoding='utf-8')
    plt.close(fig)
