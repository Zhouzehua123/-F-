"""问题二图表：读取既有参数及附件，不重新拟合或覆盖跨问共享结果。

运行：python 重绘图表.py
依赖：numpy pandas scipy matplotlib；字体：Windows 宋体、Times New Roman。
输出：原图片目录中的同名高清 PNG；结果目录中图表复核指标。
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BLUE, TEAL, RED, GRAY = '#245B86', '#2B8C91', '#B44D50', '#6B7785'
PALE, INK = '#EEF3F7', '#243545'
CMAP = LinearSegmentedColormap.from_list('effect', ['#245B86', '#FAFAF8', '#B44D50'])


def render_all(data_root=None):
    data_root = Path(data_root) if data_root else ROOT.parent / 'real_attachments'
    if not data_root.is_dir():
        data_root = ROOT / 'real_attachments'
    if not data_root.is_dir():
        raise FileNotFoundError('未找到 real_attachments，请使用 --data-dir 指定附件根目录。')
    figdir = HERE / '图片'
    figdir.mkdir(exist_ok=True)
    fontdir = Path('C:/Windows/Fonts')
    for name in ['simsun.ttc', 'times.ttf', 'timesi.ttf', 'timesbd.ttf']:
        if (fontdir / name).exists():
            fm.fontManager.addfont(str(fontdir / name))
    for family in ['SimSun', 'Times New Roman']:
        fm.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'], 'font.size': 11,
        'axes.titlesize': 11.5, 'axes.labelsize': 11, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'legend.fontsize': 9.5,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
        'axes.edgecolor': '#9AA8B3', 'axes.linewidth': .7,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': '#E0E7ED', 'grid.linewidth': .6,
        'legend.frameon': False, 'pdf.fonttype': 42,
        'savefig.dpi': 360, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    bdir = data_root / 'B_scaling_laws'
    read = lambda f: pd.read_csv(bdir / f)
    py = read('pythia_training_log_existing.csv')
    b6 = read('supplementary_NQ_experiment.csv')
    b7 = read('supplementary_NQ_experiment_expanded.csv')
    b7n = b7[~b7.experiment_id.isin(b6.experiment_id)]
    b8 = read('supplementary_NQ_experiment_large.csv')
    b8c = b8[b8.data_type.eq('calibrated')]
    par = pd.read_csv(HERE/'结果/广义标度律参数.csv').set_index('param').value
    E,A,a,B,b,C,g = [float(par[k]) for k in ['E','A','alpha','B','beta','C','gamma']]
    cl = pd.read_csv(HERE/'结果/经典标度律参数.csv').iloc[0]
    def classic(N,D): return cl.E+cl.A*np.asarray(N)**(-cl.alpha)+cl.B*np.asarray(D)**(-cl.beta)
    def model(N,D,Q): return E+A*np.asarray(N)**(-a)+B*np.asarray(D)**(-b)+C*(1-np.asarray(Q))**g
    audit = {'parameters_unchanged': par.to_dict(), 'validation': []}
    def save(fig, name):
        fig.savefig(figdir/f'{name}.png', bbox_inches='tight', pad_inches=.08)
        plt.close(fig)
    def axes_style(ax, axis='both'):
        ax.set_axisbelow(True); ax.grid(True, axis=axis)
    def note(ax, text, xy=(.04,.95)):
        ax.text(*xy, text, transform=ax.transAxes, ha='left', va='top', fontsize=9.5,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2.5))

    # Analysis map: uncalibrated cross-source links stay visibly conditional.
    fig,ax=plt.subplots(figsize=(7.2,3.05));ax.set(xlim=(0,10),ylim=(-.12,4));ax.axis('off')
    def box(x,y,w,h,title,body,color=BLUE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.08',
                     facecolor=PALE,edgecolor=color,linewidth=.85))
        ax.text(x+w/2,y+h-.22,title,ha='center',va='center',fontsize=11,color=color)
        ax.text(x+w/2,y+.30,body,ha='center',va='center',fontsize=9.5,linespacing=1.4)
    def arrow(x1,y1,x2,y2,dashed=False):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,
                     linewidth=1,color=GRAY,linestyle='--' if dashed else '-'))
    box(.1,2.8,2.6,.95,'规模信息 B1','真实训练轨迹\n经典标度律')
    box(3.7,2.8,2.6,.95,'质量信息 B6','半合成质量实验\n质量缺口项',TEAL)
    box(7.3,2.8,2.6,.95,'问题一输出','配比系数与参考配比\n领域效应结构',GRAY)
    box(2.1,1.2,5.8,1.0,'广义标度关系',r'$L=E+AN^{-\alpha}+BD^{-\beta}+C(1-Q)^{\gamma}$')
    arrow(1.4,2.77,3.2,2.23);arrow(5,2.77,5,2.23);arrow(8.6,2.77,7,2.23,True)
    ax.text(8.2,2.37,'迁移假设',fontsize=9.5,ha='center',color=GRAY)
    box(.6,.02,4.0,.77,'多来源检验','B2—B5、B7、B8、B10',GRAY)
    box(5.4,.02,4.0,.77,'资源配置解释','边际效用 · 弹性 · 等损失关系',TEAL)
    arrow(3.6,1.17,2.6,.83);arrow(6.4,1.17,7.4,.83)
    save(fig,'图0_问题二分析框架')

    # 1. Checkpoint trajectories and residuals use the same N,D per observation.
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.15),layout='constrained')
    ns=sorted(py.N_params_B.unique());colors=[BLUE,TEAL,'#7294B0',GRAY]
    for n,col in zip([ns[i] for i in [0,3,5,7]],colors):
        sub=py[py.N_params_B.eq(n)].sort_values('D_tokens_B')
        axs[0].plot(sub.D_tokens_B,classic(n,sub.D_tokens_B),color=col,lw=1.65,label=f'{n:.2f} B')
        sub=sub.iloc[::9]
        axs[0].scatter(sub.D_tokens_B,sub.val_loss,s=17,facecolor='white',edgecolor=col,lw=.7,zorder=3)
    axs[0].set(xscale='log',xlabel='训练数据量 D / B tokens',ylabel='验证损失',title='(a) 代表模型的训练轨迹')
    axs[0].legend(ncol=2,loc='upper right',title='参数量 N',title_fontsize=9.5)
    note(axs[0],'空心点：附件记录\n实线：拟合曲线',(.04,.30))
    pred=classic(py.N_params_B,py.D_tokens_B);err=pred-py.val_loss
    axs[1].scatter(py.D_tokens_B,np.asarray(err)*1e4,s=7,alpha=.45,color=BLUE,edgecolors='none',rasterized=True)
    axs[1].axhline(0,color=RED,lw=1)
    axs[1].set(xscale='log',xlabel='训练数据量 D / B tokens',ylabel='预测残差 / 10⁻⁴',title='(b) 全部 1,176 个检查点残差')
    note(axs[1],f'RMSE = {np.sqrt(np.mean(err**2)):.2e}')
    for ax in axs:axes_style(ax)
    save(fig,'图1_经典标度律拟合')

    # 2. Six sources, six panels. Given values are not uniformly observations.
    sets=[('B4  跨族收敛点','真实',read('scaling_baseline.csv'),1.,BLUE),
          ('B5  文献标度数据','真实',read('published_scaling_data.csv'),1.,BLUE),
          ('B7  新增质量点','半合成',b7n,b7n.Q_score,TEAL),
          ('B2  Cerebras 轨迹','半合成',read('cerebras_training_log.csv'),1.,TEAL),
          ('B8  校准段反向变换','半合成',b8c,1-b8c.Q_score,TEAL),
          ('B10  大规模损失','估算',read('supplementary_large_baseline.csv'),1.,GRAY)]
    fig,axs=plt.subplots(2,3,figsize=(7.2,5.7),layout='constrained')
    for ax,(name,kind,df,q,col) in zip(axs.flat,sets):
        y=df.val_loss.to_numpy();pred=model(df.N_params_B,df.D_tokens_B,q)
        r=float(np.corrcoef(y,pred)[0,1]);rmse=float(np.sqrt(np.mean((y-pred)**2)))
        mape=float(np.mean(abs(y-pred)/abs(y))*100)
        audit['validation'].append(dict(dataset=name,n=len(df),r=r,RMSE=rmse,MAPE=mape))
        lo=min(y.min(),pred.min());hi=max(y.max(),pred.max());pad=(hi-lo)*.09
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],ls='--',color=GRAY,lw=.9,zorder=1)
        ax.scatter(y,pred,s=14 if len(df)<150 else 5,color=col,alpha=.65,edgecolors='none',rasterized=True)
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='给定损失',ylabel='预测损失',title=f'{name}\n{kind} · n={len(df):,}')
        ax.set_aspect('equal',adjustable='box');axes_style(ax)
        note(ax,f'r = {r:.3f}\nMAPE = {mape:.1f}%')
    save(fig,'图2_广义标度律验证')

    # 3. Quality response and the conditional asymptotic floor.
    q=np.linspace(.1,1,250)
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    for n,col in zip([.41,2.8,12],[BLUE,TEAL,RED]):
        axs[0].plot(q,model(n,300,q),lw=1.8,color=col,label=f'N = {n:g} B')
    axs[0].set(xlabel='数据质量 Q',ylabel='预测损失',title='(a) 固定数据量 D = 300 B tokens')
    axs[0].legend(loc='upper right');axs[0].set_xlim(.1,1)
    floor=E+C*(1-q)**g
    axs[1].fill_between(q,E,floor,color=TEAL,alpha=.14,label='质量缺口项')
    axs[1].plot(q,floor,color=TEAL,lw=1.8,label='模型渐近损失')
    axs[1].axhline(E,color=GRAY,ls='--',lw=1,label=f'E = {E:.3f}')
    for q0 in [.3,.6,1.]:
        f0=E+C*(1-q0)**g
        axs[1].scatter(q0,f0,color=TEAL,s=23,zorder=3)
        axs[1].annotate(f'{f0:.3f}',(q0,f0),xytext=(-9,8 if q0!=1 else 12),textcoords='offset points',fontsize=9.5,ha='center')
    axs[1].set(xlabel='数据质量 Q',ylabel='渐近损失',title='(b) 质量缺口与损失下限',xlim=(.1,1.03),ylim=(E-.04,2.03))
    axs[1].legend(loc='upper right',fontsize=9)
    for ax in axs:axes_style(ax)
    save(fig,'图3_质量边际影响')

    # 4. Derivatives have different units; elasticities enable a dimensionless comparison.
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),layout='constrained')
    ng=np.geomspace(.07,12,200);qm=np.linspace(.1,.99,200)
    axs[0].plot(ng,a*A*ng**(-a-1),color=BLUE,lw=1.8)
    axs[0].set(xscale='log',yscale='log',xlabel='参数量 N / B',ylabel=r'$|\partial L/\partial N|$ / B$^{-1}$',title='(a) 参数边际效应')
    axs[1].plot(qm,C*g*(1-qm)**(g-1),color=TEAL,lw=1.8)
    axs[1].set(xlabel='数据质量 Q',ylabel=r'$|\partial L/\partial Q|$',title='(b) 质量边际效应',xlim=(.1,1))
    note(axs[1],r'$Q\leq 0.99$')
    el=pd.read_csv(HERE/'结果/弹性与替代关系.csv')
    x=np.arange(3)
    for i,col in enumerate([BLUE,TEAL]):
        vals=-el.iloc[i][['弹性e_N','弹性e_D','弹性e_Q']].to_numpy(float)
        axs[2].bar(x+(i-.5)*.34,vals,.31,color=col,label=f'N={el.iloc[i,0]:g}, D={el.iloc[i,1]:g}')
    axs[2].set(xticks=x,xticklabels=['参数 N','数据 D','质量 Q'],ylabel='损失弹性绝对值',title='(c) 无量纲弹性',ylim=(0,.135))
    axs[2].legend(loc='upper left',fontsize=8.5,title='N、D 以 B 为单位；Q=0.6',title_fontsize=8)
    for ax in axs:axes_style(ax,'y')
    save(fig,'图4_边际效用对比')

    # 5. Keep original targets, limit the display range, and distinguish scale extrapolation.
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),layout='constrained')
    q=np.linspace(.1,1,600)
    for target,col in zip([2.30,2.10,1.95],[BLUE,TEAL,RED]):
        rhs=target-E-B*300**(-b)-C*(1-q)**g
        n=np.full_like(q,np.nan);ok=rhs>0;n[ok]=(A/rhs[ok])**(1/a)
        within=(n>=.07)&(n<=12);outside=(n>12)&(n<=1e4)
        axs[0].plot(q,np.where(within,n,np.nan),color=col,lw=2,label=f'L = {target:.2f}')
        axs[0].plot(q,np.where(outside,n,np.nan),color=col,lw=1.6,ls='--')
    axs[0].axhspan(12,1e4,color=GRAY,alpha=.08)
    axs[0].axhline(12,color=GRAY,lw=.8,ls=':')
    axs[0].text(.12,19,'超过主拟合参数范围',fontsize=9,color=GRAY)
    axs[0].set(xlim=(.1,1),ylim=(.07,1e4),yscale='log',xlabel='数据质量 Q',ylabel='所需参数量 N / B',title='(a) 等损失曲线，D = 300 B tokens')
    axs[0].legend(loc='upper right',fontsize=9)
    note(axs[0],'实线：主拟合参数范围\n虚线：规模外推情景',(.04,.98))
    # Local substitution rate at a fixed reference quality, not equal-cost benefit.
    ng=np.geomspace(.07,12,200);rate=C*g*.4**(g-1)/(a*A*ng**(-a-1))
    axs[1].plot(ng,rate,color=BLUE,lw=1.8)
    for n0 in [1,12]:
        val=C*g*.4**(g-1)/(a*A*n0**(-a-1))
        axs[1].scatter(n0,val,s=25,color=RED,zorder=3)
        axs[1].annotate(f'{val:.3f}',(n0,val),xytext=(-34,9),textcoords='offset points',fontsize=10)
    axs[1].set(xscale='log',yscale='log',xlabel='参考参数量 N / B',ylabel='局部等效增量比 / B',title='(b) Q = 0.6 时的局部替代率')
    note(axs[1],r'$R_{NQ}\propto N^{\alpha+1}$',(.05,.95))
    note(axs[1],'每单位质量的等效参数增量',(.05,.83))
    for ax in axs:axes_style(ax)
    save(fig,'图5_质量参数替代')

    # 6. Cluster on the already computed cosine matrix; no refit or causal interpretation.
    sim=pd.read_csv(HERE/'结果/领域替代互补矩阵.csv',index_col=0)
    dist=np.sqrt(np.maximum(0,2*(1-sim.to_numpy())));np.fill_diagonal(dist,0)
    order=leaves_list(linkage(squareform(dist,checks=False),method='average',optimal_ordering=True))
    sim=sim.iloc[order,order]
    names={'arxiv':'学术预印本','freelaw':'法律判例','nih_exporter':'科研项目','pubmed_central':'医学全文',
           'wikipedia_en':'百科','dm_mathematics':'数学','github':'代码','philpapers':'哲学论文',
           'stackexchange':'专业问答','enron_emails':'邮件','gutenberg_pg_19':'图书','pile_cc':'网页',
           'ubuntu_irc':'技术聊天','europarl':'议会记录','hackernews':'科技社区','pubmed_abstracts':'医学摘要','uspto_backgrounds':'专利背景'}
    labels=[names[x] for x in sim.index]
    fig,ax=plt.subplots(figsize=(7.2,5.5),layout='constrained')
    arr=np.ma.array(sim.to_numpy(),mask=np.triu(np.ones(sim.shape,dtype=bool),k=1))
    mesh=ax.pcolormesh(arr,cmap=CMAP,vmin=-1,vmax=1,edgecolors='white',linewidth=.4)
    ax.set(xticks=np.arange(17)+.5,yticks=np.arange(17)+.5,xlim=(0,17),ylim=(17,0))
    ax.set_xticklabels(labels,rotation=55,ha='right',rotation_mode='anchor',fontsize=10)
    ax.set_yticklabels(labels,fontsize=10);ax.tick_params(length=0)
    ax.set_aspect('equal');ax.spines[['left','bottom']].set_visible(False)
    for di,dj in [('wikipedia_en','pile_cc'),('pubmed_central','pubmed_abstracts'),('pubmed_central','enron_emails')]:
        i,j=sim.index.get_loc(di),sim.index.get_loc(dj)
        if i<j:i,j=j,i
        ax.text(j+.5,i+.5,f'{sim.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color='white')
    ax.text(.53,.90,'暖色：相对效应趋同\n冷色：相对效应相反\n\n按效应相似性排序',transform=ax.transAxes,
            ha='left',va='top',fontsize=11,linespacing=1.6,color=INK)
    cb=fig.colorbar(mesh,ax=ax,shrink=.75,pad=.025,aspect=28);cb.set_label('中心化系数向量的余弦相似度')
    save(fig,'图6_领域替代互补')

    # 7. Within-group centered observations make the direction diagnostic inspectable.
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    for ax,df,title,col in [(axs[0],b6,'(a) B6 质量实验',BLUE),(axs[1],b8c,'(b) B8 校准段',RED)]:
        groups=df.groupby(['N_params_B','D_tokens_B'])
        centered=df[['Q_score','val_loss']]-groups[['Q_score','val_loss']].transform('mean')
        cors=[sub.Q_score.corr(sub.val_loss) for _,sub in groups if sub.Q_score.nunique()>=3]
        ax.scatter(centered.Q_score,centered.val_loss,s=12,alpha=.35,color=col,edgecolors='none',rasterized=True)
        ax.axhline(0,color=GRAY,lw=.8,ls='--');ax.axvline(0,color=GRAY,lw=.8,ls='--')
        ax.set(xlabel='质量字段与组均值之差',ylabel='损失与组均值之差',title=title)
        note(ax,f'半合成 · n = {len(df):,}\n平均组内 r = {np.mean(cors):+.3f}')
        axes_style(ax)
        audit[title]={'n':len(df),'mean_within_group_r':float(np.mean(cors))}
    save(fig,'图7_质量方向对照')
    (HERE/'结果/图表复核指标.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('已更新原图片目录中的八张 PNG；模型参数未更改。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir');args=parser.parse_args()
    render_all(args.data_dir)
