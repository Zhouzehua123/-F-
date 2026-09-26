# 本绘图程序及代码在人工智能工具 Codex 辅助下完成。
# 开发机构：OpenAI；模型：GPT-5.6-Luna（2026-07-09）。
# 仅读取既有附件、参数和结果绘图，不重新拟合或修改结果数据。
"""问题二图表：读取既有参数及附件，不重新拟合或覆盖跨问共享结果。

运行：python 重绘图表.py
依赖：numpy pandas scipy matplotlib；字体：Windows 宋体、Times New Roman。
输出：原图片目录中的同名 PNG；不写入任何结果或参数文件。
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
from matplotlib.ticker import ScalarFormatter, NullLocator, NullFormatter
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BLUE, TEAL, RED, GRAY = '#665080', '#367E73', '#CB8B38', '#45414C'
PALE, INK = '#FAF8F4', '#302F3D'
CMAP = LinearSegmentedColormap.from_list('effect', ['#FAF3D8', '#D8B979', '#AD8792', '#6D527D', '#382F58'])


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
        'savefig.dpi': 400, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
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
    def save(fig, name):
        from matplotlib.text import Text
        # Dense automatic log minor ticks form a serrated edge at manuscript scale.
        # Retain the scale, limits, major ticks and all plotted values.
        for ax in fig.axes:
            for axis, scale in [(ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())]:
                if scale in ('log', 'symlog'):
                    axis.set_minor_locator(NullLocator())
                    axis.set_minor_formatter(NullFormatter())
        for text in fig.findobj(Text):
            if text.get_text().strip() and text.get_fontsize() < 11:
                text.set_fontsize(11)
        fig.savefig(figdir/f'{name}.png', bbox_inches='tight', pad_inches=.08)
        plt.close(fig)
    def axes_style(ax, axis='both'):
        ax.set_axisbelow(True); ax.grid(True, axis=axis)
    def note(ax, text, xy=(.04,.95)):
        ax.text(*xy, text, transform=ax.transAxes, ha='left', va='top', fontsize=9.5,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2.5))

    redesign(py,b6,b7,b7n,b8,b8c,read,model,classic,E,A,a,B,b,C,g,save)
    summarize_large_baseline(read, model)
    print('已更新原图片目录中的八张 PNG；模型参数未更改。')


def summarize_large_baseline(read, model):
    """只读复现正文 B9/B10 对照表，不改变模型或已有结果文件。"""
    metadata = read('supplementary_large_models.csv')
    losses = read('supplementary_large_baseline.csv')
    matched = losses.merge(metadata, left_on='family', right_on='model_name',
                           how='left', validate='one_to_one',
                           suffixes=('', '_meta'), indicator=True)
    assert matched['_merge'].eq('both').all()
    for field in ['N_params_B', 'D_tokens_B']:
        assert np.allclose(matched[field], matched[field + '_meta'])
    excluded = metadata[~metadata.model_name.isin(losses.family)]
    assert len(metadata) == 132 and len(matched) == 128
    assert len(excluded) == 4 and excluded.D_tokens_B.eq(0).all()
    prediction = model(matched.N_params_B, matched.D_tokens_B, 1.)
    matched['absolute_relative_error'] = 100 * abs(prediction - matched.val_loss) / matched.val_loss
    rows = []
    for low, high, label in [(100, 300, '[100,300)'), (300, 1000, '[300,1000)'),
                             (1000, np.inf, '[1000,10000]')]:
        group = matched[matched.N_params_B.ge(low) & matched.N_params_B.lt(high)]
        rows.append({'N / B': label, '记录数': len(group),
                     'D最小值': group.D_tokens_B.min(), 'D最大值': group.D_tokens_B.max(),
                     'MAPE_%': group.absolute_relative_error.mean()})
    assert sum(row['记录数'] for row in rows) == len(matched)
    assert matched.N_params_B.max() == 10000
    print('B9/B10 一对一匹配；MAPE 对照 B10 估算值，统一 Q=1：')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    return rows


def redesign(py,b6,b7,b7n,b8,b8c,read,model,classic,E,A,a,B,b,C,g,save):
    """改变视觉编码方式，以附件记录和只读模型输出生成八张图。"""
    from scipy.cluster.hierarchy import dendrogram
    from matplotlib.lines import Line2D
    def title(ax,s): ax.set_title(s,loc='left',pad=12,fontsize=12)
    def style(ax): ax.set_axisbelow(True);ax.grid(axis='x',color='#E5E1E8');ax.tick_params(length=3)
    def foot(fig,s): fig.text(.02,.012,s,fontsize=11,color=INK,ha='left')

    # 0. Evidence lanes keep the conditional p transfer outside the fitted N/D/Q equation.
    fig,ax=plt.subplots(figsize=(7.8,3.5));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,4.1))
    for x,num,label,col in [(.1,'01','数据依据',BLUE),(3.55,'02','关系识别',TEAL),(7,'03','结果解释',RED)]:
        ax.text(x,3.82,num,color=col,fontsize=23,va='center');ax.text(x+.65,3.82,label,fontsize=13,va='center',color=col)
    def box(x,y,w,h,head,body,col):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.06',fc=PALE,ec='#E5E1E8',lw=.8))
        ax.plot([x+.12,x+.12],[y+.15,y+h-.15],color=col,lw=3,solid_capstyle='round')
        ax.text(x+.27,y+h-.23,head,fontsize=11.5,color=col,va='center')
        ax.text(x+.27,y+.22,body,fontsize=10,va='center')
    def arrow(x,y,u,v,dash=False):
        ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=11,color=GRAY,lw=1.1,linestyle='--' if dash else '-'))
    for y,h,t,col in [(2.4,'B1 · 规模轨迹','1,176 个检查点',BLUE),(1.25,'B6 · 质量实验','360 条半合成记录',TEAL),(.1,'问题一 · 配比输出','线性系数 / 核岭损失差',RED)]:box(.1,y,2.8,.86,h,t,col)
    box(3.55,1.85,2.9,1.4,'规模—质量标度律',r'$E+AN^{-\alpha}+BD^{-\beta}$'+'\n'+r'$+\ C(1-Q)^\gamma$',BLUE)
    box(3.55,.1,2.9,1.05,'配比影响分析','领域相似性 / 收益校准',RED)
    box(7,2.4,2.85,.86,'跨来源检验','真实 / 半合成 / 估算',BLUE)
    box(7,1.25,2.85,.86,'规模与质量取舍','弹性 / 等损失条件',TEAL)
    box(7,.1,2.85,.86,'配比情景修正','在损失尺度可迁移时',RED)
    arrow(2.95,2.8,3.5,2.8);arrow(2.95,1.68,3.5,2.05);arrow(2.95,.53,3.5,.53)
    arrow(6.5,2.8,6.95,2.8);arrow(6.5,2.1,6.95,1.68);arrow(6.5,.53,6.95,.53,True)
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);save(fig,'图0_问题二分析框架')

    # 1. All B1 losses in a scale map, alongside per-model residual distributions.
    ns=np.sort(py.N_params_B.unique());ds=np.sort(py.D_tokens_B.unique())
    z=py.pivot(index='N_params_B',columns='D_tokens_B',values='val_loss').loc[ns,ds].to_numpy()
    fig=plt.figure(figsize=(7.8,3.7));gs=fig.add_gridspec(1,2,width_ratios=[1.15,1],wspace=.3)
    ax=fig.add_subplot(gs[0]);ar=fig.add_subplot(gs[1])
    edges=np.r_[ds[0]/np.sqrt(ds[1]/ds[0]),np.sqrt(ds[:-1]*ds[1:]),ds[-1]*np.sqrt(ds[-1]/ds[-2])]
    im=ax.pcolormesh(edges,np.arange(9)-.5,z,cmap=CMAP,shading='flat')
    ax.set(xscale='log',yticks=np.arange(8),yticklabels=[f'{n:.2f}' for n in ns],xlabel='训练数据量 D / B tokens',ylabel='参数量 N / B',ylim=(-.5,7.5))
    ax.set_xticks([.2,1,10,100]);ax.set_xticklabels(['0.2','1','10','100']);title(ax,'(a) 附件记录的损失地图')
    cb=fig.colorbar(im,cax=ax.inset_axes([0,1.16,1,.045]),orientation='horizontal');cb.ax.tick_params(labelsize=11,length=2)
    ax.set_title('(a) 附件记录的损失地图',loc='left',pad=52,fontsize=12)
    residuals=[]
    for n in ns:
        sub=py[py.N_params_B.eq(n)];residuals.append((classic(n,sub.D_tokens_B)-sub.val_loss).to_numpy()*1e4)
    v=ar.violinplot(residuals,positions=np.arange(8),vert=False,widths=.8,showextrema=False)
    for body in v['bodies']:body.set(facecolor=BLUE,edgecolor=BLUE,alpha=.24,linewidth=.7)
    for j,err in enumerate(residuals):
        ar.scatter(err,np.full(len(err),j),s=3,color=BLUE,alpha=.2,edgecolors='none')
        q25,med,q75=np.quantile(err,[.25,.5,.75]);ar.plot([q25,q75],[j,j],color=BLUE,lw=3);ar.scatter(med,j,s=18,color=RED,zorder=4)
    ar.axvline(0,color=GRAY,ls='--',lw=.8);ar.set(yticks=np.arange(8),yticklabels=[f'{n:.2f}' for n in ns],ylim=(-.5,7.5),xlabel='预测残差 / 10⁻⁴')
    ar.set_title('(b) 各参数规模的拟合残差',loc='left',pad=52,fontsize=12);style(ar)
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.75)
    foot(fig,'B1：8 个模型 × 147 个检查点；左图颜色表示验证损失。\n右图：记录点、四分位区间及金色中位点；轮廓表示平滑密度。')
    save(fig,'图1_经典标度律拟合')

    # 2. Signed error distributions complement the summary-metric table.
    sets=[('B4  跨族收敛点','真实',read('scaling_baseline.csv'),1.,BLUE),('B5  文献标度数据','真实',read('published_scaling_data.csv'),1.,BLUE),
        ('B7  新增质量点','半合成',b7n,b7n.Q_score,TEAL),('B2  Cerebras','半合成',read('cerebras_training_log.csv'),1.,TEAL),
        ('B8  校准段','半合成',b8c,1-b8c.Q_score,TEAL),('B10 大规模损失','估算',read('supplementary_large_baseline.csv'),1.,RED)]
    fig,ax=plt.subplots(figsize=(6.2,3.8));rng=np.random.default_rng(20260925)
    saved=pd.read_csv(HERE/'结果/广义标度律分层验证.csv');labels=[]
    for j,(name,kind,df,q,col) in enumerate(sets):
        y=df.val_loss.to_numpy();pred=model(df.N_params_B,df.D_tokens_B,q);err=100*(pred-y)/y;mape=np.mean(abs(err));r=np.corrcoef(y,pred)[0,1]
        old=saved[saved.dataset.str.startswith(name.split()[0])].iloc[0]
        assert np.isclose(mape,old['MAPE_%'],atol=1e-10) and np.isclose(r,old.r,atol=1e-10)
        labels.append(f'{name}\n{kind} · n={len(df):,}');ax.axhspan(j-.45,j+.45,color=PALE,zorder=0)
        v=ax.violinplot([err],positions=[j],vert=False,widths=.62,showextrema=False);v['bodies'][0].set(facecolor=col,edgecolor=col,alpha=.2)
        ax.scatter(err,j+rng.uniform(-.1,.1,len(err)),s=5,alpha=.26,color=col,lw=0)
        q25,med,q75=np.quantile(err,[.25,.5,.75]);ax.plot([q25,q75],[j,j],lw=3,color=col);ax.scatter(med,j,s=20,fc='white',ec=col,zorder=4)
    ax.axvline(0,color=INK,lw=.8,ls='--');ax.set(yticks=np.arange(6),yticklabels=labels,ylim=(5.6,-.65),xlabel='有符号相对误差 / %');style(ax)
    ax.set_xscale('symlog',linthresh=10);ax.set_xticks([-50,-10,0,10,100,500]);ax.set_xticklabels(['−50','−10','0','10','100','500'],fontsize=11)
    fig.subplots_adjust(left=.26,right=.98,bottom=.25,top=.96)
    foot(fig,'相对误差 = 100 × (预测值 − 给定值) / 给定值；横轴在 ±10% 内线性。\nB8 使用 1−Q；空心点：中位数；粗线：四分位区间；轮廓：平滑密度。')
    save(fig,'图2_广义标度律验证')

    # 3. N-Q contour map and asymptotic loss increments; raw design sites are explicit.
    fig=plt.figure(figsize=(7.8,3.7));gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=.42)
    ax=fig.add_subplot(gs[0]);ar=fig.add_subplot(gs[1]);q=np.linspace(.1,1,240);n=np.geomspace(.07,12,240);Q,N=np.meshgrid(q,n);Z=model(N,300,Q)
    cs=ax.contourf(Q,N,Z,levels=np.linspace(2.08,3.15,20),cmap=CMAP,extend='both')
    cont=ax.contour(Q,N,Z,levels=[2.2,2.4,2.6,2.8,3.0],colors='white',linewidths=.8);ax.clabel(cont,fmt='%.1f',fontsize=9,inline_spacing=5,manual=[(.7,6),(.55,1.6),(.45,.45),(.3,.14),(.3,.075)])
    design=b6[b6.D_tokens_B.eq(300)];ax.scatter(design.Q_score,design.N_params_B,s=9,facecolors='none',edgecolors=INK,linewidths=.35,alpha=.5)
    ax.set(yscale='log',xlabel='数据质量 Q',ylabel='参数量 N / B',xlim=(.1,1),ylim=(.07,12));title(ax,'(a) 规模—质量响应面')
    cb=fig.colorbar(cs,ax=ax,pad=.03,fraction=.06,ticks=[2.2,2.6,3.0]);cb.set_label('预测损失',fontsize=10)
    for j,(q0,col) in enumerate(zip([1.,.6,.3],[TEAL,RED,BLUE])):
        floor=E+C*(1-q0)**g;ar.barh(j,floor-E,left=E,height=.4,color=col,alpha=.85);ar.scatter(floor,j,s=50,color=col,zorder=3);ar.text(floor+.012,j,f'{floor:.3f}',va='center',color=col,fontsize=12)
    ar.axvline(E,color=GRAY,ls='--',lw=1);ar.set(yticks=[0,1,2],yticklabels=['Q = 1.0','Q = 0.6','Q = 0.3'],ylim=(2.7,-.7),xlim=(1.62,2.015),xlabel='模型渐近损失')
    ar.text(E,2.57,f'E = {E:.3f}',ha='center',fontsize=10,color=GRAY);title(ar,'(b) 质量缺口对下限的抬升');style(ar)
    fig.subplots_adjust(left=.085,right=.98,bottom=.23,top=.85);foot(fig,'D=300 B tokens；空心点：B6 实验位置；底色与等值线：模型预测。');save(fig,'图3_质量边际影响')

    # 4. Marginal response locations and a dumbbell comparison of saved elasticities.
    fig,axs=plt.subplots(1,3,figsize=(7.8,3.5),gridspec_kw={'width_ratios':[1,1,1.35]})
    nvals=np.array([.07,.3,1,3,12]);qvals=np.array([.1,.3,.6,.9,.99])
    for ax,effect,col,labels,xlabel,heading in [
        (axs[0],a*A*nvals**(-a-1),BLUE,[f'{n:g}' for n in nvals],r'$|\partial L/\partial N|$ / B$^{-1}$','(a) 参数边际效应'),
        (axs[1],C*g*(1-qvals)**(g-1),RED,[f'{q:g}' for q in qvals],r'$|\partial L/\partial Q|$','(b) 质量边际效应')]:
        ypos=np.arange(5)
        ax.scatter(effect,ypos,color=col,s=42,zorder=3)
        ax.set(yticks=ypos,yticklabels=labels,ylim=(4.5,-.6),xlabel=xlabel);title(ax,heading);style(ax)
    axs[0].set_xscale('log');axs[0].set_ylabel('参考参数量 N / B')
    axs[1].set(xlim=(.34,.391),ylabel='参考质量 Q',xticks=[.35,.37,.39])
    el=pd.read_csv(HERE/'结果/弹性与替代关系.csv');v=-el[['弹性e_N','弹性e_D','弹性e_Q']].to_numpy();ar=axs[2]
    for j in range(3):
        ar.plot(v[:,j],[j,j],color='#E5E1E8',lw=5,zorder=1)
        for i,col in enumerate([BLUE,TEAL]):
            ar.scatter(v[i,j],j,s=48,color=col,zorder=3,marker=['o','D'][i],label=['N=1, D=100','N=12, D=300'][i] if j==0 else None)
            ar.annotate(f'{v[i,j]:.3f}',(v[i,j],j),xytext=(0,10 if i==0 else -16),textcoords='offset points',ha='center',fontsize=9,color=col)
    ar.set(yticks=[0,1,2],yticklabels=['参数 N','数据 D','质量 Q'],ylim=(2.55,-.75),xlim=(.01,.112),xlabel='弹性绝对值')
    title(ar,'(c) 两个参考点的敏感度');style(ar);ar.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=1,fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,bottom=.34,top=.87,wspace=.62)
    foot(fig,'左、中图：边际导数，各用自身量纲；右图：无量纲弹性。\n仅右图两个参考点均取 Q=0.6；N、D 以 B 为单位。');save(fig,'图4_边际效用对比')

    # 5. Three original loss targets, separate panels, explicit scale extrapolation.
    fig,axs=plt.subplots(1,3,figsize=(7.8,3.25),sharex=True);ng=np.geomspace(.07,1e4,1200);base=E+B*300**(-b)
    for ax,target,col in zip(axs,[2.30,2.10,1.95],[BLUE,TEAL,RED]):
        remaining=target-base-A*ng**(-a);qreq=np.full_like(ng,np.nan);valid=remaining>=0
        qreq[valid]=1-(remaining[valid]/C)**(1/g);qreq[(qreq<.1)|(qreq>1)]=np.nan
        ax.axvspan(.07,12,color=TEAL,alpha=.065);ax.axvspan(12,1e4,color=RED,alpha=.075);ax.axvline(12,color=GRAY,ls=':',lw=1)
        ax.plot(ng,np.where(ng<=12,qreq,np.nan),color=col,lw=2.5);ax.plot(ng,np.where(ng>12,qreq,np.nan),color=col,lw=2,ls='--')
        ax.set(xscale='log',xlim=(.07,1e4),ylim=(.08,1.035),xlabel='参数量 N / B',xticks=[.1,10,1e3],ylabel='所需质量 Q');title(ax,f'目标损失 L = {target:.2f}');ax.grid(axis='y')
        min_n=(A/(target-base))**(1/a)
        if .07<min_n<1e4:
            ax.scatter(min_n,1,s=32,color=col,zorder=3)
            pos=(.74,.83) if target==2.30 else ((.27,.48) if target==2.10 else (.43,.73))
            ax.annotate(f'Q=1\nN={min_n:.2f} B' if min_n<10 else f'Q=1\nN={min_n:.1f} B',(min_n,1),xytext=pos,textcoords='axes fraction',fontsize=11,color=col,ha='center',va='top',bbox=dict(facecolor=PALE,edgecolor='none',pad=1.5),arrowprops=dict(arrowstyle='-',lw=.7,color=col))
        if target==1.95:
            ax.set_ylim(.95,1.008);ax.set_yticks([.96,.98,1]);ax.text(.06,.1,'高质量区间放大',transform=ax.transAxes,fontsize=10,color=col)
    fig.subplots_adjust(left=.08,right=.985,bottom=.28,top=.84,wspace=.42)
    foot(fig,'D=300 B tokens；实线：N≤12 B；虚线：规模外推。\n右图纵轴单独放大；阴影区分拟合规模与外推规模。');save(fig,'图5_质量参数替代')

    # 6. All 17 domains via A-derived question-one coefficients, not raw A refitting.
    names={'arxiv':'学术预印本','freelaw':'法律判例','nih_exporter':'科研项目','pubmed_central':'医学全文',
        'wikipedia_en':'百科','dm_mathematics':'数学','github':'代码','philpapers':'哲学论文','stackexchange':'专业问答',
        'enron_emails':'邮件','gutenberg_pg_19':'图书','pile_cc':'网页','ubuntu_irc':'技术聊天','europarl':'议会记录',
        'hackernews':'科技社区','pubmed_abstracts':'医学摘要','uspto_backgrounds':'专利背景'}
    coef=pd.read_csv(ROOT/'求解/问题一/结果/混合系数矩阵.csv').set_index('training_domain')
    saved_sim=pd.read_csv(HERE/'结果/领域替代互补矩阵.csv',index_col=0);coef=coef.loc[saved_sim.index]
    center=coef.to_numpy()-coef.to_numpy().mean(axis=0);unit=center/np.linalg.norm(center,axis=1,keepdims=True);sim=unit@unit.T
    assert np.allclose(sim,saved_sim.to_numpy(),atol=1e-12)
    dist=np.sqrt(np.maximum(0,2*(1-sim)));np.fill_diagonal(dist,0);tree=linkage(squareform(dist,checks=False),method='average',optimal_ordering=True)
    pairs=sorted([(sim[i,j],i,j) for i in range(17) for j in range(i)],key=lambda t:t[0]);selected=list(reversed(pairs[-4:]))+pairs[:4]
    fig=plt.figure(figsize=(7.8,5.15));gs=fig.add_gridspec(1,2,width_ratios=[1,1.48],wspace=.88)
    ax=fig.add_subplot(gs[0]);ar=fig.add_subplot(gs[1])
    dendrogram(tree,orientation='left',labels=[names[x] for x in coef.index],ax=ax,color_threshold=0,above_threshold_color=BLUE,leaf_font_size=10)
    ax.set_xlabel('平均连接距离');title(ax,'(a) 17 域效应模式聚类');ax.grid(axis='x');ax.spines[['top','right','left']].set_visible(False)
    labels=[]
    for j,(value,i,k) in enumerate(selected):
        col=TEAL if value>=0 else RED;labels.append(names[coef.index[i]]+' / '+names[coef.index[k]])
        ar.barh(j,value,height=.55,color=col,alpha=.85);ar.text(value*.5,j,f'{value:+.3f}',ha='center',va='center',fontsize=10,color='white')
    ar.axvline(0,color=GRAY,lw=.8);ar.axhline(3.5,color='#E5E1E8',lw=1.5)
    ar.set(yticks=np.arange(8),yticklabels=labels,ylim=(7.7,-.8),xlim=(-1.05,1.05),xlabel='中心化系数的余弦相似度');title(ar,'(b) 最相似与最相反的领域对');style(ar)
    fig.subplots_adjust(left=.04,right=.98,top=.87,bottom=.17)
    foot(fig,'聚类距离 √(2−2s)；绿色：效应趋同；金色：效应相反。\n右图列示 s 最大与最小的各 4 对；相似性不代表互补增益。');save(fig,'图6_领域替代互补')

    # 7. Group correlation distributions rather than point clouds mixing all groups.
    fig,ax=plt.subplots(figsize=(7.8,3.1));rng=np.random.default_rng(19)
    datasets=[('B6',b6,BLUE),('B7',b7,TEAL),('B8 校准段',b8c,RED),('B8 外推段',b8[b8.data_type.eq('extrapolated')],RED)];labels=[]
    for j,(name,df,col) in enumerate(datasets):
        cors=np.array([group.Q_score.corr(group.val_loss) for _,group in df.groupby(['N_params_B','D_tokens_B']) if group.Q_score.nunique()>=3]);assert len(cors)>0
        labels.append(f'{name}\n{len(df):,} 条记录 / {len(cors)} 组');ax.axhspan(j-.44,j+.44,color=PALE)
        ax.scatter(cors,j+rng.uniform(-.19,.19,len(cors)),s=22,color=col,alpha=.43,lw=0);ax.scatter(cors.mean(),j,s=75,marker='D',color=col,ec='white',lw=.9,zorder=4)
        ax.text(1.13,j,f'{cors.mean():+.3f}',va='center',ha='left',color=col,fontsize=12)
    ax.axvline(0,color=GRAY,ls='--',lw=.9);ax.set(xlim=(-1.08,1.42),ylim=(3.6,-.65),yticks=np.arange(4),yticklabels=labels,xticks=[-1,-.5,0,.5,1],xlabel='固定 (N,D) 组内的 Pearson 相关系数')
    ax.text(1.13,-.6,'组间均值',fontsize=10,ha='left');title(ax,'质量字段与损失：组内方向及组间差异');style(ax)
    fig.subplots_adjust(left=.21,right=.98,top=.84,bottom=.32)
    foot(fig,'圆点：一个 (N,D) 组；菱形：组间均值。\n均为半合成数据；B7 含 B6，不构成独立证据。');save(fig,'图7_质量方向对照')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir');args=parser.parse_args()
    render_all(args.data_dir)
