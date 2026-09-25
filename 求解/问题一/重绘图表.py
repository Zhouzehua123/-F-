"""问题一九幅图重绘：仅读取附件和既有结果，不执行求解程序。

运行：python 重绘图表.py --data-dir E:/华为杯f/real_attachments
输出：图片目录内同名 PNG、SVG，以及绘图数据核验.json。
开发辅助工具：OpenAI Codex。图中数值均来自附件、保存结果或既有参数的直接预测。
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import PercentFormatter, MaxNLocator

from 绘图数据 import HERE, DOMAIN_NAMES, load_plot_data, verify_inputs_unchanged

PURPLE, TEAL, GOLD = '#665080', '#367E73', '#CB8B38'
INK, GRAY, GRID, PALE = '#302F3D', '#77727D', '#E5E1E8', '#FAF8F4'
SEQUENTIAL = LinearSegmentedColormap.from_list('q1_quality', ['#FAF3D8','#D8B979','#AD8792','#6D527D','#382F58'])
NAMES = ['图0_A1-A3全量质量评分', '图1_训练配比结构', '图2_线性模型R2',
         '图3_预测vs实测', '图4_混合系数热力图', '图5_领域质量评分',
         '图6_冲突与赋权诊断', '图7_外推与跨尺度', '图8_问题一求解流程']


def configure():
    for name in ['simsun.ttc', 'times.ttf', 'timesi.ttf', 'timesbd.ttf']:
        path = Path('C:/Windows/Fonts')/name
        if path.exists():
            fm.fontManager.addfont(str(path))
    for family in ['SimSun', 'Times New Roman']:
        fm.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'], 'font.size': 12,
        'font.weight': 'normal', 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 11,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
        'axes.edgecolor': '#9AA8B3', 'axes.linewidth': .7,
        'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.linewidth': .6, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'path', 'svg.hashsalt': 'q1-20260925', 'savefig.dpi': 400,
    })


def style(ax, axis='x'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis)
    ax.tick_params(length=3)


def title(ax, text):
    ax.set_title(text, loc='left', pad=12)


def note(fig, text, y=.022):
    fig.text(.025, y, text, ha='left', va='bottom', fontsize=11, color=GRAY)


def labels(domains):
    return [DOMAIN_NAMES[x] for x in domains]


def save(fig, stem, out):
    fig.canvas.draw()
    fig.savefig(out/f'{stem}.png', dpi=400, bbox_inches='tight', pad_inches=.08)
    fig.savefig(out/f'{stem}.svg', bbox_inches='tight', pad_inches=.08,
                metadata={'Date': None, 'Creator': 'Q1 result visualization'})
    svg = out/f'{stem}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8')
    plt.close(fig)


def plot_quality(data, out):
    q = data.quality
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    y = np.arange(len(q))
    ax.errorbar(q.quality_mean, y, xerr=q.quality_std, fmt='none', color=PURPLE,
                alpha=.55, capsize=4, linewidth=2)
    ax.scatter(q.quality_mean, y, s=72, color=PURPLE, edgecolor='white', linewidth=.7, zorder=3)
    for i, row in q.iterrows():
        ax.text(.872, i, f'{row.quality_mean:.3f}', va='center', ha='right', color=PURPLE)
        ax.text(.99, i, f'{int(row.n):,}', va='center', ha='right', color=GRAY)
    ax.text(.872, -.9, '均值', ha='right', color=GRAY, fontsize=11)
    ax.text(.99, -.9, '样本数', ha='right', color=GRAY, fontsize=11)
    ax.set(yticks=y, yticklabels=labels(q.domain), ylim=(6.65,-1.35), xlim=(.3,1.01),
           xticks=[.3,.4,.5,.6,.7,.8], xlabel='质量评分 Q')
    style(ax)
    fig.subplots_adjust(left=.17, right=.98, bottom=.23, top=.95)
    note(fig, 'A1—A3 全量 272,505 条记录；圆点为均值，误差线为样本标准差。')
    save(fig, NAMES[0], out)


def plot_mixture(data, out):
    order = data.reference.sort_values(ascending=False).index
    fig = plt.figure(figsize=(7.8, 5.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4,1], left=.16, right=.97,
                          top=.83, bottom=.14, wspace=.22)
    ax, ar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    matrix = data.train_x.loc[:,order].iloc[:8].to_numpy().T
    mesh = ax.imshow(matrix, aspect='auto', cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation='nearest')
    ax.set(yticks=range(17), yticklabels=labels(order), xticks=range(8),
           xticklabels=[str(i) for i in data.train_x.index[:8]], xlabel='训练配方 index')
    ax.set_xticks(np.arange(-.5,8,1), minor=True)
    ax.set_yticks(np.arange(-.5,17,1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.5)
    ax.tick_params(which='minor', length=0)
    cax = ax.inset_axes([0,1.07,1,.035])
    fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[0,.25,.5,.75,1],
                 format=PercentFormatter(1))
    ax.set_title('(a) 前 8 组训练配方', loc='left', pad=57)
    y = np.arange(17)
    values = data.reference.loc[order].to_numpy()
    ar.hlines(y, 0, values, color=TEAL, alpha=.4, linewidth=1.5)
    ar.scatter(values, y, color=TEAL, s=45, zorder=3)
    for v, pos in zip(values,y):
        ar.annotate(f'{v:.1%}', (v,pos), xytext=(6,0), textcoords='offset points',
                    va='center', color=TEAL, fontsize=11)
    ar.set(yticks=y, yticklabels=[], ylim=(16.5,-.5), xlim=(0,values.max()+.045), xlabel='平均训练占比')
    ar.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ar.xaxis.set_major_locator(MaxNLocator(4))
    style(ar)
    ar.set_title('(b) 全部 512 组的平均占比', loc='left', pad=57)
    note(fig, '两图使用相同领域顺序；按平均占比降序排列。左图颜色表示配方中的实际占比。')
    save(fig, NAMES[1], out)


def plot_fit(data, out):
    m = data.metrics.sort_values('test1m_R2', ascending=False)
    fig, axes = plt.subplots(1,2,figsize=(7.8,5.3), sharey=True)
    y = np.arange(len(m))
    for ax, c1, c2, lab1, lab2 in [
        (axes[0], 'train_R2','test1m_R2','训练 1M','检验 1M'),
        (axes[1], 'test60m_r','test1B_r','检验 60M','检验 1B')]:
        ax.hlines(y, np.minimum(m[c1],m[c2]), np.maximum(m[c1],m[c2]), color='#C4BCCF', lw=1.8)
        ax.scatter(m[c1], y, s=47, color=PURPLE, marker='o', label=lab1, zorder=3)
        ax.scatter(m[c2], y, s=50, color=GOLD, marker='^', label=lab2, zorder=3)
        ax.set(xlim=(.15,1.02), xticks=[.2,.4,.6,.8,1], ylim=(12.65,-.65))
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.16), ncol=2, columnspacing=.8,
                  handletextpad=.25, fontsize=11)
        style(ax)
    axes[0].set(yticks=y, yticklabels=labels(m.index), xlabel='决定系数 R²')
    axes[1].set_xlabel('Pearson 相关系数 r')
    title(axes[0], '(a) 同尺度拟合与检验')
    title(axes[1], '(b) 跨尺度相关性')
    fig.subplots_adjust(left=.16, right=.98, bottom=.25, top=.9, wspace=.2)
    note(fig, '连线连接同一验证域的两个结果；R² 与 Pearson r 分面展示，按 1M 检验 R² 排序。')
    save(fig, NAMES[2], out)


def plot_prediction(data, out):
    actual, pred = data.observed.ravel(), data.prediction.ravel()
    low, high = min(actual.min(),pred.min()), max(actual.max(),pred.max())
    pad = (high-low)*.04
    lim = (low-pad, high+pad)
    fig = plt.figure(figsize=(6.5,6.16))
    gs = fig.add_gridspec(2,2,width_ratios=[5,1],height_ratios=[1,5],hspace=.07,wspace=.07,
                         left=.12,right=.85,bottom=.19,top=.96)
    ax = fig.add_subplot(gs[1,0]); top = fig.add_subplot(gs[0,0],sharex=ax)
    right = fig.add_subplot(gs[1,1],sharey=ax)
    mesh = ax.hexbin(actual,pred,gridsize=34,mincnt=1,cmap=SEQUENTIAL,linewidths=.15,
                     extent=(lim[0],lim[1],lim[0],lim[1]))
    ax.plot(lim,lim,color=TEAL,lw=1.3,ls='--',label='预测 = 实测')
    ax.set(xlim=lim,ylim=lim,xlabel='实测交叉熵损失',ylabel='模型预测损失')
    ax.legend(loc='upper left')
    bins = np.linspace(*lim,30)
    top.hist(actual,bins=bins,color=PURPLE,alpha=.75,edgecolor='white',linewidth=.3)
    right.hist(pred,bins=bins,orientation='horizontal',color=GOLD,alpha=.8,edgecolor='white',linewidth=.3)
    top.set_axis_off(); right.set_axis_off()
    cax = fig.add_axes([.9,.26,.025,.48])
    cb = fig.colorbar(mesh,cax=cax)
    cb.set_label('每个六边形内的记录数',labelpad=8)
    cb.ax.yaxis.set_major_locator(MaxNLocator(4,integer=True))
    note(fig, '1M 检验：256 组配方 × 13 个验证域，共 3,328 点。\n上方和右侧分别为实测损失与预测损失的频数分布。')
    save(fig, NAMES[3], out)


def plot_coefficients(data, out):
    # No interpolation: each circle represents exactly one saved matrix entry.
    c = data.centered
    a = c.to_numpy().T
    vmax = float(np.max(np.abs(a)))
    x,y = np.meshgrid(np.arange(a.shape[1]),np.arange(a.shape[0]))
    fig, ax = plt.subplots(figsize=(7.8,5.75))
    ax.scatter(x.ravel(),y.ravel(),c=np.where(a.ravel()<0,TEAL,PURPLE),s=210*np.abs(a.ravel())/vmax,
                      linewidth=.25,edgecolor='white',zorder=3)
    ax.set(xticks=range(17),xticklabels=labels(c.index),yticks=range(13),yticklabels=labels(c.columns),
           xlim=(-.7,16.7),ylim=(12.7,-.7),xlabel='训练领域',ylabel='验证领域')
    plt.setp(ax.get_xticklabels(),rotation=55,ha='right',rotation_mode='anchor')
    ax.grid(color=GRID,linewidth=.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.16,right=.98,bottom=.29,top=.83)
    color_handles = [Line2D([],[],marker='o',ls='',color=col,label=label,markersize=7)
                     for col,label in [(TEAL,'负系数'),(PURPLE,'正系数')]]
    sign_legend = ax.legend(handles=color_handles,loc='lower left',bbox_to_anchor=(0,1.035),
                            ncol=2,columnspacing=1,handletextpad=.3)
    ax.add_artist(sign_legend)
    handles=[ax.scatter([],[],s=210*v/vmax,color=GRAY,alpha=.7,label=str(v)) for v in [2,8,15]]
    ax.legend(handles=handles,title='绝对值',loc='lower right',bbox_to_anchor=(1.02,1.035),ncol=3,
              fontsize=10.5,title_fontsize=11,columnspacing=.45,handletextpad=.2)
    note(fig, '221 个气泡对应 17 × 13 个中心化系数；面积与绝对值成正比，零系数不绘制。\n系数符号表示相对于该验证领域系数均值的方向，不表示单独增加某领域的因果效应。')
    save(fig, NAMES[4], out)


def plot_quality_comparison(data, out):
    q = data.proxy
    link = data.link.sort_values('Q_A1A3',ascending=False).copy()
    left_rank = link.Q_A1A3.rank(ascending=False).to_numpy()
    right_rank = link['Q_loss代理'].rank(ascending=False).to_numpy()
    fig = plt.figure(figsize=(7.8,5.25))
    gs = fig.add_gridspec(1,2,width_ratios=[1,1.15],left=.16,right=.98,bottom=.28,top=.9,wspace=.72)
    ax, ar = fig.add_subplot(gs[0]),fig.add_subplot(gs[1])
    y = np.arange(13)
    ax.hlines(y,0,q['quality_score_Q(loss代理)'],color=TEAL,alpha=.3,lw=1.5)
    ax.scatter(q['quality_score_Q(loss代理)'],y,color=TEAL,s=42,zorder=3)
    ax.set(yticks=y,yticklabels=labels(q.domain),ylim=(12.65,-.65),xlim=(-.04,1.04),
           xticks=[0,.5,1],xlabel='损失难度代理得分')
    title(ax,'(a) 13 个验证领域');style(ax)
    y = np.arange(6)
    ar.hlines(y,left_rank,right_rank,color='#C4BCCF',lw=2)
    ar.scatter(left_rank,y,color=PURPLE,s=55,label='A1—A3 质量',zorder=3)
    ar.scatter(right_rank,y,color=GOLD,marker='^',s=55,label='损失难度代理',zorder=3)
    ar.set(yticks=y,yticklabels=labels(link.mixture_domain),xticks=range(1,7),
           xlim=(.6,6.4),ylim=(5.8,-1),xlabel='领域名次（1 为最高）')
    ar.text(.03,.96,f'Spearman ρ = {data.rho:.2f}',transform=ar.transAxes,color=GRAY,va='top')
    fig.legend(*ar.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.62,.12),ncol=2,fontsize=11)
    title(ar,'(b) 6 个映射领域的排序');style(ar)
    note(fig, '左图得分越高表示训练损失越低；右图比较两套口径的领域名次。')
    save(fig, NAMES[5], out)


def plot_conflicts(data, out):
    w = data.weights.sort_values('weight',ascending=False)
    codes = data.weights.set_index('indicator').code.to_dict()
    pairs = data.conflicts.head(12)
    pair_labels = [' × '.join(codes[i] for i in p.split('|')) for p in pairs.indicator_pair]
    fig = plt.figure(figsize=(7.8,6.5))
    gs = fig.add_gridspec(1,2,width_ratios=[1,1.2],left=.16,right=.98,bottom=.20,top=.91,wspace=.95)
    ax, ar = fig.add_subplot(gs[0]),fig.add_subplot(gs[1])
    y = np.arange(12)
    ax.hlines(y,0,pairs.conflict_rate,color=GOLD,alpha=.45,lw=1.6)
    ax.scatter(pairs.conflict_rate,y,color=GOLD,s=52,zorder=3)
    ax.set(yticks=y,yticklabels=pair_labels,ylim=(11.6,-.6),xlim=(0,.485),
           xticks=[0,.2,.4],xlabel='冲突样本占比')
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    for pos,v in zip(y,pairs.conflict_rate):
        ax.annotate(f'{v:.1%}',(v,pos),xytext=(6,0),textcoords='offset points',
                    fontsize=10.5,va='center',color=GOLD)
    title(ax,'(a) 冲突比例最高的 12 对');style(ax)
    y = np.arange(22)
    ar.hlines(y,0,w.weight,color=PURPLE,alpha=.4,lw=1.3)
    ar.scatter(w.weight,y,color=PURPLE,s=38,zorder=3)
    ar.set(yticks=y,yticklabels=[f'{r.code}  {r.display_name}' for r in w.itertuples()],
           ylim=(21.65,-.65),xlim=(0,.185),xticks=[0,.05,.10,.15],xlabel='指标熵权')
    ar.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ar.tick_params(axis='y',labelsize=11.5)
    for pos, v in enumerate(w.weight):
        if pos < 2:
            ar.text(v+.006,pos,f'{v:.1%}',fontsize=11,va='center',color=PURPLE)
    title(ar,'(b) 22 项指标的熵权');style(ar)
    note(fig, 'M01—M22 按正文指标表的原顺序编号，右图按熵权排序。\n冲突阈值为 |zⱼ − zₖ| > 1.5；左图列出冲突比例最高的 12 对指标。')
    save(fig, NAMES[6], out)


def plot_extrapolation(data, out):
    domains = list(data.coef.columns)
    order = np.argsort(data.scale.scale_factor_10B.to_numpy())
    vals = data.ratios['10b'][:,order]
    cross = data.cross.iloc[order]
    fig, axes = plt.subplots(1,2,figsize=(7.8,5.6),sharey=True)
    ax, ar = axes; y = np.arange(13)
    violin = ax.violinplot([vals[:,i] for i in range(13)],positions=y,orientation='horizontal',
                           widths=.68,showextrema=False,bw_method=.5)
    for body in violin['bodies']:
        body.set_facecolor(PURPLE);body.set_edgecolor(PURPLE);body.set_alpha(.18);body.set_linewidth(.6)
    rng = np.random.default_rng(20260925)
    for i in range(13):
        ax.scatter(vals[:,i],i+rng.uniform(-.17,.17,len(vals)),s=5,alpha=.32,color=PURPLE,linewidths=0)
        q1,median,q3 = np.quantile(vals[:,i],[.25,.5,.75])
        ax.plot([q1,q3],[i,i],lw=3,color=PURPLE,solid_capstyle='round')
        ax.scatter(median,i,s=25,color=GOLD,zorder=4)
    ax.axvline(vals.mean(),color=GRAY,ls='--',lw=.9)
    ax.set(yticks=y,yticklabels=labels([domains[i] for i in order]),ylim=(12.6,-.6),
           xlabel='10B 外推损失 / 1M 训练损失')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    title(ax,'(a) 10B 外推记录分布');style(ax)
    ar.errorbar(cross.loss_ratio_60m_vs_1m,y,xerr=cross.loss_ratio_std,fmt='o',
                 color=TEAL,ecolor=TEAL,markersize=5,capsize=3,elinewidth=1.3)
    ar.axvline(1,color=GRAY,ls='--',lw=.9)
    ar.set(xlim=(.42,1.03),xticks=[.5,.7,.9,1],xlabel='同配方检验损失 60M / 1M')
    title(ar,'(b) 跨尺度均值与标准差');style(ar)
    fig.subplots_adjust(left=.16,right=.98,bottom=.21,top=.9,wspace=.25)
    note(fig, '左：轮廓为每域 63 条记录的密度，粗线为四分位区间，金点为中位数，虚线为全域均值。\n右：圆点为 256 组同配方记录的均值，误差线为标准差；两图使用相同领域顺序。')
    save(fig, NAMES[7], out)


def plot_workflow(data, out):
    fig, ax = plt.subplots(figsize=(7.8,5.1))
    ax.set(xlim=(0,12.3),ylim=(0,7.4));ax.axis('off')
    for x,num,text,col in [(0.05,'01','附件与处理',PURPLE),(4.2,'02','评分与回归',TEAL),(8.35,'03','诊断与输出',GOLD)]:
        ax.text(x,7.1,num,color=col,fontsize=23,va='center')
        ax.text(x+.8,7.1,text,color=col,fontsize=13,va='center')
    def box(x,y,w,h,head,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.07',
                                    facecolor=PALE,edgecolor=GRID,lw=.8))
        ax.plot([x+.14,x+.14],[y+.16,y+h-.16],color=color,lw=3,solid_capstyle='round')
        ax.text(x+.3,y+h-.25,head,color=color,fontsize=12,va='top')
        ax.text(x+.3,y+.20,body,color=INK,fontsize=11,va='bottom',linespacing=1.4)
    def arrow(points,dashed=False):
        xs,ys = zip(*points)
        if len(points)>2:
            ax.plot(xs[:-1],ys[:-1],color=GRAY,lw=1,ls='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,
                                     color=GRAY,lw=1,linestyle='--' if dashed else '-'))
    box(.05,4.45,3.3,1.9,'A1—A3 · 质量信号','22 项指标的缺失处理\n列表标量化、方向统一\n归一化并保留领域标签',PURPLE)
    box(.05,1.25,3.3,1.65,'A4—A5 · 配方与损失','按 index 对应\n17 域配比、13 域损失',PURPLE)
    box(4.2,4.5,3.3,1.65,'熵权综合评价','样本与语料评分\n领域质量评分',TEAL)
    box(8.35,4.5,3.65,1.65,'指标冲突诊断','成对标准化差异\nKendall 排序一致性',GOLD)
    box(4.2,2.85,3.3,1.08,'A16 · 领域映射','构造配比加权质量指数',TEAL)
    box(4.2,1.0,3.3,1.45,'岭正则线性混合回归','估计系数，预测验证损失',TEAL)
    box(8.35,2.3,3.65,1.65,'检验与跨尺度分析','1M / 60M / 1B 检验\n10B / 70B 外推表',GOLD)
    box(8.35,.3,3.65,1.6,'有界配比推荐','固定未测量域\n约束其余领域配比上限',GOLD)
    arrow([(3.35,5.32),(4.15,5.32)])
    # Conflict diagnostics use normalized signals, rather than treating Q as their input.
    arrow([(1.7,6.35),(1.7,6.6),(10.18,6.6),(10.18,6.2)])
    arrow([(5.85,4.5),(5.85,3.98)])
    arrow([(3.35,2.05),(4.15,2.05)])
    arrow([(5.85,2.85),(5.85,2.5)],dashed=True)
    ax.text(6.05,2.64,'增量检验',fontsize=10,color=GRAY,ha='left',va='center')
    arrow([(7.5,1.72),(7.95,1.72),(7.95,3.15),(8.3,3.15)])
    arrow([(10.18,2.3),(10.18,1.95)])
    fig.subplots_adjust(left=.015,right=.99,bottom=.12,top=.97)
    note(fig, '实线表示主要处理流程；虚线表示追加混合质量项的增量检验。')
    save(fig, NAMES[8], out)


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=HERE/'图片')
    parser.add_argument('--only', type=int, nargs='+', choices=range(9), help='只重绘指定图文件编号')
    args = parser.parse_args()
    configure()
    data = load_plot_data(args.data_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    functions = [plot_quality,plot_mixture,plot_fit,plot_prediction,plot_coefficients,
                 plot_quality_comparison,plot_conflicts,plot_extrapolation,plot_workflow]
    selected = args.only if args.only is not None else list(range(9))
    for i in selected:
        functions[i](data,args.output_dir)
        print(f'已绘制 {NAMES[i]}',flush=True)
    verify_inputs_unchanged(data)
    manifest = {'source_revision':'8ca02a9','input_checks':data.checks,
                'inputs':{k:{field:value for field,value in v.items() if field!='_path'} for k,v in data.inputs.items()},
                'output_figures':[NAMES[i] for i in selected],
                'indicator_key':data.weights[['code','indicator','display_name']].to_dict(orient='records'),
                'domain_key':DOMAIN_NAMES,'prediction_points':int(data.observed.size),
                'coefficient_entries':int(data.centered.size),'whole_paper_compiled':False}
    (args.output_dir/'绘图数据核验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(data.checks)} 项数值核对通过；{len(data.inputs)} 个输入文件哈希未变。',flush=True)


if __name__ == '__main__':
    main()
