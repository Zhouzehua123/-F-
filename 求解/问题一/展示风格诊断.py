# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6；机构：OpenAI；版本发布日期：2026-04-21。
# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-5.6）辅助。
"""问题一预测、系数、质量口径及尺度诊断的展示图；不重新求解。"""
from 展示风格公共 import panel, clean, save, PURPLE, TEAL, GOLD, BLUE, INK, GRAY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from 绘图数据 import DOMAIN_NAMES

SEQ = LinearSegmentedColormap.from_list('q1_density', ['#EEF4F5','#8DC1BE',TEAL,'#24426C'])
DIVERGING = LinearSegmentedColormap.from_list('q1_signed', [TEAL,'#ACD2CD','#F9FAFC','#C2B5D1',PURPLE])


def labels(domains):
    return [DOMAIN_NAMES[d] for d in domains]


def annotation_color(background):
    """Select the higher-contrast text colour for the actual rendered cell."""
    def luminance(rgb):
        c=np.array(rgb[:3])
        linear=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
        return float(linear @ np.array([.2126,.7152,.0722]))
    bg=luminance(background)
    dark=luminance(to_rgb(INK))
    white_contrast=1.05/(bg+.05)
    dark_contrast=(max(bg,dark)+.05)/(min(bg,dark)+.05)
    return 'white' if white_contrast>dark_contrast else INK


def plot_prediction(data, out):
    x, y = data.observed.ravel(), data.prediction.ravel()
    lo, hi = min(x.min(),y.min()), max(x.max(),y.max())
    lo, hi = np.floor(lo*2)/2-.1, np.ceil(hi*2)/2+.1
    fig = plt.figure(figsize=(4.3,3.7))
    # Equal axis lengths and identical numerical ranges preserve the 45-degree reference.
    square_height = .57 * 4.3 / 3.7
    ax = fig.add_axes([.15,.17,.57,square_height])
    top = fig.add_axes([.15,.855,.57,.085], sharex=ax)
    right = fig.add_axes([.735,.17,.075,square_height], sharey=ax)
    cax = fig.add_axes([.865,.29,.025,.39])
    h = ax.hexbin(x,y,gridsize=43,mincnt=1,cmap=SEQ,linewidths=0,
                  extent=(lo,hi,lo,hi),rasterized=False)
    ax.plot([lo,hi],[lo,hi],color=GOLD,lw=1.15,ls=(0,(4,3)),label='预测值 = 实测值')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='实测验证损失',ylabel='模型预测损失',aspect='equal')
    ax.legend(loc='lower right',fontsize=8)
    ax.text(.035,.963,f'N = {len(x):,}',transform=ax.transAxes,ha='left',va='top',
            fontsize=10,color=BLUE,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':3})
    bins=np.linspace(lo,hi,37)
    top.hist(x,bins=bins,color=TEAL,alpha=.65,edgecolor='white',linewidth=.2)
    right.hist(y,bins=bins,orientation='horizontal',color=PURPLE,alpha=.65,edgecolor='white',linewidth=.2)
    top.axis('off');right.axis('off')
    cb=fig.colorbar(h,cax=cax)
    cb.ax.set_title('记录数',fontsize=9,pad=8)
    cb.locator=MaxNLocator(5,integer=True);cb.update_ticks()
    cb.outline.set_visible(False)
    save(fig,'图3_预测vs实测',out)


def plot_coefficients(data,out):
    matrix=data.centered.to_numpy()
    vmax=float(np.abs(matrix).max())
    fig=plt.figure(figsize=(5.3,4.3))
    ax=fig.add_axes([.18,.29,.79,.57])
    # Only colour mapping changes: all 221 original coefficients are retained.
    norm=SymLogNorm(linthresh=1,linscale=.5,vmin=-vmax,vmax=vmax,base=10)
    mesh=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=DIVERGING,norm=norm)
    wrapped={'学术预印本':'学术\n预印本','法律判例':'法律\n判例','医学全文':'医学\n全文',
             '专业问答':'专业\n问答','技术聊天':'技术\n聊天','科技社区':'科技\n社区',
             '医学摘要':'医学\n摘要','专利背景':'专利\n背景'}
    ax.set(yticks=np.arange(17),yticklabels=labels(data.centered.index),
           xticks=np.arange(13),xticklabels=[wrapped.get(t,t) for t in labels(data.centered.columns)])
    ax.tick_params(length=0,pad=5)
    ax.set_xticks(np.arange(-.5,13,1),minor=True)
    ax.set_yticks(np.arange(-.5,17,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.5)
    ax.tick_params(which='minor',length=0)
    for j in range(13):
        i=int(np.argmax(np.abs(matrix[:,j])))
        v=matrix[i,j]
        ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=9,
                 color=annotation_color(DIVERGING(norm(v))))
    panel(ax,'A','17 个训练域 × 13 个验证域',PURPLE)
    ax.set_xlabel('验证领域',labelpad=9)
    cax=fig.add_axes([.29,.12,.60,.025])
    cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[-15,-5,-1,0,1,5,15])
    cb.ax.set_xticklabels(['−15','−5','−1','0','1','5','15'])
    cb.set_label('中心化系数（对称对数色标）',labelpad=4)
    cb.outline.set_visible(False)
    save(fig,'图4_混合系数热力图',out)


def plot_quality_comparison(data,out):
    p=data.proxy
    fig=plt.figure(figsize=(5.3,3.65))
    ax=fig.add_axes([.16,.17,.315,.66])
    ar=fig.add_axes([.56,.17,.40,.66])
    y=np.arange(len(p))
    vals=p['quality_score_Q(loss代理)'].to_numpy()
    ax.barh(y,vals,height=.57,color=BLUE,alpha=.85)
    for i,v in enumerate(vals):
        ax.text(v+.025,i,f'{v:.2f}',ha='left',va='center',fontsize=9,color=BLUE)
    ax.set(yticks=y,yticklabels=labels(p.domain),ylim=(12.6,-.6),xlim=(0,1.2),
           xticks=[0,.5,1],xlabel='损失代理评分')
    clean(ax,bands=True,n=13)
    panel(ax,'A','13 域代理评分',BLUE)
    link=data.link.set_index('mixture_domain')
    rank_a=link.Q_A1A3.rank(ascending=False).astype(int)
    rank_b=link['Q_loss代理'].rank(ascending=False).astype(int)
    # Lines encode identity across two methods, not a time trend or score change.
    colors=[BLUE,TEAL,GOLD,PURPLE,'#708398','#9D7465']
    for color,d in zip(colors,rank_a.sort_values().index):
        a,b=rank_a[d],rank_b[d]
        ar.plot([0,1],[a,b],color=color,lw=1.8,alpha=.85,zorder=2)
        ar.scatter([0,1],[a,b],s=110,c=color,edgecolors='white',linewidth=1,zorder=3)
        for xpos,r in [(0,a),(1,b)]:
            ar.text(xpos,r,str(r),color='white',fontsize=9,ha='center',va='center',zorder=4)
        ar.text(-.11,a,DOMAIN_NAMES[d],ha='right',va='center',fontsize=9,color=color)
        ar.text(1.11,b,DOMAIN_NAMES[d],ha='left',va='center',fontsize=9,color=color)
    ar.set(xlim=(-.82,1.82),ylim=(6.65,.2))
    for xpos,name in [(0,'指标质量'),(1,'损失代理')]:
        ar.text(xpos,.48,name,ha='center',va='center',fontsize=9,color=INK)
    ar.text(.5,6.5,rf'Spearman $\rho={data.rho:.2f}$',ha='center',va='center',fontsize=10,color=GRAY)
    ar.axis('off')
    panel(ar,'B','6 个公共领域的排名',TEAL)
    save(fig,'图5_领域质量评分',out)


def plot_extrapolation(data,out):
    domains=list(data.coef.columns)
    order=np.argsort(data.ratios['10b'].mean(axis=0))
    names=[domains[i] for i in order]
    cross=data.cross.loc[names]
    fig=plt.figure(figsize=(6.5,4.8))
    ax=fig.add_axes([.15,.15,.40,.64])
    ar=fig.add_axes([.64,.15,.325,.64],sharey=ax)
    y=np.arange(13)
    for size,offset,color in [('10b',-.17,TEAL),('70b',.17,PURPLE)]:
        samples=[data.ratios[size][:,i] for i in order]
        ax.boxplot(samples,positions=y+offset,vert=False,widths=.24,patch_artist=True,
                    whis=(0,100),showfliers=False,manage_ticks=False,
                    boxprops={'facecolor':color,'edgecolor':color,'alpha':.42,'linewidth':.7},
                    medianprops={'color':color,'linewidth':1.3},
                    whiskerprops={'color':color,'linewidth':.75},
                    capprops={'color':color,'linewidth':.75})
    ax.set(yticks=y,yticklabels=labels(names),ylim=(12.6,-.6),xlabel='外推损失 / 1M 训练损失')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ar.errorbar(cross.loss_ratio_60m_vs_1m,y,xerr=cross.loss_ratio_std,fmt='o',
                 color=GOLD,ecolor=GOLD,ms=4,capsize=2.5,elinewidth=1.1)
    ar.axvline(1,color=GRAY,lw=.8,ls=(0,(3,3)))
    ar.tick_params(labelleft=False)
    ar.set(xlim=(.42,1.035),xticks=[.5,.7,.9,1],xlabel='60M / 1M 检验损失')
    for a in [ax,ar]:
        clean(a,bands=True,n=13)
    panel(ax,'A','10B / 70B 外推分布',TEAL)
    panel(ar,'B','同配方跨尺度比值',GOLD)
    fig.legend(handles=[Patch(facecolor=TEAL,alpha=.55,label='10B'),
                        Patch(facecolor=PURPLE,alpha=.55,label='70B')],
                loc='upper left',bbox_to_anchor=(.15,.962),ncol=2,handlelength=1.3,columnspacing=1.8)
    save(fig,'图7_外推与跨尺度',out)
