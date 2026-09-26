# 本绘图程序在人工智能工具 Codex 辅助下完成。
# 开发机构：OpenAI；模型：GPT-6 系列（2026-09-03 首次发布）。
# 仅读取附件和已有计算结果，不拟合模型，不写入结果目录。
"""重绘问题四的七幅图。运行时以 --data 指定 real_attachments 目录。

例：python 重绘图表.py --data ../../../../real_attachments
可选 --audit 指定内部审查目录，输出可编辑 SVG、数据摘要和布局检查。
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULT = HERE / '结果'
PURPLE, TEAL, GOLD, INK = '#665080', '#367E73', '#CB8B38', '#302F3D'
TASKS = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
LICENSES = {'apache-2.0', 'mit', 'gpl-3.0', 'cc-by-4.0', 'mpl-2.0',
            'bsd-3-clause', 'llama2', 'llama3', 'llama3.1', 'llama3.2',
            'llama3.3', 'gemma', 'cc-by-sa-4.0', 'openrail'}
NAMES = {
    'bbh_boolean_expressions':'布尔表达式', 'bbh_causal_judgement':'因果判断',
    'bbh_date_understanding':'日期理解', 'bbh_disambiguation_qa':'歧义消解',
    'bbh_formal_fallacies':'形式谬误', 'bbh_geometric_shapes':'几何形状',
    'bbh_hyperbaton':'形容词排序', 'bbh_logical_deduction_five_objects':'逻辑推断·5对象',
    'bbh_logical_deduction_seven_objects':'逻辑推断·7对象',
    'bbh_logical_deduction_three_objects':'逻辑推断·3对象',
    'bbh_movie_recommendation':'电影推荐', 'bbh_navigate':'路径导航',
    'bbh_object_counting':'对象计数', 'bbh_penguins_in_a_table':'表格中的企鹅',
    'bbh_reasoning_about_colored_objects':'彩色对象推理', 'bbh_ruin_names':'名称变换',
    'bbh_salient_translation_error_detection':'翻译错误识别', 'bbh_snarks':'讽刺识别',
    'bbh_sports_understanding':'体育知识理解', 'bbh_temporal_sequences':'时间序列推理',
    'bbh_tracking_shuffled_objects_five_objects':'位置跟踪·5对象',
    'bbh_tracking_shuffled_objects_seven_objects':'位置跟踪·7对象',
    'bbh_tracking_shuffled_objects_three_objects':'位置跟踪·3对象',
    'bbh_web_of_lies':'真假陈述推理',
    'math_algebra_hard':'代数', 'math_counting_and_prob_hard':'计数与概率',
    'math_geometry_hard':'几何', 'math_intermediate_algebra_hard':'中级代数',
    'math_num_theory_hard':'数论', 'math_prealgebra_hard':'初等代数',
    'math_precalculus_hard':'预备微积分',
    'gpqa_diamond':'Diamond', 'gpqa_extended':'Extended', 'gpqa_main':'Main',
    'musr_murder_mysteries':'谋杀谜题', 'musr_object_placements':'物体摆放',
    'musr_team_allocation':'团队分配',
}

def read(name):
    return pd.read_csv(RESULT / (name + '.csv'))

def hashes():
    return {str(p.relative_to(RESULT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in RESULT.rglob('*') if p.is_file()}

def style():
    # 固定原导出样式，避免依赖绘图者个人目录中的可选技能包。
    plt.rcParams.update({'figure.dpi':150, 'figure.figsize':[5,3.5],
        'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.serif':['Noto Serif SC','Times New Roman','Times','DejaVu Serif'],
        'lines.linewidth':1.2, 'lines.markersize':5, 'pdf.fonttype':42,
        'ps.fonttype':42, 'savefig.dpi':300})
    for name in ['simsun.ttc', 'times.ttf']:
        p = Path('C:/Windows/Fonts') / name
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':['Times New Roman', 'SimSun'], 'font.size':10,
        'axes.labelsize':10, 'axes.titlesize':11, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'text.color':INK, 'axes.labelcolor':INK, 'axes.edgecolor':'#88838D',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.unicode_minus':False,
        'axes.linewidth':.6, 'legend.fontsize':9, 'legend.frameon':False,
        'svg.fonttype':'none', 'savefig.facecolor':'white',
        'figure.constrained_layout.use':False, 'figure.autolayout':False})

def grid(ax, axis='y'):
    ax.grid(axis=axis, color='#E8E4E9', lw=.6)
    ax.set_axisbelow(True)

def title(ax, label):
    ax.set_title(label, loc='left', pad=9)

def save(fig, name, audit):
    fig.canvas.draw()
    issues = []
    try:
        from visual_qa import audit_layout
        issues = audit_layout(fig)
    except ImportError:
        pass
    if any(level == 'FAIL' for level, _ in issues):
        raise RuntimeError(issues)
    out = HERE / '图片' / (name + '.png')
    fig.savefig(out, dpi=400)
    if audit:
        fig.savefig(audit / (name + '.svg'))
        (audit / (name + '_layout.json')).write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.close(fig)
    print(name, issues)

def bridge(audit):
    df, fits = read('Loss_Benchmark映射明细'), read('Loss_Benchmark映射').set_index('可比性')
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.05))
    for ax, level, prefix, color, marker in zip(axes, ['高可比','中可比'], ['High','Medium'], [TEAL,PURPLE], ['D','o']):
        d = df[df.Loss_Comparability.str.startswith(prefix)]
        f = fits.loc[level]
        assert len(d) == f['n']
        ax.scatter(d.Val_Loss, d.LB_Average, s=25, color=color, marker=marker, alpha=.75, edgecolors='white', linewidths=.4)
        x = np.linspace(d.Val_Loss.min(), d.Val_Loss.max(), 80)
        ax.plot(x, f['截距'] + f['斜率']*x, color=color, lw=1.7)
        title(ax, f'{"(a)" if level == "高可比" else "(b)"} {level} · n={len(d)}')
        ax.set_xlabel('验证交叉熵损失')
        grid(ax)
        ax.text(.97, .93, f'$R^2$ = {f.R2:.2f}\nMAPE = {f["MAPE_%"]:.1f}%', ha='right', va='top', transform=ax.transAxes, fontsize=9)
    axes[0].set(ylabel='Leaderboard 均分', ylim=(4.7,6.6))
    axes[1].set(ylim=(0,65))
    fig.subplots_adjust(left=.09, right=.98, bottom=.20, top=.85, wspace=.30)
    save(fig, '图1_Loss_Benchmark映射', audit)

def decomposition(audit):
    df = read('规模时间分解')
    fig, axes = plt.subplots(1,2,figsize=(6.2,3.45),sharey=True)
    for ax, (_, r), label in zip(axes, df.iterrows(), ['(a) C3 · 2019—2025','(b) C1 · 2024年6—9月']):
        v = [r['规模贡献分'], r['非规模贡献分'], r['前沿提升分']]
        ax.bar([0,1,2], v, bottom=[0,v[0],0], color=[TEAL,GOLD,PURPLE], width=.62)
        ax.plot([.31,.69],[v[0],v[0]],lw=.7,color=INK)
        ax.plot([1.31,1.69],[v[2],v[2]],lw=.7,color=INK)
        for i, (value, y) in enumerate(zip(v,[v[0],v[2],v[2]])):
            ax.text(i, y+1.2, f'{value:.2f}',ha='center',fontsize=10)
        ax.set(xticks=[0,1,2],xticklabels=['规模项','非规模项','总增量'],ylim=(0,67))
        title(ax,label)
        grid(ax)
        ax.text(.03,.93,f'规模占比 {r["规模占比%"]:.1f}%\n非规模占比 {r["非规模占比%"]:.1f}%', transform=ax.transAxes,va='top',fontsize=9)
    axes[0].set_ylabel('前沿提升 / 分')
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.85,wspace=.20)
    save(fig,'图2_规模时间分解',audit)

def forecast(lb, audit):
    par=read('logistic参数').iloc[0]
    pred=read('前沿预测')
    hist=read('前沿序列').query('Year <= 2023')
    mon=lb.groupby(lb.date.dt.to_period('M'))['avg'].apply(lambda s:s.nlargest(max(1,int(len(s)*.01))).mean())
    times=np.array([p.year+(p.month-.5)/12 for p in mon.index])
    x=np.r_[hist.Year.to_numpy()+.5,times]
    y=np.maximum.accumulate(np.r_[hist.cum_avg.to_numpy(),mon.cummax().to_numpy()])
    assert len(x)==par['n前沿点']
    log=lambda xx:par.K/(1+np.exp(-par.r*(np.asarray(xx)-2019-par['t0(2019起)'])))
    assert np.allclose(log(pred.iloc[:,0]+2019),pred.frontier_pred)
    fig=plt.figure(figsize=(6.2,4.65))
    gs=fig.add_gridspec(2,1,height_ratios=[1.35,1],hspace=.54)
    ax=fig.add_subplot(gs[0]); last=2019+par.t_last
    xx=np.linspace(x.min(),2027.3,250)
    ax.axvspan(last,2027.4,color='#F3EFF6',zorder=0)
    ax.plot(xx,log(xx),color=PURPLE,lw=1.6,label='logistic 曲线')
    slow_x=np.linspace(last,2027.3,100)
    slow_y=log(last+0.5*(slow_x-last))
    assert np.allclose(log(last+0.5*(pred.iloc[:,0]+2019-last)),pred['情景_技术增速减半'])
    ax.plot(slow_x,slow_y,color=GOLD,lw=1.4,ls='--',label='logistic 增长参数减半')
    ax.plot(x,y,ls=':',lw=.8,color='#8E8994')
    ax.scatter(x[:len(hist)],y[:len(hist)],color=GOLD,marker='s',s=25,label='C3 年度锚点',zorder=3)
    ax.scatter(x[len(hist):],y[len(hist):],color=TEAL,marker='o',s=20,label='C1 月度前沿',zorder=3)
    xp=pred.iloc[:,0].to_numpy()+2019
    ax.scatter(xp,pred['情景_技术增速减半'],marker='v',s=23,color=GOLD,zorder=4)
    ax.errorbar(xp,pred.frontier_pred,yerr=[pred.frontier_pred-pred['CI_low_5%'],pred['CI_high_95%']-pred.frontier_pred],fmt='D',color=PURPLE,capsize=4,ms=4,label='90% bootstrap 区间')
    ax.set(xlim=(2019,2027.5),ylim=(0,75),xticks=[2019,2021,2023,2025,2027],ylabel='前沿得分 / 分')
    ax.axvline(last,color='#ACA5B2',ls='--',lw=.8)
    ax.text(last+.15,7,'预测期',fontsize=9,color=PURPLE)
    title(ax,'(a) 历史前沿与曲线外推')
    ax.legend(loc='upper left',ncol=2,fontsize=8,handlelength=1.4,columnspacing=.8)
    grid(ax)
    ax=fig.add_subplot(gs[1]); scen=read('前沿预测_结构外推情景')
    styles=[('放缓',1,GOLD,'v'),('基准',0,TEAL,'o'),('扩容',2,PURPLE,'s')]
    for name,i,color,mark in styles:
        vals=[scen[scen['前瞻年数'].eq(yr)].iloc[i]['结构外推前沿'] for yr in [1,2]]
        yy=np.array([1,0])+{'放缓':-.19,'基准':0,'扩容':.19}[name]
        ax.scatter(vals,yy,color=color,marker=mark,s=32,label=name)
        for v,z in zip(vals,yy):
            ax.text(v+.4,z,f'{v:.2f}',va='center',fontsize=9)
    ax.set(yticks=[1,0],yticklabels=['2026-03','2027-03'],ylim=(-.5,1.5),xlim=(50,73),xlabel='结构外推得分 / 分')
    title(ax,'(b) 三种结构情景的分歧')
    ax.legend(loc='upper right',ncol=3,fontsize=8,handletextpad=.3,columnspacing=.8)
    grid(ax,'x')
    fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.92)
    save(fig,'图3_能力前沿预测',audit)

def subtasks(audit):
    df=read('C8逐任务聚合')
    assert len(df)==37 and set(df['子任务'])==set(NAMES)
    fig=plt.figure(figsize=(6.2,6.45))
    gs=fig.add_gridspec(3,2,width_ratios=[1.12,1],height_ratios=[7,3,3],wspace=.85,hspace=.76)
    blocks=[(fig.add_subplot(gs[:,0]),'BBH子任务',PURPLE,'(a) BBH · 24项'),
            (fig.add_subplot(gs[0,1]),'MATH子任务',GOLD,'(b) MATH · 7项'),
            (fig.add_subplot(gs[1,1]),'GPQA子集',TEAL,'(c) GPQA · 3项'),
            (fig.add_subplot(gs[2,1]),'MUSR子任务','#38678E','(d) MUSR · 3项')]
    for ax,cat,color,label in blocks:
        d=df[df['类别'].eq(cat)].sort_values('最优得分')
        yy=np.arange(len(d))
        assert (d['中位数']<=d['前5%均值']).all() and (d['前5%均值']<=d['最优得分']).all()
        ax.hlines(yy,d['中位数'],d['最优得分'],color=color,alpha=.40,lw=2.5)
        for col,mark,fill in [('中位数','o','white'),('前5%均值','|',color),('最优得分','D',color)]:
            if mark == '|':
                ax.scatter(d[col],yy,s=55,marker=mark,color=color,linewidths=.9,zorder=3)
            else:
                ax.scatter(d[col],yy,s=20,marker=mark,facecolors=fill,edgecolors=color,linewidths=.9,zorder=3)
        ax.set(yticks=yy,yticklabels=[NAMES[k] for k in d['子任务']],xlim=(-2,103),xticks=[0,50,100],ylim=(-.65,len(d)-.35))
        ax.invert_yaxis(); ax.tick_params(axis='y',length=0,labelsize=8.5)
        title(ax,label); grid(ax,'x')
        ax.set_xlabel('原始指标 / 分',fontsize=9)
    handles=[Line2D([],[],marker='o',mfc='white',mec=INK,ls='',label='中位数'),Line2D([],[],marker='|',color=INK,ls='',label='前5%均值'),Line2D([],[],marker='D',color=INK,ls='',label='最优得分')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.008),ncol=3,fontsize=9)
    fig.subplots_adjust(left=.205,right=.98,bottom=.115,top=.93)
    save(fig,'图4_逐任务聚合',audit)

def macro(audit):
    d=read('C4宏观趋势'); summary=read('C4宏观摘要').iloc[0]
    fig=plt.figure(figsize=(6.2,4.75))
    gs=fig.add_gridspec(2,2,height_ratios=[1.15,1],hspace=.72,wspace=.32)
    ax=fig.add_subplot(gs[0,:])
    ax.vlines(d.year,d['前沿算力_中位'],d['前沿算力_max'],color=TEAL,alpha=.5,lw=3)
    ax.scatter(d.year,d['前沿算力_中位'],s=26,facecolors='white',edgecolors=TEAL,label='年度中位数',zorder=3)
    ax.scatter(d.year,d['前沿算力_max'],marker='D',s=25,color=PURPLE,label='年度最大值',zorder=3)
    ax.axvspan(2025.6,2026.4,color='#F5F0E6')
    ax.text(2026,2e18,'5条算力记录',ha='center',fontsize=8,rotation=90)
    ax.set(yscale='log',ylim=(1e16,5e27),xlim=(2017.6,2026.4),xticks=d.year, ylabel='训练算力 / FLOP')
    ax.set_yticks([1e17,1e20,1e23,1e26]);title(ax,'(a) Language 领域的算力跨度');grid(ax)
    ax.legend(loc='upper left',ncol=2,fontsize=9)
    ax=fig.add_subplot(gs[1,0])
    ax.plot(d.year,d['开源权重占比']*100,color=TEAL,marker='o',ms=4,lw=1.2)
    ax.axvspan(2025.6,2026.4,color='#F5F0E6')
    ax.set(ylim=(0,100),yticks=[0,25,50,75,100],xticks=[2018,2020,2022,2024,2026],xlabel='年份',ylabel='Yes 标记比例 / %')
    title(ax,'(b) 开放权重的收录比例');grid(ax)
    ax=fig.add_subplot(gs[1,1])
    counts=[int(summary['有算力记录行数']),int(summary['有数据量记录行数']),int(summary['开源权重_Yes']+summary['开源权重_No'])]
    ax.barh([2,1,0],[3523]*3,color='#EFECF1',height=.5)
    ax.barh([2,1,0],counts,color=[PURPLE,GOLD,TEAL],height=.5)
    for yy,nn in zip([2,1,0],counts):ax.text(nn+65,yy,f'{nn:,}',va='center',fontsize=9)
    ax.set(yticks=[2,1,0],yticklabels=['算力','数据量','权重状态'],xlim=(0,3800),xticks=[0,1500,3000],xlabel='有效记录数 / 条',ylim=(-.6,2.6))
    title(ax,'(c) C4 全量字段覆盖')
    fig.subplots_adjust(left=.105,right=.98,bottom=.12,top=.92)
    save(fig,'图5_C4宏观趋势',audit)

def distributions(lb,audit):
    fig,ax=plt.subplots(figsize=(6.2,3.15))
    order=['GPQA','MUSR','MATH Lvl 5','BBH','MMLU-PRO','IFEval']
    vals=[lb[t].to_numpy() for t in order]
    # Full range is explicit; no data points are suppressed as outliers.
    bp=ax.boxplot(vals,orientation='horizontal',tick_labels=order,whis=(0,100),showfliers=False,patch_artist=True,widths=.52,
                  medianprops={'color':INK,'lw':1.4},boxprops={'edgecolor':PURPLE},whiskerprops={'color':PURPLE},capprops={'color':PURPLE})
    for patch in bp['boxes']:patch.set_facecolor('#E5DEED')
    ax.invert_yaxis();ax.set(xlim=(0,100),xticks=np.arange(0,101,20),xlabel='C1 基准得分 / 分')
    grid(ax,'x');ax.tick_params(axis='y',length=0)
    fig.subplots_adjust(left=.16,right=.98,bottom=.20,top=.94)
    save(fig,'图6_六维能力分布',audit)

def scale(audit):
    d=read('面板回归数据');alpha=read('年份效应').set_index('Year')['年效应alpha_t'];bp=read('规模时间分解').iloc[1]['bP']
    fig,axes=plt.subplots(1,2,figsize=(6.2,3.15),sharex=True,sharey=True)
    for ax,year,color,mark in zip(axes,[2024,2025],[TEAL,PURPLE],['o','^']):
        sub=d[d.Year.eq(year)]
        ax.scatter(sub.P,sub.avg,c=color,s=6,alpha=.25,marker=mark,edgecolors='none')
        xx=np.geomspace(sub.P.min(),sub.P.max(),150)
        ax.plot(xx,alpha[year]+bp*np.log10(xx),color=INK,lw=1.4)
        ax.set(xscale='log',xlim=(.001,220),ylim=(-35,60),xticks=[.001,.01,.1,1,10,100],xticklabels=['0.001','0.01','0.1','1','10','100'],xlabel='参数量 / B（对数轴）')
        assert sub.P.between(.001,220).all() and sub.avg.between(-35,60).all()
        ax.axhline(0,color='#B6AFBA',lw=.6,ls=':')
        title(ax,f'{"(a)" if year==2024 else "(b)"} {year}年 · n={len(sub):,}')
        grid(ax)
    axes[0].set_ylabel('Leaderboard 均分')
    fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.85,wspace=.12)
    save(fig,'图7_年度规模效应',audit)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--audit',type=Path)
    args=ap.parse_args(); before=hashes();style()
    if args.audit:args.audit.mkdir(parents=True,exist_ok=True)
    src=args.data/'C_efficiency_evolution'
    lb=pd.read_csv(src/'leaderboard_cleaned.csv')
    lb['P']=pd.to_numeric(lb['#Params (B)'],errors='coerce')
    lb['date']=pd.to_datetime(lb['Submission Date'],errors='coerce')
    lb['avg']=pd.to_numeric(lb['Average ⬆️'],errors='coerce')
    lb[TASKS]=lb[TASKS].apply(pd.to_numeric,errors='coerce')
    lb=lb[lb[TASKS].notna().all(axis=1)&lb.date.notna()&lb.P.notna()&lb['Hub License'].astype(str).str.strip().str.lower().isin(LICENSES)].copy()
    assert len(lb)==2346
    td=read('逐任务聚合').set_index('task')
    for task in TASKS:
        s=lb[task]
        assert np.allclose([s.max(),s.nlargest(max(1,int(len(s)*.01))).mean(),s.median()],td.loc[task,['max','top1_mean','median']].astype(float))
    actual=pd.read_csv(src/'loss_benchmark_bridge_expanded.csv')
    saved=read('Loss_Benchmark映射明细')
    assert actual.Model.tolist()==saved.Model.tolist()
    assert np.allclose(actual[['Val_Loss','LB_Average']],saved[['Val_Loss','LB_Average']])
    bridge(args.audit);decomposition(args.audit);forecast(lb,args.audit);subtasks(args.audit)
    macro(args.audit);distributions(lb,args.audit);scale(args.audit)
    assert before==hashes(),'绘图期间结果文件发生改变'
    if args.audit:
        report={'records':len(lb),'years':lb.date.dt.year.value_counts().to_dict(),
                'task_summary':lb[TASKS].describe().to_dict(),'result_files_unchanged':True,
                'chart_contracts':{
                    '图1':'分层散点与已存拟合；仅组内损失范围；两面板纵轴不同。',
                    '图2':'两个窗口的加性贡献；同一纵轴；非规模项为分解残差。',
                    '图3':'历史记录、模型曲线、仅两时点的bootstrap区间；结构情景独立面板。',
                    '图4':'37子任务中位数/前5%均值/最大值；原始指标；按类别分面。',
                    '图5':'年度算力跨度、Yes记录比例、全量字段覆盖；2026覆盖不完整。',
                    '图6':'2346条实际附件记录的四分位区间与完整极差；展示总体分布。',
                    '图7':'年度分面规模散点及既有固定效应直线；未重拟合。'}}
        (args.audit/'plot_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
