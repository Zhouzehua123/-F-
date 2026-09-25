# 本绘图程序及代码在人工智能工具 Codex 辅助下完成。
# 开发机构：OpenAI；模型：GPT-6 系列（2026-09-03 首次发布）。
# 仅读取既有附件、参数和结果绘图，不重新拟合或修改结果数据。
"""读取既有 C8 聚合结果，绘制全部子任务的中位数与最优得分；不修改结果文件。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent

def main():
    df = pd.read_csv(HERE / '结果/C8逐任务聚合.csv')
    assert len(df) == 37 and df['子任务'].is_unique
    cols = ['中位数', '最优得分']
    assert df[cols].notna().all().all()
    assert df[cols].ge(0).all().all() and df[cols].le(100).all().all()
    assert df['最优得分'].ge(df['中位数']).all()
    for name in ['simsun.ttc', 'times.ttf']:
        font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts') / name))
    plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'], 'font.size': 12,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .7})
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    styles = [('BBH子任务', '#665080', 'o'), ('MATH子任务', '#CB8B38', '^'),
              ('GPQA子集', '#367E73', 's'), ('MUSR子任务', '#38678E', 'D')]
    # The distance above the diagonal is a within-task score gap, not uncertainty.
    ax.fill_between([0, 100], [0, 100], [100, 100], color='#F4F2F6', zorder=0)
    ax.plot([0, 100], [0, 100], color='#888888', ls='--', lw=.9, zorder=1)
    for category, color, marker in styles:
        sub = df[df['类别'].eq(category)]
        assert len(sub) > 0
        ax.scatter(sub['中位数'], sub['最优得分'], label=f'{category}（{len(sub)} 项）',
                   c=color, marker=marker, s=48, alpha=.85, edgecolors='white', linewidths=.5)
    ax.set(xlim=(0, 102), ylim=(0, 103), xticks=np.arange(0, 101, 20),
           yticks=np.arange(0, 101, 20), xlabel='各子任务的中位数 / 分', ylabel='各子任务的最优得分 / 分')
    ax.grid(color='#DDDAE2', lw=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='none', facecolor='white')
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.98)
    fig.text(.12, .035, '每点对应一个子任务；采用 C8 原始指标口径。\n虚线表示中位数与最优得分相等，不表示跨任务难度等价。', fontsize=11)
    output = HERE / '图片/图4_逐任务聚合.png'
    fig.savefig(output, dpi=400, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    print(df.groupby('类别').size().to_dict())
    print(output)

if __name__ == '__main__':
    main()
