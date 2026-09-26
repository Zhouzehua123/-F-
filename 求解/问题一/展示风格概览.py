# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-6）辅助。
"""问题一质量、配方结构与拟合概览的展示图。

仅读取绘图数据模块提供的既有结果；图中均值、标准差、配方占比及
拟合统计均按原值展示。学术图表规范来源与项目字体约定同公共模块。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from 绘图数据 import DOMAIN_NAMES
from 展示风格公共 import (
    PURPLE, TEAL, GOLD, BLUE, INK, GRAY, GRID, configure, panel, clean, save,
)


STEMS = ['图0_A1-A3全量质量评分', '图1_训练配比结构', '图2_线性模型R2']


def _names(domains):
    return [DOMAIN_NAMES[name] for name in domains]


def plot_quality(data, out):
    q = data.quality.sort_values('quality_mean', ascending=False).reset_index(drop=True)
    short_labels = {
        'arxiv': '学术\n预印本', 'stackexchange': '专业\n问答',
        'commoncrawl': 'Common\nCrawl', 'github': '代码',
        'wikipedia': '百科', 'book': '图书', 'c4': 'C4',
    }
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colors = [PURPLE] + [TEAL] * (len(q) - 1)
    ax.bar(x, q.quality_mean, width=.66, color=colors, edgecolor='white',
           linewidth=.6, zorder=3)
    ax.errorbar(x, q.quality_mean, yerr=q.quality_std, fmt='none',
                ecolor=INK, elinewidth=.9, capsize=3, capthick=.9, zorder=4)
    for pos, row in q.iterrows():
        ax.text(pos, row.quality_mean + row.quality_std + .025,
                f'{row.quality_mean:.3f}', fontsize=9.5,
                ha='center', va='bottom', color=colors[pos])
        ax.text(pos, .055, f'n =\n{int(row.n):,}', fontsize=9,
                ha='center', va='bottom', linespacing=1.2, color='white', zorder=5)
    ax.set(xticks=x, xticklabels=[short_labels[name] for name in q.domain],
           ylabel='质量评分 Q', ylim=(0, .87), yticks=[0, .2, .4, .6, .8],
           xlim=(-.65, len(q)-.35))
    ax.text(.99, .985, '误差线：标准差', transform=ax.transAxes,
            fontsize=9, color=GRAY, ha='right', va='top')
    ax.spines['bottom'].set_color(GRID)
    clean(ax, axis='y')
    panel(ax, 'A', '语料领域的质量评分', PURPLE)
    fig.subplots_adjust(left=.11, right=.975, bottom=.18, top=.82)
    save(fig, STEMS[0], out)


def plot_mixture(data, out):
    order = data.reference.sort_values(ascending=False).index
    assert len(order) == 17
    matrix = data.train_x.loc[:, order].iloc[:8].to_numpy().T
    assert matrix.shape == (17, 8)
    values = data.reference.loc[order].to_numpy()
    fig = plt.figure(figsize=(5.0, 3.65))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.30, 1],
                           left=.20, right=.98, bottom=.21, top=.86, wspace=.18)
    ax = fig.add_subplot(grid[0])
    right = fig.add_subplot(grid[1], sharey=ax)
    cmap = LinearSegmentedColormap.from_list(
        'q1_mixture_share', ['#F4F7FA', '#D4E8E6', '#68A5A0', TEAL, PURPLE])
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap,
                      interpolation='nearest', aspect='auto')
    ax.set(xticks=np.arange(8),
           xticklabels=[str(i) for i in data.train_x.index[:8]],
           yticks=np.arange(17), yticklabels=_names(order), xlabel='配方编号',
           ylim=(16.5, -.5))
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 17, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.65)
    ax.tick_params(which='both', length=0, pad=5)
    for row, col in zip(*np.where(matrix >= .10)):
        value = matrix[row, col]
        ax.text(col, row, f'{value:.0%}', ha='center', va='center', fontsize=9,
                color='white' if value >= .45 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel(ax, 'A', '前 8 组训练配方', PURPLE)

    y = np.arange(17)
    right.barh(y, values, color=TEAL, height=.62, edgecolor='white', linewidth=.5,
               zorder=3)
    for pos, value in enumerate(values):
        right.text(value + .005, pos, f'{value:.1%}',
                   ha='left', va='center', fontsize=9, color=TEAL)
    right.set(xlim=(0, .17), xticks=[0, .05, .10, .15],
              xlabel='平均训练占比')
    right.tick_params(axis='y', labelleft=False)
    right.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.spines['left'].set_visible(False)
    clean(right, bands=True, n=17)
    right.tick_params(axis='y', which='both', length=0)
    panel(right, 'B', '平均领域占比', TEAL)
    cax = fig.add_axes([.205, .055, .38, .020])
    bar = fig.colorbar(image, cax=cax, orientation='horizontal',
                      ticks=[0, .25, .5, .75, 1], format=PercentFormatter(1, decimals=0))
    bar.ax.tick_params(length=2, labelsize=9, pad=2)
    bar.outline.set_linewidth(.4)
    fig.text(.64, .065, '配方实际占比', ha='left', va='center', fontsize=9, color=GRAY)
    save(fig, STEMS[1], out)


def plot_fit(data, out):
    m = data.metrics.sort_values('test1m_R2', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 3.45), sharey=True)
    y = np.arange(len(m))
    specs = [
        (axes[0], 'train_R2', 'test1m_R2', '训练 1M', '检验 1M',
         PURPLE, TEAL, 'o', 'o', '决定系数 R²'),
        (axes[1], 'test60m_r', 'test1B_r', '检验 60M', '检验 1B',
         BLUE, GOLD, '^', 's', 'Pearson 相关系数 r'),
    ]
    for ax, col1, col2, lab1, lab2, color1, color2, marker1, marker2, xlabel in specs:
        ax.hlines(y, np.minimum(m[col1], m[col2]), np.maximum(m[col1], m[col2]),
                  color='#B9C5D3', linewidth=1.35, zorder=2)
        ax.scatter(m[col1], y, s=29, marker=marker1,
                   color=color1, edgecolor='white', linewidth=.45, label=lab1, zorder=4)
        ax.scatter(m[col2], y, s=33, marker=marker2,
                   facecolor='none', edgecolor=color2, linewidth=1.05, label=lab2, zorder=5)
        ax.set(xlim=(0, 1), xticks=[0, .25, .5, .75, 1],
               ylim=(len(m)-.6, -.6), xlabel=xlabel)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.27), ncol=2,
                  fontsize=9, handletextpad=.28, columnspacing=.8, borderaxespad=0)
        clean(ax, bands=True, n=len(m))
    axes[0].set(yticks=y, yticklabels=_names(m.index))
    axes[1].tick_params(axis='y', labelleft=False)
    axes[1].spines['left'].set_visible(False)
    panel(axes[0], 'A', '同尺度拟合与检验', PURPLE)
    panel(axes[1], 'B', '跨尺度相关性', BLUE)
    fig.subplots_adjust(left=.20, right=.96, bottom=.25, top=.85, wspace=.18)
    save(fig, STEMS[2], out)


if __name__ == '__main__':
    import argparse
    import sys
    from 绘图数据 import load_plot_data, verify_inputs_unchanged
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    configure()
    data = load_plot_data(args.data_dir)
    for draw in [plot_quality, plot_mixture, plot_fit]:
        draw(data, args.output_dir)
    verify_inputs_unchanged(data)
    print(f'{len(data.checks)} 项数据核对通过；已绘制 3 张概览图。')
