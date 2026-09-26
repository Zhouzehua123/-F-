# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6-Luna；机构：OpenAI；版本发布日期：2026-07-09。
# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-5.6-Luna）辅助。
"""用原有前 12 对冲突关系和 22 项熵权绘制网络与权重面板。"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from 展示风格公共 import configure, save, panel, PURPLE, TEAL, GOLD, BLUE, INK, GRAY, GRID


def plot_conflicts(data, out):
    """节点等大、连线宽度从零按冲突率线性编码，不推断其他连接。"""
    weights = data.weights.sort_values('weight', ascending=False)
    key = data.weights.set_index('indicator')
    pairs = data.conflicts.sort_values('conflict_rate', ascending=False).head(12)
    edges = [(key.loc[a, 'code'], key.loc[b, 'code'], float(row.conflict_rate))
             for row in pairs.itertuples()
             for a, b in [row.indicator_pair.split('|')]]
    nodes = {node for edge in edges for node in edge[:2]}
    # A fixed tree layout keeps every observed edge visible and avoids crossings.
    positions = {
        'M15': (.39, .64), 'M14': (.05, .92), 'M06': (.54, .92),
        'M02': (.93, .69), 'M04': (.04, .63),
        'M05': (.05, .38), 'M19': (.37, .30), 'M01': (.62, .40),
        'M22': (.95, .38), 'M16': (.74, .14), 'M03': (1.04, .13),
        'M12': (.38, .08), 'M18': (.06, .08),
    }
    if nodes != set(positions):
        raise ValueError('冲突指标集合改变，需要重新检查固定网络布局。')
    kind = data.weights.set_index('code').list_type.to_dict()

    fig = plt.figure(figsize=(6.5, 5.8))
    network = fig.add_axes([.035, .14, .415, .71])
    bars = fig.add_axes([.73, .14, .245, .71])
    # A separate header spans both the bar labels and plotting area.
    header = fig.add_axes([.48, .14, .495, .71], frameon=False)
    header.set_axis_off()
    panel(network, 'A', '高冲突指标网络', color=GOLD)
    panel(header, 'B', '22 项质量指标的熵权', color=PURPLE)
    network.set(xlim=(-.055, 1.135), ylim=(-.015, 1.06))
    network.set_axis_off()

    for a, b, value in edges:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        network.plot([x1, x2], [y1, y2], color=GOLD, alpha=.65,
                     linewidth=6 * value, zorder=1, solid_capstyle='round')
        # Offset perpendicular to each edge by a small screen-space distance.
        dx = (x2-x1) * .415 * 6.5
        dy = (y2-y1) * .71 * 5.8
        length = np.hypot(dx, dy)
        offset = (-dy/length*5.8, dx/length*5.8)
        network.annotate(f'{value:.1%}', ((x1+x2)/2, (y1+y2)/2),
                         xytext=offset, textcoords='offset points',
                         ha='center', va='center', fontsize=9, color=GRAY,
                         bbox={'facecolor':'white', 'edgecolor':'none', 'pad':.35}, zorder=3)
    for node in sorted(nodes):
        color = PURPLE if kind[node] else TEAL
        network.scatter(*positions[node], s=475, color=color, edgecolor='white',
                        linewidth=1.2, zorder=4)
        network.text(*positions[node], node, color='white', fontsize=9,
                     ha='center', va='center', zorder=5)

    y = np.arange(len(weights))
    colors = [PURPLE if is_list else TEAL for is_list in weights.list_type]
    for i in range(0, len(weights), 2):
        bars.axhspan(i-.5, i+.5, color='#F4F6FA', zorder=0)
    bars.barh(y, weights.weight, color=colors, height=.58, edgecolor='none', zorder=2)
    for i, value in enumerate(weights.weight):
        bars.text(value+.006, i, f'{value:.1%}', color=INK, fontsize=9,
                  ha='left', va='center')
    bars.set(yticks=y,
             yticklabels=[f'{row.code}  {row.display_name}' for row in weights.itertuples()],
             ylim=(21.7, -.7), xlim=(0, .19), xticks=[0,.05,.10,.15],
             xlabel='熵权占比')
    bars.tick_params(axis='y', length=0, pad=5, labelsize=9)
    bars.tick_params(axis='x', length=0, labelsize=9)
    bars.spines['left'].set_visible(False)
    bars.spines['bottom'].set_visible(False)
    bars.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bars.grid(axis='x', color=GRID, linewidth=.55)

    handles = [Line2D([], [], marker='o', color='none', markerfacecolor=color,
                      markeredgecolor='none', markersize=6, label=label)
               for color, label in [(PURPLE, '列表型指标'), (TEAL, '标量型指标')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.50, .025),
               ncol=2, columnspacing=2.5, handletextpad=.5, fontsize=9)
    network.text(.5, -.065, '前 12 对冲突关系；线宽正比于冲突率',
                 transform=network.transAxes, ha='center', va='top', fontsize=9, color=GRAY)
    save(fig, '图6_冲突与赋权诊断', Path(out))


if __name__ == '__main__':
    import argparse
    from 绘图数据 import load_plot_data, verify_inputs_unchanged
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    configure()
    source = load_plot_data(args.data_dir)
    plot_conflicts(source, args.output_dir)
    verify_inputs_unchanged(source)
