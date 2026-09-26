# 本程序的整理与核对使用 Codex 辅助；模型：GPT-6 系列；机构：OpenAI；系列首次发布日期：2026-09-03。
# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-6）辅助。
"""问题一双泳道流程图；仅调整展示，不运行模型或读取原始附件。

使用 academic-research-suite 的 visualization_agent 绘图检查规范；
中文宋体、英文 Times New Roman 与竞赛图注格式遵循本项目要求。
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import to_rgb

from 展示风格公共 import PURPLE, TEAL, GOLD, BLUE, INK, GRAY, GRID, save


def _tint(color, strength=.075):
    """Blend a group color into the white paper background."""
    return tuple(1 - strength * (1 - value) for value in to_rgb(color))


def plot_workflow(data, out):
    """Render the existing Q1 procedure as two clearly connected swimlanes.

    ``data`` is accepted for compatibility with the numeric figure generators.
    There are no computed quantities in this conceptual figure.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 6.5), ylim=(0, 4.5))
    ax.axis('off')

    def lane(y, h, label, color):
        ax.add_patch(FancyBboxPatch(
            (.08, y), 6.34, h,
            boxstyle='round,pad=0,rounding_size=.06',
            facecolor=_tint(color, .045), edgecolor=_tint(color, .22),
            linewidth=.7, zorder=0))
        ax.add_patch(Rectangle((.08, y + h - .33), .045, .22,
                               facecolor=color, edgecolor='none', zorder=1))
        ax.text(.23, y + h - .22, label, fontsize=10.5, color=color,
                ha='left', va='center')

    def node(x, y, w, h, heading, body, color, body_size=9.2):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=.035',
            facecolor='white', edgecolor=_tint(color, .55),
            linewidth=.85, zorder=3))
        ax.add_patch(Rectangle((x + .012, y + h - .27), w - .024, .258,
                               facecolor=_tint(color, .105), edgecolor='none',
                               zorder=4))
        ax.text(x + w/2, y + h - .139, heading, fontsize=9.4,
                color=color, ha='center', va='center', zorder=5)
        ax.text(x + w/2, y + (h - .27)/2 - .006, body, fontsize=body_size,
                color=INK, ha='center', va='center', linespacing=1.33, zorder=5)

    def arrow(points, color=GRAY, dashed=False):
        linestyle = (0, (3.2, 2.1)) if dashed else '-'
        if len(points) > 2:
            xx, yy = zip(*points[:-1])
            ax.plot(xx, yy, color=color, linewidth=.95,
                    linestyle=linestyle, solid_capstyle='round', zorder=2)
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle='-|>', mutation_scale=8,
            linewidth=.95, color=color, linestyle=linestyle,
            shrinkA=0, shrinkB=0, zorder=2))

    lane(2.14, 2.21, '质量评价', PURPLE)
    lane(.44, 1.57, '配比回归', TEAL)

    # Upper lane: normalized quality signals drive both scoring and diagnostics.
    node(.20, 2.95, 1.88, 1.01, 'A1—A3  质量信号',
         '22 项指标的缺失处理\n列表标量化、方向统一\n归一化并保留领域标签', PURPLE)
    node(2.55, 3.06, 1.50, .90, '熵权综合评价',
         '样本与语料评分\n领域质量评分', PURPLE)
    node(4.54, 3.06, 1.72, .90, 'A16  领域映射',
         '构造配比加权\n质量指数', PURPLE)
    node(.20, 2.18, 1.88, .64, '指标冲突诊断',
         '成对标准化差异\nKendall 排序一致性', GOLD, body_size=9)

    arrow([(2.11, 3.50), (2.50, 3.50)], PURPLE)
    arrow([(4.08, 3.50), (4.49, 3.50)], PURPLE)
    arrow([(1.14, 2.92), (1.14, 2.85)], GOLD)

    # Lower lane: exactly the same four stages as the existing manuscript.
    y, w, h = .59, 1.40, 1.04
    node(.18, y, w, h, 'A4—A5 配方与损失',
         '按 index 对齐\n17 域训练配比\n13 域验证损失', TEAL, body_size=9)
    node(1.76, y, w, h, '岭回归与核岭',
         '线性系数解释\n核岭同尺度预测', TEAL)
    node(3.34, y, w, h, '检验与跨尺度',
         '1M / 60M / 1B 检验\n10B / 70B 外推表', BLUE, body_size=9)
    node(4.92, y, w, h, '线性有界推荐',
         '固定未测量域\n其余领域配比\n受设定上限约束', GOLD, body_size=9)
    for right, left in [(1.58, 1.76), (3.16, 3.34), (4.74, 4.92)]:
        arrow([(right + .025, 1.105), (left - .025, 1.105)], TEAL)

    # The quality term is an incremental diagnostic, not the sole regression input.
    arrow([(5.40, 3.02), (5.40, 2.20), (2.46, 2.20), (2.46, 1.66)],
          color=PURPLE, dashed=True)
    ax.text(3.88, 2.31, '追加质量项的增量检验', fontsize=9,
            color=PURPLE, ha='center', va='center',
            bbox={'boxstyle': 'square,pad=.15',
                  'facecolor': _tint(PURPLE, .045), 'edgecolor': 'none'}, zorder=5)

    # Compact line-style key; figure number and caption remain in the LaTeX file.
    arrow([(.22, .21), (.62, .21)], GRAY)
    ax.text(.72, .21, '主要处理流程', fontsize=9, va='center', color=GRAY)
    arrow([(3.08, .21), (3.48, .21)], PURPLE, dashed=True)
    ax.text(3.58, .21, '质量项增量检验', fontsize=9, va='center', color=GRAY)

    save(fig, '图8_问题一求解流程', out)


if __name__ == '__main__':
    import argparse
    from 展示风格公共 import configure
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    configure()
    plot_workflow(None, args.output_dir)
