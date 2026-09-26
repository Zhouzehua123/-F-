# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6；机构：OpenAI；版本发布日期：2026-04-21。
# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-5.6）辅助。
"""按论文已有分析绘制问题三、四流程图，不读取数据、不运行求解模型。

运行：python 求解/绘制问题三四流程图.py
输出：论文/figures 下的 PDF、SVG、400 dpi PNG 及文字边界核验记录。
"""
from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'huawei-expand-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PURPLE, TEAL, GOLD, BLUE = '#665080', '#287C78', '#BC8030', '#365E8D'
INK, GRAY = '#243348', '#637186'


def configure():
    for name in ('simsun.ttc', 'times.ttf', 'timesi.ttf', 'timesbd.ttf'):
        path = Path('C:/Windows/Fonts') / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    for family in ('SimSun', 'Times New Roman'):
        font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'], 'font.size': 10,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
        'text.color': INK, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
        'svg.fonttype': 'path', 'svg.hashsalt': 'huawei-expand-workflows',
    })


def tint(color, strength):
    return tuple(1 - strength * (1 - value) for value in to_rgb(color))


class Workflow:
    def __init__(self, height):
        self.fig = plt.figure(figsize=(6.5, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 6.5), ylim=(0, height))
        self.ax.axis('off')
        self.labels = []

    def text(self, x, y, text, size=10, color=INK, bounds=None, **kwargs):
        if '$' in text:
            # Mathtext needs a Chinese-capable default font for its non-math runs.
            kwargs['fontfamily'] = 'SimSun'
        artist = self.ax.text(x, y, text, fontsize=size, color=color,
                              ha='center', va='center', linespacing=1.4,
                              zorder=5, **kwargs)
        self.labels.append((artist, bounds))

    def box(self, x, y, w, h, title, content, color, size=10):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=.045',
            facecolor='white', edgecolor=tint(color, .55), linewidth=.85,
            zorder=3))
        self.ax.add_patch(Rectangle(
            (x+.012, y+h-.29), w-.024, .278, facecolor=tint(color, .11),
            edgecolor='none', zorder=4))
        self.text(x+w/2, y+h-.145, title, 10.4, color,
                  bounds=(x+.03, y+h-.28, w-.06, .26))
        self.text(x+w/2, y+(h-.29)/2, content, size,
                  bounds=(x+.04, y+.025, w-.08, h-.33))

    def arrow(self, points, color=GRAY, dashed=False):
        style = (0, (3, 2)) if dashed else '-'
        if len(points) > 2:
            xx, yy = zip(*points[:-1])
            self.ax.plot(xx, yy, color=color, lw=.95, linestyle=style, zorder=2)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle='-|>', mutation_scale=8,
            color=color, lw=.95, linestyle=style, shrinkA=0, shrinkB=0,
            zorder=2))

    def lane(self, y, title, color):
        self.ax.add_patch(FancyBboxPatch(
            (.07, y), 6.36, 1.19,
            boxstyle='round,pad=0,rounding_size=.055',
            facecolor=tint(color, .04), edgecolor=tint(color, .19),
            linewidth=.65, zorder=0))
        self.ax.add_patch(Rectangle((.17, y+.97), .04, .14,
                                   facecolor=color, edgecolor='none'))
        self.text(.29, y+1.04, title, 10.1, color)
        self.labels[-1][0].set_ha('left')

    def save(self, out, stem):
        out.mkdir(parents=True, exist_ok=True)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        errors = []
        for artist, bounds in self.labels:
            box = artist.get_window_extent(renderer)
            if bounds is None:
                outer = self.fig.bbox
            else:
                x, y, w, h = bounds
                from matplotlib.transforms import Bbox
                outer = Bbox.from_bounds(x, y, w, h).transformed(self.ax.transData)
            if (box.x0 < outer.x0-1 or box.x1 > outer.x1+1 or
                    box.y0 < outer.y0-1 or box.y1 > outer.y1+1):
                errors.append(artist.get_text())
        if errors:
            raise ValueError(f'{stem}: text exceeds bounds: {errors}')
        self.fig.savefig(out / f'{stem}.pdf', metadata={
            'Title': stem, 'CreationDate': None, 'ModDate': None})
        svg = out / f'{stem}.svg'
        self.fig.savefig(svg, metadata={'Date': None})
        svg.write_text('\n'.join(line.rstrip() for line in
                                 svg.read_text(encoding='utf-8').splitlines())+'\n',
                       encoding='utf-8')
        self.fig.savefig(out / f'{stem}.png', dpi=400)
        (out / f'{stem}-layout.json').write_text(json.dumps({
            'canvas_inches': list(self.fig.get_size_inches()),
            'text_count': len(self.labels), 'text_bounds_issues': errors,
            'scope': 'Text within nodes and canvas; visual review also required.',
        }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        plt.close(self.fig)
        print(f'{stem}: exported PDF/SVG/PNG, text bounds passed')


def q3(out):
    f = Workflow(5.18)
    f.box(.15, 4.02, 1.96, 1.00, '问题一：质量与配比',
          '配比加权质量 $Q_0$\n配比修正 $M$', PURPLE)
    f.box(2.27, 4.02, 1.96, 1.00, '问题二：损失关系',
          '广义标度律参数\n构建 $L(N,D,Q)$', TEAL)
    f.box(4.39, 4.02, 1.96, 1.00, '题设：预算与费用',
          '训练、质量、注意力\n预算与上下文条件', BLUE)
    for x in (1.13, 3.25, 5.37):
        f.arrow([(x, 3.995), (x, 3.79), (3.25, 3.79), (3.25, 3.72)])
    f.box(.15, 2.77, 6.2, .91, '构建预算约束下的资源配置模型',
          '以预测验证损失最小为目标，合计三类算力费用\n'
          r'$C_{\mathrm{tot}}\leq C_{\mathrm{b}},\quad Q_0\leq Q\leq1$'
          '；固定配比与上下文，统一规模计量口径', PURPLE)
    f.arrow([(3.25, 2.74), (3.25, 2.55)], PURPLE)
    f.box(.15, 1.60, 6.2, .91, '多初值 SLSQP 求解',
          r'采用 $\ln N,\ \ln D$ 与 $Q$；三档预算设 8 组初值，预算扫描设 5 组'
          '\n比较求解成功的结果，取其中损失最低者，并处理质量上界', TEAL)
    for x in (1.13, 3.25, 5.37):
        f.arrow([(3.25, 1.57), (3.25, 1.44), (x, 1.44), (x, 1.36)], TEAL)
    f.box(.15, .23, 1.96, 1.09, '三档预算配置',
          '参数、数据与质量\n预测损失与费用份额\n比较投入结构', PURPLE, 9.7)
    f.box(2.27, .23, 1.96, 1.09, '连续预算扫描',
          '141 个对数网格点\n质量完成度与费用份额\n定位启动、饱和区间', TEAL, 9.7)
    f.box(4.39, .23, 1.96, 1.09, '情景与敏感性分析',
          '三种质量成本\n五档上下文与配比情景\n比较配置与损失变化', GOLD, 9.7)
    f.text(3.25, .09, '扫描中的失败点不补算；阶段边界按有效预算网格报告。', 9, GRAY)
    f.save(out, 'q3-workflow')


def q4(out):
    f = Workflow(5.45)
    rows = [
        (4.14, '损失与得分的联系', PURPLE,
         'C6 分层、C5 对照', '区分记录可比程度\n分别整理损失与得分',
         '分层最小二乘回归', '拟合损失—得分关系\n检查相关性与换算误差',
         '方向判断', '说明变化方向\n保留换算误差边界'),
        (2.81, '历史增量的来源', TEAL,
         'C1 与 C3 分别整理', 'C1 字段与许可筛选\nC3 得分一致性检查',
         '规模、年份回归与分解', '估计规模和年度系数\n按长、短窗口作 shift-share 分解',
         '历史贡献核算', '规模项与非规模项\n分别报告两类窗口'),
        (1.48, '未来前沿的条件预测', BLUE,
         '前沿序列与回归系数', 'C3 年度、C1 月度前沿\n沿用规模和年度系数',
         '两条预测路径', 'logistic 拟合与 bootstrap 区间\n规模、非规模增长的结构情景',
         '前瞻 12 / 24 个月', '曲线预测与区间\n结构情景点值'),
        (.15, '任务瓶颈与宏观参照', GOLD,
         'C1 / C8 与 C4', '六维得分与原始子任务\n宏观资源和字段覆盖',
         '任务比较与资源趋势', 'C1—C8 名称匹配与口径核对\nC4 单独统计算力与开放权重',
         '适用范围说明', '识别评测薄弱项\n说明宏观证据边界'),
    ]
    for y, label, color, t1, b1, t2, b2, t3, b3 in rows:
        f.lane(y, label, color)
        f.box(.17, y+.09, 1.70, .80, t1, b1, color, 9.25)
        f.box(2.09, y+.09, 2.39, .80, t2, b2, color, 9.25)
        f.box(4.71, y+.09, 1.60, .80, t3, b3, color, 9.25)
        f.arrow([(1.90, y+.46), (2.06, y+.46)], color)
        f.arrow([(4.51, y+.46), (4.68, y+.46)], color)
    f.save(out, 'q4-workflow')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / '论文' / 'figures')
    args = parser.parse_args()
    configure()
    q3(args.output_dir)
    q4(args.output_dir)
