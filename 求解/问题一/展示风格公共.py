# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6-Luna；机构：OpenAI；版本发布日期：2026-07-09。
# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-5.6-Luna）辅助。
"""Q1 展示版的字体、分组标题和导出约定。"""
from pathlib import Path
import json
import os

# Keep matplotlib's cache in a writable, task-local directory.
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'__pycache__'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.text import Text

PURPLE, TEAL, GOLD, BLUE = '#665080', '#287C78', '#BC8030', '#365E8D'
INK, GRAY, GRID = '#243348', '#637186', '#E6EBF1'


def configure():
    for name in ['simsun.ttc','times.ttf','timesi.ttf','timesbd.ttf']:
        path=Path('C:/Windows/Fonts')/name
        if path.exists():
            fm.fontManager.addfont(str(path))
    for family in ['SimSun','Times New Roman']:
        fm.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        'font.family':['Times New Roman','SimSun'], 'font.size':10,
        'axes.labelsize':10, 'axes.titlesize':11, 'xtick.labelsize':9,
        'ytick.labelsize':9, 'legend.fontsize':9, 'mathtext.fontset':'stix',
        'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.edgecolor':'#9AA7B7','axes.linewidth':.6,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':INK,
        'grid.color':GRID,'grid.linewidth':.55,'legend.frameon':False,
        'axes.axisbelow':True, 'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','savefig.dpi':400,'svg.fonttype':'path',
        'svg.hashsalt':'Q1-ARS-display-v2',
    })


def panel(ax, letter, title, color=BLUE):
    """A compact group header; keep enough top margin for its 11 pt title."""
    ax.add_patch(Rectangle((0,1.035),1,.095,transform=ax.transAxes,
                            facecolor=color,alpha=.075,edgecolor='none',clip_on=False))
    ax.text(.015,1.082,letter,transform=ax.transAxes,va='center',ha='left',
             fontsize=11,fontweight='bold',color=color)
    ax.text(.105,1.082,title,transform=ax.transAxes,va='center',ha='left',
             fontsize=10.5,color=INK)


def clean(ax, axis='x', bands=False, n=None):
    ax.grid(axis=axis)
    ax.tick_params(length=0,pad=5)
    if bands:
        for y in range(0,n,2):
            ax.axhspan(y-.5,y+.5,color='#F4F6FA',zorder=-2)


def save(fig, stem, out):
    """Export fixed-size, 400 dpi PNG/SVG and a canvas-boundary audit."""
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    canvas=fig.bbox
    issues=[]
    # Tick labels outside the current axis interval may be invisible after clipping.
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if text.axes is not None and text.get_clip_on():
            continue
        b=text.get_window_extent(renderer)
        if b.width and b.height and (b.x0<canvas.x0-1 or b.x1>canvas.x1+1 or
                                    b.y0<canvas.y0-1 or b.y1>canvas.y1+1):
            issues.append({'text':text.get_text(),'bbox':[round(v,2) for v in b.bounds]})
    fig.savefig(out/f'{stem}.png',dpi=400)
    svg=out/f'{stem}.svg'
    fig.savefig(svg,metadata={'Date':None,'Creator':'Q1 ARS visualization workflow'})
    svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    # This only checks canvas bounds, not semantic correctness or internal overlaps.
    (out/f'{stem}_布局核验.json').write_text(json.dumps({
        'canvas_inches':list(fig.get_size_inches()),'dpi':400,
        'boundary_issues':issues,'scope':'canvas text bounds only; visual review still required'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
    if issues:
        print(f'{stem}: {len(issues)} canvas-boundary item(s) need review',flush=True)
