"""从已保存的 CSV 重绘论文图；不运行优化器，也不修改原始结果。

用法：python 求解/问题三/论文绘图.py
依赖：numpy、pandas、matplotlib。输出同名 PNG（300 dpi）和矢量 PDF。
缺失求解点保留为 NaN，不插值补造结果。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, ticker

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "结果", ROOT / "图片"
FONT = ROOT.parents[1] / "论文" / "fonts" / "SourceHanSerifCN-Regular.otf"
fm.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": [fm.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "savefig.dpi": 300, "axes.linewidth": .7,
})
BLUE, ORANGE, CYAN, PURPLE = "#356B9A", "#C87530", "#238A8D", "#8064A2"


def read(name):
    return pd.read_csv(DATA / (name + ".csv"))


def finish(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E4E7EA", lw=.55)
        ax.set_axisbelow(True)
        ax.tick_params(which="both", direction="out", length=3)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / (name + "." + ext), bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def panel(ax, text):
    ax.text(0, 1.045, text, transform=ax.transAxes, va="bottom", fontsize=10)


def recover_grid(df, grid, key):
    """将已输出的有效点放回原脚本定义的网格，显式暴露被跳过的失败点。"""
    output = pd.DataFrame({key: grid})
    df = df.select_dtypes(include="number")
    for col in df.columns:
        if col != key:
            output[col] = np.nan
    for _, row in df.iterrows():
        index = np.argmin(abs(np.log(grid / row[key])))
        if not np.isclose(grid[index], row[key], rtol=1e-9):
            raise ValueError(f"{key}={row[key]} 不属于原始扫描网格")
        for col in df.columns:
            if col != key:
                output.loc[index, col] = row[col]
    return output


opt = read("三档预算最优配置").sort_values("C")
q0 = float(opt.iloc[0].Q)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.65), layout="constrained")
for col, color, marker, label in [("N_B", BLUE, "o", "参数量（B）"),
                                   ("D_B", CYAN, "s", "数据量（B tokens）")]:
    ax[0].plot(opt.C, opt[col], marker=marker, ls="--", color=color, lw=1.4,
               ms=5, label=label)
    for x, y in zip(opt.C, opt[col]):
        label_value = f"{y:.1f}" if col == "D_B" else f"{y:.3f}"
        ax[0].annotate(label_value, (x, y), xytext=(0, 8 if col == "D_B" else -13),
                       textcoords="offset points", ha="center", fontsize=8, color=color)
ax[0].set_yscale("log")
ax[0].set_ylim(.045, 10000)
ax[0].set_ylabel("规模（对数刻度）")
ax[0].legend(loc="upper left")
panel(ax[0], "(a) 参数量与数据量")
ax[1].plot(opt.C, opt.Q, "o--", color=ORANGE, lw=1.4, ms=5)
ax[1].axhline(q0, color="#777777", ls=":", lw=1)
for x, y in zip(opt.C, opt.Q):
    ax[1].annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=8)
ax[1].text(1.4e20, q0-.065, r"基线 $Q_0=0.551$", fontsize=9, color="#666666")
ax[1].set_ylim(.45, 1.12)
ax[1].set_ylabel(r"最优质量 $Q^*$")
panel(ax[1], "(b) 质量水平")
for a in ax:
    a.set_xscale("log")
    a.set_xlim(3.5e18, 3e24)
    a.set_xticks(opt.C)
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.set_xlabel(r"算力预算 $C_{\mathrm{b}}$（FLOPs）")
finish(fig, "图1_三档预算最优配置")

sw = read("预算扫略").sort_values("C")
theta = (sw.Q-q0)/(1-q0)
valid = sw.Q.notna()
start = float(sw.loc[valid & (sw.Q > q0+1e-7), "C"].iloc[0])
sat = float(sw.loc[valid & (sw.Q >= 1-1e-7), "C"].iloc[0])
peak = sw.loc[sw.f_qual.idxmax()]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.35), layout="constrained")
a, b = axs
a.plot(sw.C, theta*100, color=ORANGE, lw=1.8)
for col, color, ls, label in [("f_base", BLUE, "-", "基础训练"),
                             ("f_qual", ORANGE, "--", "质量提升"),
                             ("f_attn", PURPLE, "-.", "注意力计算")]:
    b.plot(sw.C, sw[col]*100, color=color, ls=ls, lw=1.6, label=label)
for aa in axs:
    aa.set_xscale("log")
    aa.set_xlim(1e18, 1e25)
    aa.set_ylim(-3, 108)
    aa.set_xticks([1e18, 1e20, 1e22, 1e24])
    aa.xaxis.set_minor_locator(ticker.NullLocator())
    aa.axvline(start, color="#777777", lw=.8, ls="--")
    aa.axvline(sat, color="#777777", lw=.8, ls=":")
    aa.axvspan(start, sat, color=ORANGE, alpha=.07)
    aa.set_xlabel(r"算力预算 $C_{\mathrm{b}}$（FLOPs）")
a.set_ylabel(r"质量完成度 $\Theta$（%）")
b.set_ylabel("预算份额（%）")
panel(a, "(a) 质量由基线升至上界")
panel(b, "(b) 质量份额先升后降")
b.legend(loc="upper right", bbox_to_anchor=(1,.78), fontsize=8, framealpha=.95)
b.annotate("峰值 40.7%", (peak.C, peak.f_qual*100), xytext=(2e21, 42),
           fontsize=8, arrowprops={"arrowstyle":"-", "color":ORANGE, "lw":.8})
zoom = a.inset_axes([.40, .14, .57, .50])
zoom.plot(sw.C, theta*100, color=ORANGE, lw=1.4)
zoom.set_xscale("log")
zoom.set_xlim(1e19, 1e20)
zoom.set_ylim(-4,104)
zoom.set_xticks([1e19,1e20])
zoom.set_xticklabels([r"$10^{19}$",r"$10^{20}$"],fontsize=8)
zoom.xaxis.set_minor_locator(ticker.NullLocator())
zoom.set_yticks([0,50,100])
zoom.tick_params(labelsize=8)
zoom.axvline(start,color="#777777",ls="--",lw=.8)
zoom.axvline(sat,color="#777777",ls=":",lw=.8)
zoom.text(.04,.9,"转移区间放大",transform=zoom.transAxes,fontsize=8)
zoom.grid(axis="y",color="#e4e7ea",lw=.5)
finish(fig, "图2_预算份额结构性转移")

ctx = read("上下文长度敏感性")
cont = recover_grid(read("上下文长度连续扫描"), np.geomspace(1024, 200000, 40), "L_ctx")
ql = read("上下文长度与最优质量")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.85), layout="constrained")
ax[0].plot(cont.L_ctx, cont.N_B, color=BLUE, lw=1.8)
ax[0].plot(ctx.L_ctx, ctx.N_B, "s", color=BLUE, ms=4, label="C7 五档配置")
ax[0].set_ylabel(r"最优参数量 $N^*$（B）")
ax[0].set_ylim(2, 6.6)
ax[0].legend(loc="lower left", fontsize=8)
panel(ax[0], r"(a) 规模配置：$10^{22}$ FLOPs")
for budget, color, ls, marker, label in [(1e19, ORANGE, "-", "o", r"$10^{19}$ FLOPs"),
                                        (1e20, BLUE, "-", "s", r"$10^{20}$ FLOPs"),
                                        (1e21, CYAN, "--", None, r"$10^{21}$ FLOPs")]:
    d = recover_grid(ql.loc[ql.C.eq(budget)], np.geomspace(1024, 150000, 14), "L_ctx")
    ax[1].plot(d.L_ctx, d.Q, color=color, ls=ls, marker=marker, ms=3,
               lw=1.6, label=label)
ax[1].set_ylim(.49, 1.085)
ax[1].set_ylabel(r"最优质量 $Q^*$")
ax[1].text(1600, .925, r"$10^{20}$ 与 $10^{21}$ 曲线重合", fontsize=8)
ax[1].legend(loc="center left", fontsize=8)
panel(ax[1], "(b) 不同预算下的质量响应")
for a in ax:
    a.set_xscale("log")
    a.set_xlim(850, 220000)
    a.axvline(30000, color=PURPLE, ls=":", lw=1.1)
    a.set_xticks([1e3, 1e4, 3e4, 1e5])
    a.set_xticklabels(["$10^3$", "$10^4$", "$3×10^4$", "$10^5$"])
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.set_xlabel(r"上下文长度 $L_{\mathrm{ctx}}$（tokens）")
finish(fig, "图3_上下文长度敏感性")

fig, ax = plt.subplots(figsize=(5.0, 2.7), layout="constrained")
q = np.linspace(q0, 1, 300)
for func, color, ls, label in [(lambda x: 5e9*x**4, BLUE, "-", "幂函数型"),
                               (lambda x: 1e7*np.exp(6*x), ORANGE, "--", "指数型"),
                               (lambda x: 2e9*np.log1p(10*x), CYAN, "-.", "对数渐进型")]:
    ax.plot(q, (func(q)-func(q0))/1e9, color=color, ls=ls, lw=1.8, label=label)
ax.set_xlabel(r"数据质量 $Q$")
ax.set_ylabel("增量质量成本（$10^9$ FLOPs/token）")
ax.set_xlim(q0-.015, 1.01)
ax.set_ylim(-.12, 4.85)
ax.legend(loc="upper left")
ax.annotate(r"$Q_0=0.551$", (q0, 0), xytext=(8, 13), textcoords="offset points", fontsize=9)
finish(fig, "图4_质量成本函数对比")

print(f"已生成四组 PNG/PDF；扫描点 {len(sw)}，有效点 {valid.sum()}。")
print(f"首次正投入 {start:.6g}；首次饱和 {sat:.6g}；份额峰值 {peak.f_qual:.6f}。")
