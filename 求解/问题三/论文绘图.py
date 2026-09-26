# 本程序的整理与核对使用 Codex 辅助；模型：GPT-6 系列；机构：OpenAI；系列首次发布日期：2026-09-03。
# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-6）辅助。
"""从已保存的 CSV 重绘论文图；不运行优化器，也不修改原始结果。

用法：python 求解/问题三/论文绘图.py
依赖：numpy、pandas、matplotlib。输出同名 PNG（300 dpi）和矢量 PDF。
圆点对应 CSV 有效记录；缺失区间用虚线连接两端，仅引导视线，不补算或插值。
"""
from pathlib import Path
import hashlib
import json
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


AUDIT = {}

def read(name):
    path = DATA / (name + ".csv")
    df = pd.read_csv(path)
    AUDIT[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(df)}
    return df


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


def measured_line(ax, x, y, *, color, lw=1.3, ls="-", label=None, marker="o", ms=2.5):
    """实心标记只画有效记录；跨缺测的连接单独用点线，不生成任何中间数据。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    ax.plot(x, y, color=color, lw=lw, ls=ls, label=label)
    ax.plot(x[good], y[good], ls="none", marker=marker, ms=ms, color=color)
    ids = np.flatnonzero(good)
    for left, right in zip(ids[:-1], ids[1:]):
        if right-left > 1:
            ax.plot(x[[left,right]], y[[left,right]], color=color, lw=lw, ls=":")


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
ax[1].text(1.4e20, q0-.065, rf"基线 $Q_0={q0:.3f}$", fontsize=9, color="#666666")
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
measured_line(a, sw.C, theta*100, color=ORANGE)
for col, color, ls, label in [("f_base", BLUE, "-", "基础训练"),
                             ("f_qual", ORANGE, "--", "质量提升"),
                             ("f_attn", PURPLE, "-.", "注意力计算")]:
    measured_line(b, sw.C, sw[col]*100, color=color, ls=ls, label=label, ms=2)
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
b.annotate(f"峰值 {peak.f_qual*100:.1f}%", (peak.C, peak.f_qual*100), xytext=(2e21, 42),
           fontsize=8, arrowprops={"arrowstyle":"-", "color":ORANGE, "lw":.8})
zoom = a.inset_axes([.40, .14, .57, .50])
measured_line(zoom, sw.C, theta*100, color=ORANGE, ms=3)
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
measured_line(ax[0], cont.L_ctx, cont.N_B, color=BLUE, label="扫描有效记录")
ax[0].plot(ctx.L_ctx, ctx.N_B, "s", color=BLUE, ms=4, label="C7 五档配置")
ax[0].set_ylabel(r"最优参数量 $N^*$（B）")
ax[0].set_ylim(2, 6.6)
ax[0].legend(loc="lower left", fontsize=8)
panel(ax[0], r"(a) 规模配置：$10^{22}$ FLOPs")
for budget, color, ls, marker, label in [(1e19, ORANGE, "-", "o", r"$10^{19}$ FLOPs"),
                                        (1e20, BLUE, "-", "s", r"$10^{20}$ FLOPs"),
                                        (1e21, CYAN, "--", "^", r"$10^{21}$ FLOPs")]:
    d = recover_grid(ql.loc[ql.C.eq(budget)], np.geomspace(1024, 150000, 14), "L_ctx")
    measured_line(ax[1], d.L_ctx, d.Q, color=color, ls=ls, marker=marker, ms=3, label=label)
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

# 图4只使用成本函数对比.csv的九组求解记录，不再绘制解析函数采样线。
comp = read("成本函数对比")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.85), layout="constrained")
budgets = sorted(comp.C.unique())
x = np.arange(len(budgets))
for i, (form, color) in enumerate([("幂函数型", BLUE), ("指数型", ORANGE), ("对数渐进型", CYAN)]):
    d = comp.loc[comp.form.eq(form)].sort_values("C")
    assert d.C.tolist() == budgets and len(d) == 3
    for a, column, factor in [(ax[0], "Q", 1), (ax[1], "f_qual", 100)]:
        bars = a.bar(x+(i-1)*.24, d[column]*factor, width=.22, color=color, label=form)
        a.bar_label(bars, labels=[f"{v:.3f}" if column == "Q" else f"{v:.1f}" for v in d[column]*factor], fontsize=8, padding=3, rotation=90)
for a in ax:
    a.set_xticks(x, [rf"$10^{{{int(np.log10(c))}}}$" for c in budgets])
    a.set_xlabel(r"算力预算 $C_{\mathrm{b}}$（FLOPs）")
ax[0].set_ylim(0, 1.3)
ax[1].set_ylim(0, 40)
ax[0].set_ylabel(r"最优质量 $Q^*$")
ax[1].set_ylabel("质量投入份额（%）")
panel(ax[0], "(a) 质量水平")
panel(ax[1], "(b) 质量投入份额")
fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
finish(fig, "图4_质量成本函数对比")

AUDIT["预算扫略"]["valid_Q"] = int(valid.sum())
AUDIT["预算扫略"]["missing_C"] = sw.loc[~valid, "C"].tolist()
AUDIT["上下文长度连续扫描"]["expected_grid_rows"] = 40
AUDIT["上下文长度连续扫描"]["missing_L_ctx"] = cont.loc[cont.N_B.isna(), "L_ctx"].tolist()
AUDIT["上下文长度与最优质量"]["valid_rows_by_budget"] = {str(c): len(d) for c,d in ql.groupby("C")}
(OUT / "论文绘图数据核验.json").write_text(json.dumps(AUDIT, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"已生成四组 PNG/PDF；扫描点 {len(sw)}，有效点 {valid.sum()}。")
print(f"首次正投入 {start:.6g}；首次饱和 {sat:.6g}；份额峰值 {peak.f_qual:.6f}。")
