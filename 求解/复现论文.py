# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6-Luna；机构：OpenAI；版本发布日期：2026-07-09。
# 本程序由 OpenAI Codex（GPT-5.6-Luna）辅助实现，用于统一执行正式绘图与隔离复核。
"""在仓库根目录执行：
python 求解/复现论文.py --data-dir <附件根目录> --mode figures
python 求解/复现论文.py --data-dir <附件根目录> --mode verify-models

figures：读取已存结果、重绘正文全部图。verify-models：在临时副本运行完整四问，
按预期清单逐一比较重新生成的 CSV 及跨问 JSON；不会回写原结果。
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 每问正式数值输出的固定清单；新增正式输出时须同步登记。
EXPECTED_OUTPUTS = {
    '问题一': [
        '求解/问题一/结果/线性有界配比对照.csv',
        '求解/问题一/结果/核岭推荐_训练凸组合.csv',
        '求解/问题一/结果/核岭推荐_多初值.csv',
        '求解/问题一/结果/核岭推荐_目标对照.csv',
        '求解/问题一/结果/17域质量映射.csv',
        '求解/问题一/结果/Loss代理领域难度.csv',
        '求解/问题一/结果/冲突指标对_top20.csv',
        '求解/问题一/结果/冲突样本裁决汇总.csv',
        '求解/问题一/结果/冲突阈值灵敏度.csv',
        '求解/问题一/结果/原文抽查.csv',
        '求解/问题一/结果/外推尺度因子.csv',
        '求解/问题一/结果/岭正则敏感性.csv',
        '求解/问题一/结果/广告方向敏感性.csv',
        '求解/问题一/结果/截距.csv',
        '求解/问题一/结果/抽样集与扩展集域级对照.csv',
        '求解/问题一/结果/推荐配比调整.csv',
        '求解/问题一/结果/最优配比预测.csv',
        '求解/问题一/结果/样本级质量评分.csv',
        '求解/问题一/结果/混合系数矩阵.csv',
        '求解/问题一/结果/混合系数矩阵_中心化.csv',
        '求解/问题一/结果/训练域边际效应.csv',
        '求解/问题一/结果/质量与损失难度一致性.csv',
        '求解/问题一/结果/质量信号缺失审计.csv',
        '求解/问题一/结果/质量冲突与一致性.csv',
        '求解/问题一/结果/质量域评分_全量.csv',
        '求解/问题一/结果/质量域评分_抽样与扩展.csv',
        '求解/问题一/结果/质量指标熵权.csv',
        '求解/问题一/结果/质量项增量检验.csv',
        '求解/问题一/结果/赋权方案对比.csv',
        '求解/问题一/结果/跨尺度损失比_60m_vs_1m.csv',
        '求解/问题一/结果/问题一_拟合汇总.csv',
        '求解/问题一/结果/非线性代理对照_五折.csv',
        '求解/问题一/结果/非线性代理对照_参数搜索.csv',
        '求解/问题一/结果/非线性代理对照_折分配.csv',
        '求解/问题一/结果/非线性代理对照_汇总.csv',
        '求解/问题一/结果/非线性代理对照_逐域.csv',
        '求解/问题一/结果/非线性代理对照_预测明细.csv',
        '求解/问题一/结果/领域质量评分.csv',
        '求解/17域质量映射.csv',
        '求解/问题一_关键量.json',
    ],
    '问题二': [
        '求解/问题二/结果/B3轨迹验证.csv',
        '求解/问题二/结果/广义标度律分层验证.csv',
        '求解/问题二/结果/广义标度律参数.csv',
        '求解/问题二/结果/广义标度律形式对比.csv',
        '求解/问题二/结果/弹性与替代关系.csv',
        '求解/问题二/结果/替代率随质量变化.csv',
        '求解/问题二/结果/百亿参数以上外推.csv',
        '求解/问题二/结果/经典标度律参数.csv',
        '求解/问题二/结果/质量与损失下限.csv',
        '求解/问题二/结果/质量参数替代关系.csv',
        '求解/问题二/结果/质量字段方向审计.csv',
        '求解/问题二/结果/配比项校准.csv',
        '求解/问题二/结果/问题二_拟合汇总.csv',
        '求解/问题二/结果/领域替代互补_top.csv',
        '求解/问题二/结果/领域替代互补矩阵.csv',
        '求解/广义标度律参数.csv',
    ],
    '问题三': [
        '求解/问题三/结果/三档预算最优配置.csv',
        '求解/问题三/结果/上下文长度与最优质量.csv',
        '求解/问题三/结果/上下文长度敏感性.csv',
        '求解/问题三/结果/上下文长度连续扫描.csv',
        '求解/问题三/结果/成本函数对比.csv',
        '求解/问题三/结果/结构性转移识别.csv',
        '求解/问题三/结果/质量基线灵敏度.csv',
        '求解/问题三/结果/配比项算力等价.csv',
        '求解/问题三/结果/预算扫略.csv',
    ],
    '问题四': [
        '求解/问题四/结果/C3口径异常行.csv',
        '求解/问题四/结果/C4宏观摘要.csv',
        '求解/问题四/结果/C4宏观趋势.csv',
        '求解/问题四/结果/C8与C1一致性校验.csv',
        '求解/问题四/结果/C8子任务类别汇总.csv',
        '求解/问题四/结果/C8逐任务聚合.csv',
        '求解/问题四/结果/C8逐任务解析结果.csv',
        '求解/问题四/结果/logistic参数.csv',
        '求解/问题四/结果/Loss_Benchmark映射.csv',
        '求解/问题四/结果/Loss_Benchmark映射明细.csv',
        '求解/问题四/结果/前沿序列.csv',
        '求解/问题四/结果/前沿预测.csv',
        '求解/问题四/结果/前沿预测_结构外推情景.csv',
        '求解/问题四/结果/年份效应.csv',
        '求解/问题四/结果/筛选漏斗.csv',
        '求解/问题四/结果/规模时间分解.csv',
        '求解/问题四/结果/规模时间分解_长周期.csv',
        '求解/问题四/结果/逐任务聚合.csv',
        '求解/问题四/结果/面板回归数据.csv',
    ],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshots(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in (root/'求解').rglob('*')
            if p.is_file() and p.suffix in ('.csv','.json') and '图片' not in p.parts
            and '__pycache__' not in p.parts}


def run(root, script, args, data):
    env = dict(os.environ, MODELING_DATA_DIR=str(data), PYTHONIOENCODING='utf-8')
    print(f'执行：{script}', flush=True)
    subprocess.run([sys.executable, str(root/script), *map(str,args)], cwd=root, env=env, check=True)


def figure_map(root):
    # 从真实主入口递归读取；旧拆分稿和注释不参与清单。
    visited = set(); rows = []
    def walk(p):
        p=p.resolve()
        if p in visited: return
        visited.add(p)
        text='\n'.join(re.split(r'(?<!\\)%',line)[0] for line in p.read_text(encoding='utf-8').splitlines())
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',text):
            target=root/'论文'/name
            if not target.suffix: target=target.with_suffix('.tex')
            if target.exists(): walk(target)
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
            target=(root/'论文'/name).resolve()
            if not target.is_file(): raise FileNotFoundError(target)
            rel=target.relative_to(root).as_posix()
            if '/问题一/' in rel: source='求解/问题一/重绘图表.py'
            elif '/问题二/' in rel: source='求解/问题二/重绘图表.py'
            elif '/问题三/' in rel: source='求解/问题三/论文绘图.py'
            elif '/问题四/' in rel: source='求解/问题四/重绘图表.py'
            elif 'workflow' in rel: source='求解/绘制问题三四流程图.py'
            else: source='模板静态资源'
            rows.append(dict(tex=p.relative_to(root).as_posix(), figure=rel, source=source, sha256=sha(target)))
    walk(root/'论文/论文.tex')
    rows.append(dict(tex='论文/2.总体分析.tex',figure='论文/figures/overall-architecture.tex',
                     source='原生TikZ，随论文编译',sha256=sha(root/'论文/figures/overall-architecture.tex')))
    return rows


def figures(root, data):
    before=snapshots(root)
    for script,args in [
        ('求解/问题一/重绘图表.py',['--data-dir',data]),
        ('求解/问题二/重绘图表.py',['--data-dir',data]),
        ('求解/问题三/论文绘图.py',[]),
        ('求解/问题四/重绘图表.py',['--data',data]),
        ('求解/绘制问题三四流程图.py',[]),
    ]: run(root,Path(script),args,data)
    if before != snapshots(root): raise RuntimeError('绘图修改了结果或跨问接口')
    return figure_map(root)


def compare_csv(a,b,atol=1e-8):
    x,y=pd.read_csv(a),pd.read_csv(b)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return dict(equal=False,reason='shape/columns differ',saved_shape=list(x.shape),new_shape=list(y.shape))
    # 数值相同的零份额可能有不同稳定排序；按明确唯一的业务键对齐。
    for key in ['index','domain','training_domain','loss_domain','param']:
        if key in x and not x[key].duplicated().any() and not y[key].duplicated().any():
            x=x.sort_values(key).reset_index(drop=True)
            y=y.sort_values(key).reset_index(drop=True)
            break
    bad=[]
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            if not np.allclose(x[c],y[c],rtol=1e-6,atol=atol,equal_nan=True): bad.append(c)
        elif not x[c].fillna('<NA>').astype(str).equals(y[c].fillna('<NA>').astype(str)):
            bad.append(c)
    return dict(equal=not bad,columns_differ=bad)


def compare_json(a, b):
    """递归比较跨问接口，数值容差与 CSV 一致。"""
    def equal(x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y))
        if type(x) in (int, float) and type(y) in (int, float):
            return bool(np.isclose(x, y, rtol=1e-6, atol=1e-8, equal_nan=True))
        return type(x) is type(y) and x == y
    x = json.loads(a.read_text(encoding='utf-8-sig'))
    y = json.loads(b.read_text(encoding='utf-8-sig'))
    return dict(equal=equal(x, y))


def check_kernel_certificate(root, data):
    """凸组合表示不唯一：核对其数学含义，不要求权重逐项重现。"""
    folder=root/'求解/问题一/结果'
    train=pd.read_csv(data/'A_data_value/regmix_tables/train_mixture_1m.csv').set_index('index').sort_index()
    columns=[c for c in train if c.startswith('train_the_pile_')]
    domains=[c.removeprefix('train_the_pile_') for c in columns]
    raw=train[columns].to_numpy(float);closed=raw/raw.sum(axis=1,keepdims=True)
    ref=raw.mean(axis=0);ref/=ref.sum()
    certificate=pd.read_csv(folder/'核岭推荐_训练凸组合.csv').set_index('index')
    if certificate.index.has_duplicates or set(certificate.index)!=set(train.index):
        return dict(equal=False,reason='训练凸组合 index 缺失或重复')
    z=certificate.loc[train.index,'convex_weight'].to_numpy(float)
    recipe=pd.read_csv(folder/'推荐配比调整.csv').set_index('domain').loc[domains]
    p=recipe.recommended_mixture.to_numpy(float)
    loss_columns=pd.read_csv(data/'A_data_value/regmix_tables/train_pile_loss_1m.csv',nrows=0).columns
    measured=[c.removeprefix('metric/the_pile_').removesuffix('_val_loss') for c in loss_columns if c.endswith('_val_loss')]
    fixed=[i for i,d in enumerate(domains) if d not in measured]
    residual=max(abs(z.sum()-1),max(0.,-z.min()),float(np.max(abs(z@closed-p))),
                 abs(p.sum()-1),float(np.max(abs(p[fixed]-ref[fixed]))),
                 float(np.maximum(p-2.5*ref,0).max()),float(np.max(abs(recipe.reference_mixture-ref))))
    return dict(equal=bool(np.isfinite(residual) and np.isfinite(z).all() and np.isfinite(p).all() and residual<1e-7),
                validation='nonnegative convex reconstruction and recipe constraints',max_residual=float(residual))


def check_kernel_runs(root):
    folder=root/'求解/问题一/结果'
    runs=pd.read_csv(folder/'核岭推荐_多初值.csv')
    target=pd.read_csv(folder/'核岭推荐_目标对照.csv').set_index('mixture').loc['问题一有界推荐','predicted_weighted_loss']
    good=runs[runs.success.eq(True)]
    ok=(len(runs)==4 and len(good)>0 and np.isfinite(runs.objective).all()
        and (runs.constraint_residual<1e-7).all() and (good.stationarity_gap<1e-6).all()
        and abs(good.objective.min()-target)<1e-7 and runs.objective.max()-runs.objective.min()<1e-6)
    return dict(equal=bool(ok),validation='feasibility, stationarity and objective agreement',successful_starts=len(good))


def compare_output(source, work, rel, data):
    if rel.name=='核岭推荐_训练凸组合.csv':
        return check_kernel_certificate(work,data)
    if rel.name=='核岭推荐_多初值.csv':
        return check_kernel_runs(work)
    if rel.name=='推荐配比调整.csv':
        # 配方为数值驻点；容差与求解停止精度一致，另由凸组合证书严格检查可行性。
        return compare_csv(source/rel,work/rel,atol=1e-6)
    return (compare_csv if rel.suffix=='.csv' else compare_json)(source/rel,work/rel)


def verify_stage(source, work, name, data, report, runner=run):
    """先删除隔离副本内的预期输出，禁止把旧文件当作新结果。"""
    required = [Path(p) for p in EXPECTED_OUTPUTS[name]]
    if not required:
        raise RuntimeError(f'{name}的预期输出清单为空')
    for rel in required:
        if not (source / rel).is_file():
            raise FileNotFoundError(f'缺少比较基准：{rel}')
        target = (work / rel).resolve()
        if not target.is_relative_to(work.resolve()) or source.resolve() == work.resolve():
            raise ValueError('输出清理仅允许在隔离副本内进行')
        target.unlink(missing_ok=True)
    before = snapshots(work)
    script = Path('求解') / name / (name + '.py')
    runner(work, script, [], data)
    report['commands'].append(str(script))
    failed = []
    for rel in required:
        p = work / rel
        if not p.is_file():
            result = dict(equal=False, status='missing', reason='未重新生成预期输出')
        else:
            try:
                result = compare_output(source,work,rel,data)
                result['status'] = 'matched' if result['equal'] else 'different'
            except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
                result = dict(equal=False, status='unreadable', reason=str(exc))
        report['outputs'][rel.as_posix()] = result
        if not result['equal']:
            failed.append(rel.as_posix())
    expected_keys = {p.as_posix() for p in required}
    # 新出现但没有基准的结果单独列示，不计入核对成功数量。
    for rel, value in snapshots(work).items():
        if rel not in expected_keys and before.get(rel) != value:
            report['uncompared'][rel] = dict(status='not_compared', stage=name)
    if failed:
        raise RuntimeError('预期输出缺失或不一致：' + ', '.join(failed))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--mode',choices=['figures','verify-models'],default='figures')
    ap.add_argument('--report-dir',type=Path,default=ROOT/'复现记录')
    args=ap.parse_args();data=args.data_dir.resolve()
    if not (data/'A_data_value').is_dir(): raise FileNotFoundError(data)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=='figures':
        rows=figures(ROOT,data)
        (args.report_dir/'图表来源.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{len(rows)} 项图形资源已对应到源码；结果文件未改动。')
        return
    before=snapshots(ROOT)
    # 隔离目录保留供差异调查；求解器永远不写原结果。
    work=Path(tempfile.mkdtemp(prefix='model-check-',dir=args.report_dir))
    shutil.copytree(ROOT/'求解',work/'求解',ignore=shutil.ignore_patterns('__pycache__','图片','*.ipynb'))
    shutil.copytree(ROOT/'论文/fonts',work/'论文/fonts')
    report=dict(workspace=str(work),outputs={},uncompared={},commands=[],complete=False)
    try:
        for name in ['问题一','问题二','问题三','问题四']:
            verify_stage(ROOT,work,name,data,report)
        report['complete']=True
    finally:
        report['source_results_unchanged']=(before==snapshots(ROOT))
        (args.report_dir/'模型复现比较.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    count=sum(r['status']=='matched' for r in report['outputs'].values())
    if count != sum(map(len, EXPECTED_OUTPUTS.values())) or not report['complete']:
        raise RuntimeError('预期输出覆盖不完整')
    if not report['source_results_unchanged']:
        raise RuntimeError('原结果发生变化')
    print(f'已重新生成并核对 {count} 项预期输出；另有 {len(report["uncompared"])} 项未比较；原结果未改动。')


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    main()
