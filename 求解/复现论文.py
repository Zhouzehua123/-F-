# 本程序由 OpenAI Codex（GPT-6）辅助实现，用于统一执行正式绘图与隔离复核。
"""在仓库根目录执行：
python 求解/复现论文.py --data-dir <附件根目录> --mode figures
python 求解/复现论文.py --data-dir <附件根目录> --mode verify-models

figures：读取已存结果、重绘正文全部图。verify-models：在临时副本运行完整四问，
逐一比较重新输出的CSV；不会将重新求解的结果回写到本仓库。
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


def compare_csv(a,b):
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
            if not np.allclose(x[c],y[c],rtol=1e-6,atol=1e-8,equal_nan=True): bad.append(c)
        elif not x[c].fillna('<NA>').astype(str).equals(y[c].fillna('<NA>').astype(str)):
            bad.append(c)
    return dict(equal=not bad,columns_differ=bad)


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
    report=dict(workspace=str(work),csv={},commands=[])
    try:
        for name in ['问题一','问题二','问题三','问题四']:
            times={p:p.stat().st_mtime_ns for p in (work/'求解').rglob('*.csv')}
            script=Path('求解')/name/(name+'.py')
            run(work,script,[],data);report['commands'].append(str(script))
            for p in (work/'求解').rglob('*.csv'):
                if times.get(p)==p.stat().st_mtime_ns: continue
                rel=p.relative_to(work);source=ROOT/rel
                if source.exists(): report['csv'][rel.as_posix()]=compare_csv(source,p)
                else: report['csv'][rel.as_posix()]=dict(equal=True,new_output=True)
    finally:
        report['source_results_unchanged']=(before==snapshots(ROOT))
        (args.report_dir/'模型复现比较.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    bad=[p for p,r in report['csv'].items() if not r['equal']]
    print(f'已核对 {len(report["csv"])} 个CSV；不一致 {len(bad)}；原结果未改动。')
    if bad: raise RuntimeError('隔离复现发现差异：'+', '.join(bad))
    assert report['source_results_unchanged']


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    main()
