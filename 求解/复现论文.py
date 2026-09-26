# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6-Luna；机构：OpenAI；版本发布日期：2026-07-09。
# 本程序由 OpenAI Codex（GPT-5.6-Luna）辅助实现，用于统一执行正式绘图与隔离复核。
"""在仓库根目录执行：
python 求解/复现论文.py --data-dir <附件根目录> --mode figures
python 求解/复现论文.py --data-dir <附件根目录> --mode verify-models

figures：读取已存结果、重绘正文全部图。verify-models：在临时副本运行完整四问，
按复现预期输出.json逐一比较重新生成的CSV/JSON；不会回写原结果。
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


def compare_json(a, b, ignored_keys=()):
    """数值容差与 CSV 一致；仅忽略清单中明示的顶层运行环境字段。"""
    x, y = [json.loads(p.read_text(encoding='utf-8-sig')) for p in (a, b)]
    ignored = {}
    for key in ignored_keys:
        if key not in x or key not in y:
            return dict(equal=False, reason=f'ignored metadata field missing: {key}')
        ignored[key] = dict(saved=x.pop(key), regenerated=y.pop(key))
    bad = []
    def walk(left, right, location='$'):
        if isinstance(left, dict) and isinstance(right, dict):
            # 输入路径分隔符随操作系统变化，但文件及其哈希必须相同。
            if location == '$.inputs':
                left = {k.replace('\\', '/'): v for k, v in left.items()}
                right = {k.replace('\\', '/'): v for k, v in right.items()}
            if left.keys() != right.keys():
                bad.append(location + ': keys differ')
            for key in left.keys() & right.keys():
                walk(left[key], right[key], location + '.' + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right): bad.append(location + ': length differs')
            for i, (u, v) in enumerate(zip(left, right)): walk(u, v, f'{location}[{i}]')
        elif isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            if not np.isclose(left, right, rtol=1e-6, atol=1e-8, equal_nan=True): bad.append(location)
        elif type(left) is not type(right) or left != right:
            bad.append(location)
    walk(x, y)
    return dict(equal=not bad, fields_differ=bad, metadata_not_compared=ignored)


def checked_output(root, relative):
    """只接受隔离副本求解目录内的相对文件；不允许越界或符号链接。"""
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or rel.parts[:1] != ('求解',):
        raise ValueError(f'非法预期输出路径: {relative}')
    target = root / rel
    if not target.resolve().is_relative_to((root / '求解').resolve()):
        raise ValueError(f'预期输出路径越界: {relative}')
    if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError(f'预期输出不得经过符号链接: {relative}')
    return target


def verify_models(root, data, report_dir, *, full_scope=True):
    """旧结果先从副本移除，要求每项重新生成并与基准比较；失败也保存报告。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    before = snapshots(root)
    report = dict(passed=False, expected_count=0, checked_count=0, outputs={}, commands=[])
    try:
        manifest = json.loads((root/'求解/复现预期输出.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError('预期输出清单不能为空')
        if full_scope and set(manifest) != {'问题一', '问题二', '问题三', '问题四'}:
            raise ValueError('正式复现清单必须覆盖全部四问')
        seen = set()
        for name, entries in manifest.items():
            if name not in ['问题一', '问题二', '问题三', '问题四'] or not entries:
                raise ValueError(f'无效的问题或空清单: {name}')
            for item in entries:
                rel = item['path']
                source = checked_output(root, rel)
                if rel in seen or item['kind'] not in ('csv', 'json'):
                    raise ValueError(f'重复路径或无效格式: {rel}')
                if not source.is_file(): raise FileNotFoundError(f'缺少比较基准: {rel}')
                seen.add(rel)
        if full_scope:
            # CLI 自身检查覆盖，不能依赖使用者先运行单元测试。
            required = {p.relative_to(root).as_posix() for name in manifest
                        for p in (root/'求解'/name/'结果').iterdir()
                        if p.is_file() and p.suffix in ('.csv', '.json')}
            required.update({'求解/17域质量映射.csv', '求解/问题一_关键量.json', '求解/广义标度律参数.csv'})
            if seen != required:
                raise ValueError(f'预期输出清单覆盖不完整: missing={sorted(required-seen)}, extra={sorted(seen-required)}')
        report['expected_count'] = len(seen)
        work = Path(tempfile.mkdtemp(prefix='model-check-', dir=report_dir))
        report['workspace'] = str(work)
        shutil.copytree(root/'求解', work/'求解', ignore=shutil.ignore_patterns('__pycache__', '图片', '*.ipynb'))
        if (root/'论文/fonts').exists(): shutil.copytree(root/'论文/fonts', work/'论文/fonts')
        for name, entries in manifest.items():
            # 删除只发生在隔离副本，且逐项验证其绝对路径仍在副本求解目录内。
            for item in entries:
                target = checked_output(work, item['path'])
                target.unlink(missing_ok=True)
            script = Path('求解')/name/(name+'.py')
            command = dict(script=script.as_posix(), status='running')
            report['commands'].append(command)
            try:
                run(work, script, [], data)
                command['status'] = 'completed'
            except Exception as exc:
                command.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            for item in entries:
                rel = item['path']; target = checked_output(work, rel)
                if not target.is_file():
                    result = dict(equal=False, regenerated=False, reason='预期输出未生成或缺失')
                else:
                    result = (compare_csv(root/rel, target) if item['kind']=='csv'
                              else compare_json(root/rel, target, item.get('ignore_top_level', [])))
                    result.update(regenerated=True, saved_sha256=sha(root/rel), new_sha256=sha(target))
                    report['checked_count'] += 1
                report['outputs'][rel] = result
            failed = [rel for rel, result in report['outputs'].items() if not result['equal']]
            if failed: raise RuntimeError('复现输出缺失或结果不一致: ' + ', '.join(failed))
        if report['checked_count'] != report['expected_count'] or report['checked_count'] == 0:
            raise RuntimeError('未完整核对预期输出')
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['source_results_unchanged'] = before == snapshots(root)
        if not report['source_results_unchanged']: report['passed'] = False
        (report_dir/'模型复现比较.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not report['source_results_unchanged']: raise RuntimeError('源结果发生变化')
    print(f'已完整核对 {report["checked_count"]} 项重新生成的 CSV/JSON；原结果未改动。')
    return report


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
    verify_models(ROOT, data, args.report_dir)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    main()
