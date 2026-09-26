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


def check_kernel_certificate(root, data):
    """凸组合表示不唯一：核对其数学含义，不要求权重逐项重现。"""
    folder=root/'求解/问题一/结果'
    train=pd.read_csv(data/'A_data_value/regmix_tables/train_mixture_1m.csv').set_index('index').sort_index()
    columns=[c for c in train if c.startswith('train_the_pile_')]
    domains=[c.removeprefix('train_the_pile_') for c in columns]
    raw=train[columns].to_numpy(float)
    if (not columns or not len(raw) or train.index.has_duplicates or train.index.isna().any()
            or not np.isfinite(raw).all() or (raw<0).any() or (raw.sum(axis=1)<=0).any()):
        return dict(equal=False,reason='训练配方存在重复索引、非有限数或无效份额')
    closed=raw/raw.sum(axis=1,keepdims=True)
    ref=raw.mean(axis=0);ref/=ref.sum()
    certificate=pd.read_csv(folder/'核岭推荐_训练凸组合.csv').set_index('index')
    if certificate.index.has_duplicates or set(certificate.index)!=set(train.index):
        return dict(equal=False,reason='训练凸组合 index 缺失或重复')
    z=certificate.loc[train.index,'convex_weight'].to_numpy(float)
    recipe=pd.read_csv(folder/'推荐配比调整.csv').set_index('domain').loc[domains]
    p=recipe.recommended_mixture.to_numpy(float)
    # 先检查所有输入，再做 max：Python max 可能忽略后续位置的 NaN。
    reference=recipe.reference_mixture.to_numpy(float)
    if recipe.index.has_duplicates or not all(np.isfinite(x).all() for x in [closed,ref,z,p,reference]):
        return dict(equal=False,reason='凸组合或配方字段包含非有限数、重复领域')
    loss_columns=pd.read_csv(data/'A_data_value/regmix_tables/train_pile_loss_1m.csv',nrows=0).columns
    measured=[c.removeprefix('metric/the_pile_').removesuffix('_val_loss') for c in loss_columns if c.endswith('_val_loss')]
    fixed=[i for i,d in enumerate(domains) if d not in measured]
    residual=max(abs(z.sum()-1),max(0.,-z.min()),float(np.max(abs(z@closed-p))),
                 abs(p.sum()-1),float(np.max(abs(p[fixed]-ref[fixed]))),
                 float(np.maximum(p-2.5*ref,0).max()),float(np.max(abs(reference-ref))))
    return dict(equal=bool(np.isfinite(residual) and np.isfinite(z).all() and np.isfinite(p).all() and residual<1e-7),
                validation='nonnegative convex reconstruction and recipe constraints',max_residual=float(residual))


def check_kernel_runs(root):
    """检查四个初值的记录；保护梯度间隙不构成边界光滑驻点证明。"""
    folder=root/'求解/问题一/结果'
    runs=pd.read_csv(folder/'核岭推荐_多初值.csv')
    required={'start','success','objective','constraint_residual','iterations','stationarity_gap'}
    if not required.issubset(runs.columns) or len(runs)!=4:
        return dict(equal=False,reason='多初值记录必须包含完整字段及四行数据')
    if not pd.api.types.is_bool_dtype(runs.success) or runs.success.isna().any():
        return dict(equal=False,reason='success 必须是无缺失的布尔值')
    numeric=['start','objective','constraint_residual','iterations','stationarity_gap']
    if any(not pd.api.types.is_numeric_dtype(runs[c]) or pd.api.types.is_bool_dtype(runs[c]) for c in numeric):
        return dict(equal=False,reason='多初值数值字段类型错误')
    if not np.isfinite(runs[numeric].to_numpy(float)).all():
        return dict(equal=False,reason='多初值数值字段包含非有限数')
    if set(runs.start)!={0,1,2,3} or runs.start.duplicated().any():
        return dict(equal=False,reason='start 必须为互异的 0、1、2、3')
    if ((runs.iterations<1)|(runs.iterations>150)|(runs.iterations!=np.floor(runs.iterations))).any():
        return dict(equal=False,reason='外层迭代次数必须是 1 至 150 的整数')
    targets=pd.read_csv(folder/'核岭推荐_目标对照.csv')
    if not {'mixture','predicted_weighted_loss'}.issubset(targets.columns):
        return dict(equal=False,reason='核岭目标对照缺少字段')
    target_rows=targets.loc[targets.mixture.eq('问题一有界推荐'),'predicted_weighted_loss']
    if len(target_rows)!=1 or not pd.api.types.is_numeric_dtype(target_rows):
        return dict(equal=False,reason='核岭推荐目标必须为唯一数值')
    target=float(target_rows.iloc[0])
    if not np.isfinite(target):
        return dict(equal=False,reason='核岭推荐目标不是有限数')
    good=runs[runs.success.eq(True)]
    # 精确线性子问题的间隙非负。仅容许 1e-12 的负舍入记录，
    # 比原 1e-6 停止阈值小六个数量级；这不是梯度有效性或驻点证明。
    negative_gap_roundoff=1e-12
    ok=(len(good)>0 and (runs.constraint_residual>=0).all()
        and (runs.constraint_residual<1e-7).all()
        and (runs.stationarity_gap>=-negative_gap_roundoff).all()
        and (good.stationarity_gap<1e-6).all()
        and abs(good.objective.min()-target)<1e-7 and runs.objective.max()-runs.objective.min()<1e-6)
    return dict(equal=bool(ok),validation='feasibility, recorded protected-gradient gap threshold and objective agreement',
                successful_starts=len(good),negative_gap_roundoff=negative_gap_roundoff,
                limitation='recorded protected-gradient diagnostic only; no smooth-stationarity or global-optimality certificate')


def compare_expected_output(source, work, item, data):
    """一般输出按数值比较；核岭非唯一证书沿用主线的可行性核验。"""
    rel = item['path']
    policy = item.get('comparison', 'numeric')
    checks = {'kernel_certificate': check_kernel_certificate, 'kernel_runs': check_kernel_runs}
    if policy in checks:
        saved = checks[policy](source, data) if policy == 'kernel_certificate' else checks[policy](source)
        regenerated = checks[policy](work, data) if policy == 'kernel_certificate' else checks[policy](work)
        return dict(equal=bool(saved['equal'] and regenerated['equal']), comparison=policy,
                    saved_validation=saved, regenerated_validation=regenerated)
    if item['kind'] == 'csv':
        atol = item.get('atol', 1e-8)
        result = compare_csv(source/rel, work/rel, atol=atol)
        result.update(comparison='numeric', rtol=1e-6, atol=atol)
        return result
    return compare_json(source/rel, work/rel, item.get('ignore_top_level', []))


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
                policy = item.get('comparison', 'numeric')
                special = {
                    'kernel_certificate': '求解/问题一/结果/核岭推荐_训练凸组合.csv',
                    'kernel_runs': '求解/问题一/结果/核岭推荐_多初值.csv',
                }
                if policy != 'numeric' and special.get(policy) != rel:
                    raise ValueError(f'无效的输出比较规则: {rel}: {policy}')
                if 'atol' in item and (rel != '求解/问题一/结果/推荐配比调整.csv' or item['atol'] != 1e-6):
                    raise ValueError(f'未约定的数值容差: {rel}')
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
                    result = compare_expected_output(root, work, item, data)
                    result.update(regenerated=True, saved_sha256=sha(root/rel), new_sha256=sha(target))
                    report['checked_count'] += 1
                report['outputs'][rel] = result
        # 数值差异先记录，让可运行的后续问也得到检查；最终仍整体失败。
        # 进程异常（例如上游缺文件）由上面的异常分支及时报告。
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
