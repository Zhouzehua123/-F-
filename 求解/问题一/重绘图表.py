"""问题一九幅图的展示版入口：只读既有结果，不执行求解程序。

运行：python 重绘图表.py --data-dir E:/华为杯f/real_attachments
输出：同名400 dpi PNG、矢量SVG、数据及画布边界核验记录。
开发辅助工具：OpenAI Codex；采用 academic-research-suite 图形工作流。
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from 展示风格公共 import configure
from 绘图数据 import HERE, DOMAIN_NAMES, load_plot_data, verify_inputs_unchanged
from 展示风格概览 import plot_quality, plot_mixture, plot_fit
from 展示风格诊断 import plot_prediction, plot_coefficients, plot_quality_comparison, plot_extrapolation
from 展示风格冲突 import plot_conflicts
from 展示风格流程 import plot_workflow

NAMES = ['图0_A1-A3全量质量评分', '图1_训练配比结构', '图2_线性模型R2',
         '图3_预测vs实测', '图4_混合系数热力图', '图5_领域质量评分',
         '图6_冲突与赋权诊断', '图7_外推与跨尺度', '图8_问题一求解流程']


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=HERE/'图片')
    parser.add_argument('--only', type=int, nargs='+', choices=range(9), help='只重绘指定图文件编号')
    args = parser.parse_args()
    configure()
    data = load_plot_data(args.data_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    functions = [plot_quality,plot_mixture,plot_fit,plot_prediction,plot_coefficients,
                 plot_quality_comparison,plot_conflicts,plot_extrapolation,plot_workflow]
    selected = args.only if args.only is not None else list(range(9))
    for i in selected:
        functions[i](data,args.output_dir)
        print(f'已绘制 {NAMES[i]}',flush=True)
    verify_inputs_unchanged(data)
    manifest = {'source_revision':'fb9e541f382bfd3ba48ff8f4c813beda689f1db7', 'style':'建模展示风',
                'skill':'academic-research-suite 3.22.0 / visualization_agent',
                'scripts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),HERE/'绘图数据.py',*sorted(HERE.glob('展示风格*.py'))]},'input_checks':data.checks,
                'inputs':{k:{field:value for field,value in v.items() if field!='_path'} for k,v in data.inputs.items()},
                'output_figures':[NAMES[i] for i in selected],
                'indicator_key':data.weights[['code','indicator','display_name']].to_dict(orient='records'),
                'domain_key':DOMAIN_NAMES,'prediction_points':int(data.observed.size),
                'coefficient_entries':int(data.centered.size),'whole_paper_compiled':False}
    (args.output_dir/'绘图数据核验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(data.checks)} 项数值核对通过；{len(data.inputs)} 个输入文件哈希未变。',flush=True)


if __name__ == '__main__':
    main()
