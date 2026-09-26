# 本程序的整理与核对使用 Codex 辅助；模型：GPT-5.6-Luna；机构：OpenAI；版本发布日期：2026-04-21。
# 本程序的整理、复现核对或绘图实现使用 OpenAI Codex（GPT-5.6-Luna）辅助。
"""读取问题一既有结果并核对绘图数值；不拟合、不优化、不写结果表。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DOMAIN_NAMES = {
    'arxiv': '学术预印本', 'freelaw': '法律判例', 'nih_exporter': '科研项目',
    'pubmed_central': '医学全文', 'wikipedia_en': '百科', 'dm_mathematics': '数学',
    'github': '代码', 'philpapers': '哲学论文', 'stackexchange': '专业问答',
    'enron_emails': '邮件', 'gutenberg_pg_19': '图书', 'pile_cc': '网页',
    'ubuntu_irc': '技术聊天', 'europarl': '议会记录', 'hackernews': '科技社区',
    'pubmed_abstracts': '医学摘要', 'uspto_backgrounds': '专利背景',
    'wikipedia': '百科', 'book': '图书', 'commoncrawl': 'Common Crawl', 'c4': 'C4',
}

INDICATOR_NAMES = {
    'fineweb_edu': '教育价值', 'fluency_en': '英语流畅度',
    'modernbert_cleanliness': '文本洁净度', 'modernbert_readability': '可读性',
    'modernbert_reasoning': '推理性', 'modernbert_professionalism': '专业性',
    'dsir_books': '书籍域权重', 'dsir_wiki': '维基域权重', 'dsir_math': '数学域权重',
    'qurater': '四维质量聚合', 'ad_en': '广告信号', 'rps_doc_word_count': '文档词数',
    'rps_doc_num_sentences': '文档句数', 'rps_doc_unigram_entropy': '一元词熵',
    'rps_doc_frac_unique_words': '不同词占比', 'rps_doc_frac_no_alph_words': '非字母信号',
    'rps_doc_frac_chars_top_2gram': '2-gram 重复度',
    'rps_doc_frac_chars_top_3gram': '3-gram 重复度',
    'rps_lines_uppercase_letter_fraction': '大写字母行占比',
    'rps_lines_ending_with_terminal_punctution_mark': '终止标点行占比',
    'rps_lines_numerical_chars_fraction': '含数字行占比',
    'rps_doc_mean_word_length': '平均词长',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def short_name(name):
    return name.replace('train_the_pile_', '').replace('metric/the_pile_', '').replace('_val_loss', '')


def load_plot_data(data_dir=None):
    candidates = [Path(data_dir)] if data_dir else [ROOT/'real_attachments', ROOT.parent/'real_attachments']
    data_root = next((p for p in candidates if p.is_dir()), None)
    if data_root is None:
        raise FileNotFoundError('请用 --data-dir 指定 real_attachments 根目录。')
    regmix = data_root/'A_data_value/regmix_tables'
    if not regmix.is_dir():
        raise FileNotFoundError(f'缺少配比附件目录：{regmix}')
    inputs, checks = {}, {}

    def read(path):
        path = path.resolve()
        df = pd.read_csv(path)
        try:
            label = path.relative_to(ROOT).as_posix()
        except ValueError:
            label = '附件/' + path.relative_to(data_root).as_posix()
        inputs[label] = {'sha256': sha256(path), 'rows': len(df), 'columns': len(df.columns),
                         '_path': path}
        return df

    def result(name, index=None):
        frame = read(HERE/'结果'/name)
        return frame.set_index(index, verify_integrity=True) if index else frame

    def attachment(name, prefix):
        frame = read(regmix/name).set_index('index', verify_integrity=True).sort_index()
        columns = [c for c in frame.columns if c.startswith(prefix)]
        frame = frame.loc[:, columns].rename(columns=short_name)
        if frame.empty or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f'附件 {name} 无有效数值或含缺失。')
        return frame

    def same(name, actual, expected, tol=1e-10):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=tol, rtol=tol):
            raise ValueError(f'绘图数值与保存结果不一致：{name}')
        checks[name] = {'passed': True, 'max_abs_diff': float(np.max(np.abs(actual-expected)))}

    coef = result('混合系数矩阵.csv', 'training_domain')
    centered = result('混合系数矩阵_中心化.csv', 'training_domain').loc[coef.index, coef.columns]
    intercept = result('截距.csv', 'loss_domain').loc[coef.columns, 'intercept']
    metrics = result('问题一_拟合汇总.csv', 'loss_domain').loc[coef.columns]
    recommended = result('推荐配比调整.csv', 'domain').loc[coef.index]
    scale = result('外推尺度因子.csv', 'loss_domain').loc[coef.columns]
    cross = result('跨尺度损失比_60m_vs_1m.csv', 'loss_domain').loc[coef.columns]
    quality = result('质量域评分_全量.csv').sort_values('quality_mean', ascending=False).reset_index(drop=True)
    weights = result('质量指标熵权.csv')
    conflicts = result('冲突指标对_top20.csv').sort_values('conflict_rate', ascending=False)
    proxy = result('Loss代理领域难度.csv').sort_values('quality_score_Q(loss代理)', ascending=False)
    link = result('质量与损失难度一致性.csv')
    nonlinear_domains = result('非线性代理对照_逐域.csv')
    nonlinear_predictions = result('非线性代理对照_预测明细.csv')

    same('中心化系数', coef.to_numpy()-coef.to_numpy().mean(axis=0), centered)
    same('熵权之和', weights.weight.sum(), 1.)
    assert len(weights) == 22 and len(quality) == 7 and coef.shape == (17, 13)
    assert quality.n.sum() == 272505
    assert set(weights.indicator) == set(INDICATOR_NAMES)
    weights['code'] = [f'M{i:02d}' for i in range(1, 23)]
    weights['display_name'] = weights.indicator.map(INDICATOR_NAMES)

    train_x = attachment('train_mixture_1m.csv', 'train_the_pile_').loc[:, coef.index]
    train_y = attachment('train_pile_loss_1m.csv', 'metric/the_pile_').loc[:, coef.columns]
    if not train_x.index.equals(train_y.index):
        raise ValueError('训练配方与损失的 index 不一致。')
    same('参考配比', train_x.mean(axis=0), recommended.reference_mixture)
    same('训练领域平均损失', train_y.mean(axis=0), metrics.mean_loss)

    def r_squared(y, pred):
        return 1-((y-pred)**2).sum(axis=0)/((y-y.mean(axis=0))**2).sum(axis=0)

    pred_train = train_x.to_numpy() @ coef.to_numpy() + intercept.to_numpy()
    same('训练 R2', r_squared(train_y.to_numpy(), pred_train), metrics.train_R2)
    test_x = attachment('test_mixture_1m.csv', 'train_the_pile_').loc[:, coef.index]
    test_y = attachment('test_pile_loss_1m.csv', 'metric/the_pile_').loc[:, coef.columns]
    if not test_x.index.equals(test_y.index):
        raise ValueError('1M 检验配方与损失的 index 不一致。')
    prediction = test_x.to_numpy() @ coef.to_numpy() + intercept.to_numpy()
    same('1M 检验 R2', r_squared(test_y.to_numpy(), prediction), metrics.test1m_R2)

    ratios = {}
    for size, column in [('10b','scale_factor_10B'), ('70b','scale_factor_70B')]:
        est_y = attachment(f'est_pile_loss_{size}.csv', 'metric/the_pile_').loc[:, coef.columns]
        if not est_y.index.isin(train_y.index).all():
            raise ValueError(f'{size} 外推损失无法按 index 对应训练记录。')
        values = est_y.to_numpy()/train_y.loc[est_y.index].to_numpy()
        same(f'{size} 外推均值', values.mean(axis=0), scale[column])
        ratios[size] = values

    test60 = attachment('test_pile_loss_60m.csv', 'metric/the_pile_').loc[:, coef.columns]
    if set(test60.index) != set(test_y.index):
        raise ValueError('60M、1M 检验记录的 index 集合不同。')
    cross_values = test60.loc[test_y.index].to_numpy()/test_y.to_numpy()
    same('60M/1M 均值', cross_values.mean(axis=0), cross.loss_ratio_60m_vs_1m)
    same('60M/1M 标准差', cross_values.std(axis=0), cross.loss_ratio_std)
    rho = float(link.Q_A1A3.corr(link['Q_loss代理'], method='spearman'))
    shared_path = HERE.parent/'问题一_关键量.json'
    shared = json.loads(shared_path.read_text(encoding='utf-8'))
    inputs[shared_path.relative_to(ROOT).as_posix()] = {'sha256': sha256(shared_path), '_path': shared_path}
    same('质量口径 Spearman', rho, shared['spearman_quality_vs_lossproxy'])
    return SimpleNamespace(quality=quality, weights=weights, conflicts=conflicts, metrics=metrics,
                           coef=coef, centered=centered, train_x=train_x, reference=train_x.mean(),
                           observed=test_y.to_numpy(), prediction=prediction, proxy=proxy, link=link,
                           rho=rho, ratios=ratios, scale=scale, cross=cross, inputs=inputs, checks=checks,
                           nonlinear_domains=nonlinear_domains, nonlinear_predictions=nonlinear_predictions)


def verify_inputs_unchanged(data):
    for label, info in data.inputs.items():
        if sha256(info['_path']) != info['sha256']:
            raise RuntimeError(f'绘图期间输入文件发生变化：{label}')
