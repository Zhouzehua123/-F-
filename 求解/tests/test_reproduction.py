"""复现核验的回归测试；使用微型输出，不运行论文求解器。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("reproduction", Path(__file__).resolve().parents[1] / "复现论文.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReproductionTests(unittest.TestCase):
    def test_repository_manifest_covers_saved_model_outputs(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((root / '求解/复现预期输出.json').read_text(encoding='utf-8'))
        self.assertEqual(set(manifest), {'问题一', '问题二', '问题三', '问题四'})
        listed = [row['path'] for rows in manifest.values() for row in rows]
        saved = {p.relative_to(root).as_posix() for name in manifest
                 for p in (root / '求解' / name / '结果').iterdir() if p.suffix in ('.csv', '.json')}
        saved.update({'求解/17域质量映射.csv', '求解/问题一_关键量.json', '求解/广义标度律参数.csv'})
        self.assertEqual(set(listed), saved)
        self.assertEqual(len(listed), len(set(listed)))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.report_dir = self.root / 'report'
        self.csv = '求解/问题一/结果/评分.csv'
        self.meta = '求解/问题一/结果/统计.json'
        (self.root / self.csv).parent.mkdir(parents=True)
        (self.root / self.csv).write_text('domain,value\na,1\nb,2\n', encoding='utf-8')
        (self.root / self.meta).write_text('{"score": 0.5, "elapsed": 10}', encoding='utf-8')
        self.manifest = {'问题一': [{'path': self.csv, 'kind': 'csv'}, {'path': self.meta, 'kind': 'json', 'ignore_top_level': ['elapsed']}]}
        self.save_manifest()

    def save_manifest(self):
        (self.root / '求解/复现预期输出.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def verify(self, runner):
        with patch.object(MODULE, 'run', side_effect=runner):
            return MODULE.verify_models(self.root, self.root / 'data', self.report_dir, full_scope=False)

    def write_good(self, work, *args):
        self.assertFalse((work / self.csv).exists())
        self.assertFalse((work / self.meta).exists())
        (work / self.csv).write_text('domain,value\nb,2\na,1.0000001\n', encoding='utf-8')
        (work / self.meta).write_text('{"score": 0.50000001, "elapsed": 20}', encoding='utf-8')

    def test_fresh_outputs_and_explicit_metadata_policy(self):
        report = self.verify(self.write_good)
        self.assertTrue(report['passed'])
        self.assertEqual(report['checked_count'], 2)
        self.assertTrue(report['source_results_unchanged'])
        self.assertEqual(report['outputs'][self.meta]['metadata_not_compared']['elapsed']['regenerated'], 20)

    def test_missing_second_output_fails(self):
        def runner(work, *args):
            (work / self.csv).write_text((self.root / self.csv).read_text(encoding='utf-8'), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, '缺失'):
            self.verify(runner)

    def test_csv_and_json_differences_fail(self):
        for path, text in [(self.csv, 'domain,value\na,9\nb,2\n'), (self.meta, '{"score": 0.9, "elapsed": 20}')]:
            with self.subTest(path=path):
                def runner(work, *args):
                    self.write_good(work)
                    (work / path).write_text(text, encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, '不一致'):
                    self.verify(runner)

    def test_solver_exception_is_reported(self):
        def runner(*args): raise RuntimeError('solver failed')
        with self.assertRaisesRegex(RuntimeError, 'solver failed'): self.verify(runner)
        report = json.loads((self.report_dir / '模型复现比较.json').read_text(encoding='utf-8'))
        self.assertFalse(report['passed'])
        self.assertEqual(report['commands'][0]['status'], 'failed')
        self.assertTrue(report['source_results_unchanged'])

    def test_difference_does_not_skip_independent_later_question(self):
        later = '求解/问题二/结果/later.csv'
        (self.root / later).parent.mkdir(parents=True)
        (self.root / later).write_text('value\n1\n', encoding='utf-8')
        self.manifest['问题二'] = [{'path': later, 'kind': 'csv'}]
        self.save_manifest()
        called = []
        def runner(work, script, *args):
            called.append(script.parent.name)
            if script.parent.name == '问题一':
                self.write_good(work)
                (work / self.csv).write_text('domain,value\na,9\nb,2\n', encoding='utf-8')
            else:
                self.assertFalse((work / later).exists())
                (work / later).write_text('value\n1\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, '不一致'): self.verify(runner)
        self.assertEqual(called, ['问题一', '问题二'])
        report = json.loads((self.report_dir / '模型复现比较.json').read_text(encoding='utf-8'))
        self.assertFalse(report['passed'])
        self.assertTrue(report['outputs'][later]['equal'])

    def test_empty_manifest_and_unsafe_path_are_rejected(self):
        for manifest in [{}, {'问题一': []}, {'问题一': [{'path': '求解/../../outside.csv', 'kind': 'csv'}]}]:
            with self.subTest(manifest=manifest):
                self.manifest = manifest; self.save_manifest()
                with self.assertRaises(ValueError): self.verify(lambda *args: None)
                report = json.loads((self.report_dir / '模型复现比较.json').read_text(encoding='utf-8'))
                self.assertFalse(report['passed'])

    def test_no_outputs_must_fail_even_when_old_csv_is_present(self):
        with self.assertRaisesRegex(RuntimeError, '缺失|未生成'):
            self.verify(lambda *args: None)
        report = json.loads((self.report_dir / '模型复现比较.json').read_text(encoding='utf-8'))
        self.assertFalse(report['passed'])
        self.assertTrue(report['source_results_unchanged'])

    def test_production_scope_rejects_omitted_question_or_file(self):
        with self.assertRaisesRegex(ValueError, '全部四问'):
            MODULE.verify_models(self.root, self.root / 'data', self.report_dir)
        for name in ['问题二', '问题三', '问题四']:
            rel = f'求解/{name}/结果/out.csv'
            (self.root / rel).parent.mkdir(parents=True)
            (self.root / rel).write_text('value\n1\n', encoding='utf-8')
            self.manifest[name] = [{'path': rel, 'kind': 'csv'}]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, '覆盖不完整'):
            MODULE.verify_models(self.root, self.root / 'data', self.report_dir)

    def test_unknown_policy_cannot_bypass_numeric_comparison(self):
        self.manifest['问题一'][0]['comparison'] = 'ignore_values'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, '比较规则'):
            self.verify(self.write_good)

    def test_kernel_certificate_requires_valid_saved_and_regenerated_points(self):
        certificate = '求解/问题一/结果/核岭推荐_训练凸组合.csv'
        (self.root / certificate).write_text('index,convex_weight\n0,1\n', encoding='utf-8')
        self.manifest['问题一'].append({'path': certificate, 'kind': 'csv', 'comparison': 'kernel_certificate'})
        self.save_manifest()
        def runner(work, *args):
            self.write_good(work)
            (work / certificate).write_text('index,convex_weight\n0,1\n', encoding='utf-8')
        for saved_ok, new_ok in [(False, True), (True, False)]:
            with self.subTest(saved=saved_ok, regenerated=new_ok):
                with patch.object(MODULE, 'check_kernel_certificate', create=True,
                                  side_effect=[{'equal': saved_ok}, {'equal': new_ok}]):
                    with self.assertRaisesRegex(RuntimeError, '不一致'):
                        self.verify(runner)

    def test_recipe_tolerance_is_explicit_and_scoped(self):
        recipe = '求解/问题一/结果/推荐配比调整.csv'
        (self.root / recipe).write_text('domain,recommended_mixture\na,0.2\nb,0.8\n', encoding='utf-8')
        self.manifest['问题一'].append({'path': recipe, 'kind': 'csv', 'atol': 1e-6})
        self.save_manifest()
        def runner(work, *args):
            self.write_good(work)
            (work / recipe).write_text('domain,recommended_mixture\na,0.2000005\nb,0.7999995\n', encoding='utf-8')
        report = self.verify(runner)
        self.assertEqual(report['outputs'][recipe]['atol'], 1e-6)
        self.manifest['问题一'][0]['atol'] = 1e-6
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, '容差'):
            self.verify(runner)

    def write_kernel_runs(self):
        folder = self.root / '求解/问题一/结果'
        rows = MODULE.pd.DataFrame({'start': range(4), 'success': [True] * 4,
                                   'objective': [4.5] * 4, 'constraint_residual': [1e-13] * 4,
                                   'iterations': [20] * 4, 'stationarity_gap': [3e-7] * 4})
        (folder / '核岭推荐_目标对照.csv').write_text(
            'mixture,predicted_weighted_loss,M_p\n参考配比,4.7,0\n问题一有界推荐,4.5,-0.2\n', encoding='utf-8')
        rows.to_csv(folder / '核岭推荐_多初值.csv', index=False)
        return rows, folder / '核岭推荐_多初值.csv'

    def test_real_kernel_runs_reject_nonfinite_and_negative_diagnostics(self):
        rows, path = self.write_kernel_runs()
        self.assertTrue(MODULE.check_kernel_runs(self.root)['equal'])
        for field in ['objective', 'constraint_residual', 'stationarity_gap', 'iterations']:
            for value in [float('-inf'), float('inf'), float('nan')]:
                with self.subTest(field=field, value=value):
                    bad = rows.astype({field: float}).copy(); bad.loc[0, field] = value
                    bad.to_csv(path, index=False)
                    self.assertFalse(MODULE.check_kernel_runs(self.root)['equal'])
        bad = rows.copy(); bad.loc[0, 'constraint_residual'] = -1e-12
        bad.to_csv(path, index=False)
        self.assertFalse(MODULE.check_kernel_runs(self.root)['equal'])

    def test_real_kernel_runs_require_four_distinct_starts_and_typed_fields(self):
        rows, path = self.write_kernel_runs()
        invalid = [MODULE.pd.concat([rows.iloc[[0]]] * 4, ignore_index=True),
                   rows.assign(start=[0, 1, 2, 4]), rows.assign(success=[1] * 4),
                   rows.assign(iterations=[1.5] * 4), rows.assign(iterations=[0] * 4),
                   rows.drop(columns=['stationarity_gap'])]
        for bad in invalid:
            with self.subTest(columns=list(bad), values=bad.iloc[0].to_dict()):
                bad.to_csv(path, index=False)
                self.assertFalse(MODULE.check_kernel_runs(self.root)['equal'])

    def test_real_kernel_runs_allow_only_roundoff_negative_gap(self):
        rows, path = self.write_kernel_runs()
        rows.loc[0, 'stationarity_gap'] = -1e-13
        rows.to_csv(path, index=False)
        result = MODULE.check_kernel_runs(self.root)
        self.assertTrue(result['equal'])
        self.assertIn('recorded protected-gradient gap', result['validation'])
        rows.loc[0, 'stationarity_gap'] = -1e-7
        rows.to_csv(path, index=False)
        self.assertFalse(MODULE.check_kernel_runs(self.root)['equal'])

    def write_kernel_certificate(self):
        data = self.root / 'data'; raw = data / 'A_data_value/regmix_tables'
        raw.mkdir(parents=True)
        (raw / 'train_mixture_1m.csv').write_text(
            'index,train_the_pile_a,train_the_pile_b\n0,0.2,0.8\n1,0.8,0.2\n2,0.5,0.5\n', encoding='utf-8')
        (raw / 'train_pile_loss_1m.csv').write_text('index,metric/the_pile_a_val_loss\n0,4\n', encoding='utf-8')
        folder = self.root / '求解/问题一/结果'
        (folder / '核岭推荐_训练凸组合.csv').write_text('index,convex_weight\n0,0.5\n1,0.5\n2,0\n', encoding='utf-8')
        recipe = folder / '推荐配比调整.csv'
        recipe.write_text('domain,reference_mixture,recommended_mixture\na,0.5,0.5\nb,0.5,0.5\n', encoding='utf-8')
        return data, folder, recipe

    def test_real_kernel_certificate_rejects_nan_reference(self):
        data, folder, recipe = self.write_kernel_certificate()
        self.assertTrue(MODULE.check_kernel_certificate(self.root, data)['equal'])
        recipe.write_text('domain,reference_mixture,recommended_mixture\na,,0.5\nb,0.5,0.5\n', encoding='utf-8')
        self.assertFalse(MODULE.check_kernel_certificate(self.root, data)['equal'])

    def test_real_kernel_certificate_allows_nonunique_convex_weights(self):
        data, folder, recipe = self.write_kernel_certificate()
        self.assertTrue(MODULE.check_kernel_certificate(self.root, data)['equal'])
        (folder / '核岭推荐_训练凸组合.csv').write_text('index,convex_weight\n0,0\n1,0\n2,1\n', encoding='utf-8')
        self.assertTrue(MODULE.check_kernel_certificate(self.root, data)['equal'])


if __name__ == "__main__":
    unittest.main()
