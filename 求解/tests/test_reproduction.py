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


if __name__ == "__main__":
    unittest.main()
