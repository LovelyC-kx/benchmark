"""Coverage regressions: protocol sections cannot disappear behind literature maps."""
import json
from pathlib import Path
import tempfile
import unittest

from snnbench.coverage import plan_inventory


class ProtocolHeadingCoverage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'SNN_Research_Master_Index_2026-10-24.md').write_text(
            '# Master\n## L3.6 跨时间\n**执行逻辑。** 独立场景。\n'
            '## L4.3 内容干预\n**执行逻辑。** 独立验证。\n'
            '## X.STATS 统计\n**执行逻辑。** 正确统计单位。\n')

    def tearDown(self):
        self.temp.cleanup()

    def sections(self):
        return {r['source_section_id']: r for r in plan_inventory(self.root)['specialized_subsections']}

    def test_unrouted_protocol_section_is_retained(self):
        p = self.root / 'SNN_L4_3_Deep_Review_and_Execution_2026-10-27.md'
        p.write_text('## 7. 4.3.7：任务效应\n## 8. 4.3.8：诊断到设计\n')
        rows = self.sections()
        self.assertEqual(set(rows), {'4.3.7', '4.3.8'})
        self.assertEqual(rows['4.3.8']['title'], '诊断到设计')
        self.assertFalse(rows['4.3.8']['complete_section_claim'])
        self.assertEqual(rows['4.3.8']['protocol_headings'][0]['line'], 2)

    def test_merge_map_and_protocol_without_losing_code_routes(self):
        references = self.root / 'package' / 'references'
        references.mkdir(parents=True)
        (references / 'plan_coverage.json').write_text(json.dumps({'subsections': [
            {'id': '4.3.2', 'purpose': 'erasure', 'code': ['erase.py']}]}))
        (self.root / 'SNN_L4_3_Deep_Review.md').write_text('## 2. 4.3.2：擦除\n')
        row = self.sections()['4.3.2']
        self.assertEqual(len(row['sources']), 2)
        self.assertEqual(row['code_candidates'][0]['path'], 'package/erase.py')
        self.assertEqual(row['title'], 'erasure')

    def test_references_counters_other_modules_and_nested_snapshots_are_not_requirements(self):
        (self.root / 'SNN_L4_3_Deep_Review.md').write_text(
            '## 1.2 普通计数\n正文提及4.3.8。\n## 4.4.1 另一模块引用\n'
            '## 1. 4.3.1 本模块\n')
        nested = self.root / 'historical_snapshot'
        nested.mkdir()
        (nested / 'SNN_L4_3_Deep_Review.md').write_text('## 4.3.99 旧快照\n')
        self.assertEqual(set(self.sections()), {'4.3.1'})

    def test_deeper_subsections_and_multiple_sources_preserve_provenance(self):
        for name in ['Deep_Review', 'Mathematical_Specification']:
            (self.root / ('SNN_L3_3_' + name + '.md')).write_text('### 3.3.3.2 条件交互\n')
        row = self.sections()['3.3.3.2']
        self.assertEqual(len(row['protocol_headings']), 2)
        self.assertEqual(len(row['sources']), 2)
        self.assertTrue(all(len(r['source_sha256']) == 64 for r in row['protocol_headings']))

    def test_inventory_is_not_execution_and_master_cards_remain_intact(self):
        (self.root / 'SNN_L6_1_Credit_Assignment.md').write_text('## 6.1.1 前向匹配\n')
        out = self.root / 'inventory.json'
        result = plan_inventory(self.root, out)
        self.assertEqual(result['counts']['master_cards'], 3)
        self.assertIn('6.1.1', {r['source_section_id'] for r in result['specialized_subsections']})
        self.assertFalse(result['all_planned_analysis_complete'])
        self.assertEqual(json.loads(out.read_text())['master_cards'][0]['source_module_id'], 'L3.6')


if __name__ == '__main__':
    unittest.main()
