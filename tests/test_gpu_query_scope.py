import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import efficient_campaign as ec


class GPUQueryScope(unittest.TestCase):
    def check(self,scope):
        seen=[]
        def fake(command,**kwargs):
            seen.append(command)
            return '1, GPU-test, 85651, 80000\n' if any(
                part.startswith('--query-gpu=') for part in command) else 'GPU-test, 123\n'
        with patch.dict(os.environ,scope,clear=True),patch.object(ec.subprocess,'check_output',side_effect=fake):
            self.assertEqual(ec.gpu_inventory()['GPU-test']['index'],1)
            self.assertEqual(ec.gpu_compute_processes(),[('GPU-test',123)])
        return seen

    def test_both_inventory_and_process_queries_exclude_unselected_failed_card(self):
        self.assertTrue(all(command[-2:]==['-i','1,3'] for command in self.check({'SNN_GPU_QUERY_IDS':'1,3'})))

    def test_no_environment_override_preserves_original_query_arguments(self):
        self.assertTrue(all('-i' not in command for command in self.check({})))


if __name__=='__main__':unittest.main()
