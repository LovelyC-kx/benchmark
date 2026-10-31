import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'scripts/audit_campaign_failures.py'
spec=importlib.util.spec_from_file_location('failure_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class FailureAudit(unittest.TestCase):
    def test_log_and_engine_evidence_and_selected_intersection_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old=base/'old';stage=base/'stage';runs=old/'runs'
            for p in (stage,runs/'_queue_specs'):p.mkdir(parents=True)
            payload=json.dumps({'task':{'family':'tagged','name':'tagged_history','T':16},
                'model':{'backbone':'mlp','neuron':'lif_hard'},'train':{'seed':11}})
            keys=[f'{i:024x}' for i in range(3)]
            with sqlite3.connect(old/'train.sqlite') as con:
                con.execute('CREATE TABLE jobs(id TEXT,spec TEXT,state TEXT,error TEXT)')
                for key in keys:con.execute('INSERT INTO jobs VALUES(?,?,?,?)',(key,payload,'failed','train exit 1'))
            with sqlite3.connect(stage/'train.sqlite') as con:
                con.execute('CREATE TABLE jobs(id TEXT)');con.execute('INSERT INTO jobs VALUES(?)',(keys[0],))
            (stage/'campaign.json').write_text(json.dumps({'source_root':str(old),'training_output':str(runs)}))
            run=runs/keys[0];run.mkdir()
            (run/'failure.json').write_text(json.dumps({'error':'FloatingPointError: Nonfinite native loss','stage':'native_loss'}))
            (runs/'_queue_specs'/(keys[0]+'.log')).write_text('RuntimeError: generic wrapper\n')
            (runs/'_queue_specs'/(keys[1]+'.log')).write_text('Traceback...\nModuleNotFoundError: No module named test_package\n')
            before=(old/'train.sqlite').read_bytes()
            with patch('builtins.print'):report=audit.audit(stage,base/'report.json')
            self.assertEqual(before,(old/'train.sqlite').read_bytes())
            self.assertEqual(report['failed_rows_examined'],3)
            self.assertEqual(report['selected_stage_failures'][0]['cause'],'FloatingPointError: Nonfinite native loss')
            self.assertIn(('ModuleNotFoundError',1),report['cause_classes'])
            self.assertEqual(len(report['selected_stage_failures']),1)
            self.assertEqual(report['all_failed_rows'][2]['cause_source'],'queue_only')

    def test_tail_is_bounded_and_incomplete_json_can_fall_back_to_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs=Path(tmp);key='a'*24;run=runs/key;run.mkdir();(runs/'_queue_specs').mkdir()
            (run/'failure.json').write_text('{incomplete')
            log=runs/'_queue_specs'/(key+'.log');log.write_text('x'*20000+'\nRuntimeError: shape mismatch\n')
            row=audit.evidence(key,'{}','train exit 1',runs)
            self.assertLessEqual(len(row['log_tail']),8192)
            self.assertEqual(row['cause'],'RuntimeError: shape mismatch')


if __name__=='__main__':unittest.main()
