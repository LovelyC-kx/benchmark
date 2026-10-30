"""Server path contracts survive scp layouts and reach actual CLI operations."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from snnbench.campaign import campaign_plan

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('server_layout', REPO/'scripts/setup_server_layout.py')
LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)


class ServerLayout(unittest.TestCase):
    def test_wrappers_reach_cli_with_quoted_paths_and_preserve_custom_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"project with 'quoted' paths"
            result = LAYOUT.setup(root, REPO)
            self.assertEqual(result['paths']['SNN_DATA'], str(root/'data/cache/campaign_v04'))
            for path in (root/'scripts').glob('*.sh'):
                subprocess.run(['bash','-n',str(path)],check=True)
                self.assertNotIn(b'\r\n',path.read_bytes())
            env=dict(os.environ,SNN_PYTHON=sys.executable)
            help_result=subprocess.run(['bash',str(root/'scripts/benchmark.sh'),'--help'],cwd=tmp,env=env,capture_output=True,text=True,check=True)
            self.assertIn('campaign-plan',help_result.stdout)
            self.assertEqual(result,LAYOUT.setup(root,REPO))
            config=root/'configs/server_paths.sh';original=config.read_text()+'\n# local customization\n';config.write_text(original)
            with self.assertRaises(FileExistsError):LAYOUT.setup(root,REPO)
            self.assertEqual(config.read_text(),original)
            LAYOUT.setup(root,REPO,replace_config=True)
            backups=list(config.parent.glob('server_paths.sh.previous-*'))
            self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_text(),original)

    def test_archive_revision_marker_and_missing_master_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'src/benchmark';(code/'snnbench').mkdir(parents=True)
            for relative in ('pyproject.toml','snnbench/cli.py','SNN_Research_Master_Index_2026-10-24.md'):
                (code/relative).write_text('fixture\n')
            (root/'src/benchmark_source.commit').write_text('a'*40+'\n')
            result=LAYOUT.setup(root,code);self.assertEqual(result['source_revision'],'a'*40)
            (code/'SNN_Research_Master_Index_2026-10-24.md').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'complete repository'):LAYOUT.setup(root,code)

    def test_campaign_cache_override_is_persisted_and_cannot_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);campaign=root/'results/campaign_v04';cache=root/'data/cache/campaign_v04'
            result=campaign_plan(campaign,seeds=(11,),profiles=('smoke',),data_root=cache)
            self.assertEqual(result['data_output'],str(cache.resolve()))
            result2=campaign_plan(campaign,seeds=(11,),profiles=('smoke',),data_root=cache)
            self.assertEqual(result2['data_output'],result['data_output'])
            with self.assertRaisesRegex(ValueError,'data location changed'):
                campaign_plan(campaign,seeds=(11,),profiles=('smoke',),data_root=root/'other')


if __name__=='__main__':unittest.main()
