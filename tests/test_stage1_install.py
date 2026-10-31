"""Offline installer tests: no downloaded file replaces code before all verify."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class Stage1Installer(unittest.TestCase):
    def fixture(self,base):
        project=base/'project';code=project/'src/benchmark';binpath=base/'bin'
        (project/'scripts').mkdir(parents=True);(code/'snnbench').mkdir(parents=True);binpath.mkdir()
        targets=[project/'scripts/compile_stage1.py',project/'scripts/efficient_campaign.py',code/'snnbench/coverage.py']
        for target in targets:target.write_bytes(b'original local file\n')
        fake=binpath/'curl'
        fake.write_text('#!'+sys.executable+'\n'+
            'import os,pathlib,sys\n'
            'args=sys.argv[1:];url=next(a for a in args if a.startswith("https://"))\n'
            'rel=url.split("/benchmark/",1)[1].split("/",1)[1]\n'
            'data=(pathlib.Path(os.environ["CURL_TEST_SOURCE"])/rel).read_bytes()\n'
            'if pathlib.Path(rel).name==os.environ.get("CURL_TEST_CORRUPT"):data+=b"bad download"\n'
            'pathlib.Path(args[args.index("-o")+1]).write_bytes(data)\n')
        fake.chmod(0o755)
        env=dict(os.environ,SNN_PROJECT_ROOT=str(project),SNN_CODE=str(code),SNN_PYTHON=sys.executable,
                 CURL_TEST_SOURCE=str(ROOT),PATH=str(binpath)+os.pathsep+os.environ['PATH'])
        return project,targets,env

    def test_verified_install_keeps_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            project,targets,env=self.fixture(Path(tmp))
            r=subprocess.run(['bash',str(ROOT/'scripts/install_stage1_compiler.sh'),'a'*40],env=env,
                             capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            for target,source in zip(targets,['scripts/compile_stage1.py','scripts/efficient_campaign.py','snnbench/coverage.py']):
                self.assertEqual(target.read_bytes(),(ROOT/source).read_bytes())
                backups=list(target.parent.glob(target.name+'.before_stage1_*'))
                self.assertEqual(len(backups),1)
                self.assertEqual(backups[0].read_bytes(),b'original local file\n')
            self.assertIn('INSTALL_OK',r.stdout)

    def test_bad_hash_changes_no_installed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            project,targets,env=self.fixture(Path(tmp));env['CURL_TEST_CORRUPT']='efficient_campaign.py'
            r=subprocess.run(['bash',str(ROOT/'scripts/install_stage1_compiler.sh'),'a'*40],env=env,
                             capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertTrue(all(t.read_bytes()==b'original local file\n' for t in targets))
            self.assertFalse(list(project.rglob('*.before_stage1_*')))

    def test_unpinned_ref_is_rejected(self):
        r=subprocess.run(['bash',str(ROOT/'scripts/install_stage1_compiler.sh'),'main'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)


if __name__=='__main__':unittest.main()
