"""Resource installation boundaries using synthetic text only."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.install_rob_resources import install
from scripts.prepare_rob_prompts import prepare
from extensions.rob_resources import load_compact_guidance, load_resource_text
from extensions.common import ContractError


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'; self.source.mkdir()
        self.target = self.root/'target'; self.target.mkdir()
        def row(name, text):
            path = self.source/name; path.parent.mkdir(parents=True, exist_ok=True)
            data = text.encode(); path.write_bytes(data)
            return {'path':name, 'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)}
        self.entry = {'id':'fixture', 'tool':'Fixture', 'version':'1',
                      'guide':row('text/guide.txt','Synthetic source only. '*10),
                      'prompt':row('prompts/fixture.txt','Synthetic prompt only. '*10)}
        self.entries = [self.entry]
        (self.target/'catalog.json').write_text(json.dumps({'tools':self.entries}))

    def test_install_check_and_load_preserve_bytes_and_catalog(self):
        before = (self.target/'catalog.json').read_bytes()
        self.assertEqual(2, install(self.source,self.target,self.entries,check_only=True))
        self.assertFalse((self.target/'text/guide.txt').exists())
        install(self.source,self.target,self.entries)
        install(self.target,self.target,self.entries)
        self.assertEqual(before,(self.target/'catalog.json').read_bytes())
        self.assertEqual((self.source/'prompts/fixture.txt').read_text(),
                         load_compact_guidance(self.target,'Fixture','1'))

    def test_bad_second_file_leaves_existing_installation_untouched(self):
        install(self.source,self.target,self.entries)
        before={p.relative_to(self.target):p.read_bytes() for p in self.target.rglob('*') if p.is_file()}
        (self.source/'prompts/fixture.txt').write_text('Tampered')
        with self.assertRaises(ContractError): install(self.source,self.target,self.entries)
        self.assertEqual(before,{p.relative_to(self.target):p.read_bytes() for p in self.target.rglob('*') if p.is_file()})

    def test_missing_input_does_not_partially_install(self):
        (self.source/'prompts/fixture.txt').unlink()
        with self.assertRaisesRegex(ContractError,'README.md'): install(self.source,self.target,self.entries)
        self.assertFalse((self.target/'text').exists())

    def test_runtime_rejects_modified_installed_resource_and_wrong_version(self):
        install(self.source,self.target,self.entries)
        (self.target/'text/guide.txt').write_text('Changed')
        with self.assertRaises(ContractError): load_resource_text(self.target,'text/guide.txt')
        with self.assertRaises(ContractError): load_compact_guidance(self.target,'Fixture','other')

    def test_destination_symlink_cannot_escape(self):
        outside=self.root/'outside';outside.mkdir()
        (self.target/'prompts').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError): install(self.source,self.target,self.entries)
        self.assertFalse((self.target/'text').exists())
        self.assertEqual([],list(outside.iterdir()))

    def test_prompt_recovery_never_executes_historical_source(self):
        text=(self.source/'prompts/fixture.txt').read_text()
        legacy=self.root/'old.py'
        legacy.write_text('raise RuntimeError("must never execute")\nCOMPACT_TOOL_GUIDANCE = '+repr({'Fixture':text})+'\n')
        out=self.root/'recovered'
        self.assertEqual(1,prepare(legacy,out,self.entries))
        self.assertEqual(text.encode(),(out/'prompts/fixture.txt').read_bytes())
        legacy.write_text('COMPACT_TOOL_GUIDANCE = '+repr({'Fixture':'different'})+'\n')
        with self.assertRaises(ValueError): prepare(legacy,self.root/'failed',self.entries)
        self.assertFalse((self.root/'failed').exists())

    def test_source_scanner_rejects_external_text_even_if_force_added(self):
        from scripts import check_repository
        from unittest.mock import patch
        path=self.root/'resources/rob/prompts/fixture.txt';path.parent.mkdir(parents=True);path.write_text('Synthetic')
        with patch.object(check_repository,'ROOT',self.root):
            self.assertTrue(any('external RoB' in reason for _,reason in check_repository.findings([path])))


if __name__=='__main__': unittest.main()
