import base64
import json
import tempfile
import unittest
from pathlib import Path
from superagent.governance.usage_ledger import ModelCatalog


class EvidenceArchiveTests(unittest.TestCase):
    def test_lossless_archives_roundtrip_and_detect_tampering(self):
        root=Path(__file__).resolve().parents[2]
        for registry in (root/'configs').glob('model_registry*.json'):
            catalog=ModelCatalog(registry)
            with tempfile.TemporaryDirectory() as d:
                target=Path(d)/'policy'
                catalog.snapshot(target)
                copied=ModelCatalog(target/'registry.json')
                evidence=next(e for model in copied.models.values() for e in model.get('evidence',[]) if e.get('encoding')=='base64')
                path=target/evidence['snapshot']
                archive=json.loads(path.read_text())
                archive['data_base64']=base64.b64encode(b'tampered evidence').decode()
                path.write_text(json.dumps(archive),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'evidence snapshot changed'):
                    ModelCatalog(target/'registry.json')
