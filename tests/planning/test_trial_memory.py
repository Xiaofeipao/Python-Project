import tempfile
import unittest
from pathlib import Path
from superagent.planning.memory import trial_key, family_stats
from superagent.governance.test_access import install_test_guard


class MemoryTests(unittest.TestCase):
    def test_hash_and_family_retirement(self):
        self.assertEqual(trial_key({'family':'x','C':1}), trial_key({'C':1,'family':'x'}))
        stats=family_stats([{'candidate':{'family':'x'},'status':'failed'}]*2,'accuracy')
        self.assertTrue(stats['x']['retired'])

    def test_runtime_test_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.csv';p.write_text('x')
            guard=install_test_guard(p)
            try:
                with self.assertRaises(PermissionError):p.read_text()
            finally:guard[0]=False
            self.assertEqual(p.read_text(),'x')
