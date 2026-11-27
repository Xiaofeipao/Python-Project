"""One decision owner, credential-free workers and final-only test access."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from superagent.config import TaskConfig
from superagent.governance.test_access import install_test_guard
from superagent.runtime.orchestrator import run_worker


class SingleAgentContract(unittest.TestCase):
    def test_only_planner_identity_is_supported(self):
        TaskConfig(train_csv='unused', target_column='label').validate()
        for identity in ('critic', 'worker', '', None):
            with self.assertRaisesRegex(ValueError, 'single-agent'):
                TaskConfig(train_csv='unused', target_column='label', llm_agent_id=identity).validate()

    def test_worker_does_not_inherit_planner_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            config = TaskConfig(train_csv='unused', target_column='label', llm_api_key_env='CUSTOM_KEY')
            with patch.dict(os.environ, {'CUSTOM_KEY': 'synthetic-secret', 'HKUST_API_KEY': 'synthetic-secret'}), patch(
                    'superagent.runtime.orchestrator.subprocess.Popen') as launch:
                launch.return_value.pid = 123
                launch.return_value.wait.return_value = 1
                run_worker({'config': config.to_dict()}, Path(folder)/'worker', 1, 1)
                environment = launch.call_args.kwargs['env']
                self.assertNotIn('CUSTOM_KEY', environment)
                self.assertNotIn('HKUST_API_KEY', environment)
                from superagent.models.cache import text_cache_root, torch_cache_root
                self.assertEqual(environment['SUPERAGENT_MODEL_CACHE'], str(text_cache_root()))
                self.assertEqual(environment['TORCH_HOME'], str(torch_cache_root()))
                self.assertEqual(os.environ['CUSTOM_KEY'], 'synthetic-secret')

    def test_test_guard_blocks_path_and_file_identity_alias_until_released(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.csv'
            path.write_text('feature\n1\n', encoding='utf-8')
            alias = Path(folder)/'alias.csv'
            alias.write_text('feature\n1\n', encoding='utf-8')
            active = install_test_guard(path)
            try:
                # Simulate a hardlink identity without requiring OS link privileges.
                with patch('superagent.governance.test_access.os.path.samefile', return_value=True):
                    for candidate in (path, alias):
                        with self.assertRaises(PermissionError):
                            candidate.read_text(encoding='utf-8')
            finally:
                active[0] = False
            self.assertEqual(path.read_text(encoding='utf-8'), alias.read_text(encoding='utf-8'))

    def test_batch_uses_shared_policy_without_implicit_freeze(self):
        import scripts.workflows.run_released_batch as batch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []
            def invoke(directory, entry, *args):
                calls.append(args[0])
                if args[0] == 'train':
                    config = json.loads(Path(args[args.index('--config')+1]).read_text(encoding='utf-8'))
                    self.assertEqual(config['evaluation_stage'], 'development')
                    self.assertNotIn('frozen_manifest', config)
                    return {'status': 'completed'}
                return {'score': .5}
            with patch('sys.argv', ['batch', '--output', str(root/'run')]), patch.object(batch, 'ROOT', root), patch.object(
                    batch, 'runtime_fingerprint', return_value='synthetic-source'), patch.object(batch, 'invoke', side_effect=invoke), patch(
                    'builtins.print'), patch.dict(os.environ):
                batch.main()
            self.assertEqual(calls, ['train', 'predict', 'evaluate']*2)
            self.assertFalse((root/'run/frozen_agent').exists())


if __name__ == '__main__':
    unittest.main()
