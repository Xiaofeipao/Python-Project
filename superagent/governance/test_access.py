"""Audit-hook backstop for the configured test file in this interpreter."""
import os
import sys


def install_test_guard(path):
    active = [bool(path)]
    target = os.path.normcase(os.path.abspath(path)) if path else None
    def audit(event, args):
        if active[0] and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.normcase(os.path.abspath(os.fsdecode(args[0]))) == target:
                raise PermissionError('Test file access blocked before final selection')
    sys.addaudithook(audit)
    return active
