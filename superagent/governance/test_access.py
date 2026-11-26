"""Audit-hook backstop for the configured test file in this interpreter."""
import os
import sys


def install_test_guard(path):
    active = [bool(path)]
    target = os.path.normcase(os.path.realpath(path)) if path else None
    def audit(event, args):
        if active[0] and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            candidate = os.path.normcase(os.path.realpath(os.fsdecode(args[0])))
            alias = candidate == target
            if not alias:
                try:
                    alias = os.path.samefile(candidate, target)
                except OSError:
                    pass
            if alias:
                raise PermissionError('Test file access blocked before final selection')
    sys.addaudithook(audit)
    return active
