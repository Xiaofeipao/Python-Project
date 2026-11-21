"""Bounded automatic process restart, always into a new audit directory."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--attempts',type=int,choices=[1,2,3],default=2)
    args=p.parse_args()
    entry=Path(__file__).resolve().parents[2]/'main.py'
    previous=None
    for index in range(args.attempts):
        output=args.output/f'attempt_{index:02d}'
        command=[sys.executable,str(entry),'train','--config',str(Path(args.config).resolve()),'--output',str(output)]
        if previous:
            command+=['--resume-from',str(previous)]
        result=subprocess.run(command)
        if result.returncode in (0,130):
            return result.returncode
        if not (output/'manifest.json').exists():
            return result.returncode
        previous=output
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
