"""Resume only verified completed trials; preserve elapsed time and API ledger."""
import json
import contextlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.restype=wintypes.HANDLE
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=api.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5
        code=wintypes.DWORD()
        try:
            return not api.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value==259
        finally:api.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:return False
    except PermissionError:return True


@contextlib.contextmanager
def run_lock(folder):
    folder=Path(folder).resolve()
    folder.parent.mkdir(parents=True,exist_ok=True)
    handle=open(folder.with_name(folder.name+'.run.lock'),'a+b')
    try:
        handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        import os
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise RuntimeError('Run is active or locked; concurrent resume refused')
    try:
        yield
    finally:
        handle.close()


def load_resume(folder, config):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['config'] != config.to_dict():
        raise ValueError('Resume requires exactly the original configuration')
    summary = folder / 'summary.json'
    if summary.exists() and json.loads(summary.read_text())['status'] == 'completed':
        raise ValueError('Completed runs cannot be resumed')
    events = [json.loads(l) for l in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    origin = manifest.get('original_started_at', events[0]['timestamp_utc'])
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(origin)).total_seconds()
    if elapsed >= config.time_budget_seconds:
        raise ValueError('Original wall-clock budget expired, including downtime')
    path = folder / 'trials.jsonl'
    history = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] if path.exists() else []
    committed = {h['trial'] for h in history}
    for entry in events:
        if entry.get('event') == 'trial_started' and entry['trial'] not in committed:
            trial_folder=folder / 'experiments' / f"trial_{entry['trial']:03d}"
            pid_file=trial_folder/'worker_pid.json'
            if pid_file.exists():
                pid=json.loads(pid_file.read_text())['pid']
                while process_alive(pid):
                    if (datetime.now(timezone.utc)-datetime.fromisoformat(origin)).total_seconds()>=config.time_budget_seconds:
                        raise ValueError('Original budget expired while awaiting orphan worker')
                    time.sleep(.5)
            elif (trial_folder/'request.json').exists():
                raise ValueError('Cannot verify interrupted worker state; concurrent resume refused')
            result = trial_folder/'result.json'
            record = json.loads(result.read_text(encoding='utf-8')) if result.exists() else {
                'status':'interrupted', 'error':'Uncommitted interrupted candidate retained in memory; not repeated'}
            history.append({'trial':entry['trial'], 'candidate':entry['candidate'], 'reason':entry['reason'], **record})
            committed.add(entry['trial'])
    history.sort(key=lambda h:h['trial'])
    checkpoint = folder / 'planner_checkpoint.json'
    feedback=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}
    initial=next((e.get('restored_invalid_plans',0) for e in events if e.get('event')=='resume_verified'),0)
    invalid=initial;rejection=feedback.get('last_rejection')
    for entry in events:
        if entry.get('event')=='llm_invalid_plan':
            invalid+=1;rejection=entry.get('reason','Invalid structured plan')
        elif entry.get('event')=='llm_plan_selected':
            invalid=0;rejection=None
    feedback.update(consecutive_invalid_plans=invalid,last_rejection=rejection)
    reflections=(feedback.get('restored_reflections',[])+[{k:e.get(k) for k in ('diagnosis','next_action','candidate')} for e in events if e.get('event')=='llm_plan_selected'])[-3:]
    usage=json.loads((folder/'model_usage.json').read_text(encoding='utf-8'))
    ledger_path=Path(usage['ledger_path']).resolve()
    ledger_root=ledger_path.parent.parent
    configured_root=os.environ.get('SUPERAGENT_USAGE_ROOT')
    if configured_root and Path(configured_root).resolve()!=ledger_root:
        raise ValueError('Resume must reuse the original shared usage root')
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(origin)).total_seconds()
    return {'manifest': manifest, 'history': history, 'elapsed': max(0, elapsed), 'reflections':reflections,
            'ledger_root':str(ledger_root), 'ledger_path':str(ledger_path),
            'origin': origin, 'folder': folder,
            'planner': feedback,
            'environment': json.loads((folder / 'environment.json').read_text(encoding='utf-8'))}
