"""Grounding jobs use lcy's shared advisory gpu.lock and record wyz ownership."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coordination",required=True);p.add_argument("--run-id",required=True)
    p.add_argument("--output",required=True);p.add_argument("--seconds",required=True,type=int)
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command or a.seconds<=0 or Path(a.run_id).name!=a.run_id: p.error("invalid command, budget or run ID")
    root=Path(a.coordination)
    with (root/"gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in root.glob("*.json"):
            if json.loads(path.read_text()).get("status")=="running":
                raise RuntimeError("Unresolved running record: "+path.name)
        processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
        memory=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
        if processes or any(int(v)>256 for v in memory.splitlines()): raise RuntimeError("GPU occupied")
        path=root/(a.run_id+".json")
        record=dict(role="grounding",executed_by="wyz",run_id=a.run_id,status="running",
                    wrapper_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
                    budget_seconds=a.seconds,output=a.output,command=command,
                    git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
        with path.open("x") as f: json.dump(record,f,indent=2)
        process=None;start=time.monotonic()
        try:
            process=subprocess.Popen(command,start_new_session=True)
            record["child_pid"]=process.pid
            path.write_text(json.dumps(record,indent=2)+"\n")
            code=process.wait(timeout=a.seconds)
            record.update(status="completed" if code==0 else "failed",exit_code=code)
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            record["status"]="failed";raise
        finally:
            record["elapsed_seconds"]=time.monotonic()-start
            record["finished_at"]=datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(record,indent=2)+"\n")
        raise SystemExit(code)
if __name__=="__main__": main()
