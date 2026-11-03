"""Run the experiment on a Colab GPU (a G4 runtime, NVIDIA RTX PRO 6000 Blackwell 96 GB, was used). In a Colab cell:

  !git clone -q https://github.com/speechlab0210/small-model-self-edit /content/selfedit_code
  !python /content/selfedit_code/colab/run_in_colab.py full      # or: pilot (tiny sizes, only to find bugs)

It installs the packages, runs run_all.py in the foreground (so the runtime stays busy), writes the whole console
to <work>/console.log and, at the end, packs <work>/out into /content/selfedit_results_<mode>.tar.gz.
"""
import os
import subprocess
import sys
import time
from packing import pack_results

MODE = sys.argv[1] if len(sys.argv) > 1 else "pilot"
if MODE not in ("pilot", "full"):
    raise SystemExit("mode must be pilot or full")
EXTRA = sys.argv[2:]
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.environ.get("SELFEDIT_WORK_PILOT" if MODE == "pilot" else "SELFEDIT_WORK",
                   "/content/selfedit_pilot" if MODE == "pilot" else "/content/selfedit")
os.makedirs(W, exist_ok=True)

marker = "/content/.selfedit_pip_audit_v2_done"
if not os.path.exists(marker):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(CODE, "requirements-run.txt")],
                       capture_output=True, text=True, timeout=1800)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if r.returncode == 0:
        open(marker, "w").write("ok")
    else:
        raise SystemExit(r.returncode)
print(subprocess.run([sys.executable, "-c", "import vllm, transformers, torch; print('vllm', vllm.__version__, "
                      "'transformers', transformers.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"],
                     capture_output=True, text=True).stdout, flush=True)

env = dict(os.environ, SELFEDIT_WORK=W, SELFEDIT_WORK_PILOT=W, TOKENIZERS_PARALLELISM="false",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
args = [sys.executable, "-u", os.path.join(CODE, "run_all.py")] + (["--pilot"] if MODE == "pilot" else []) + EXTRA
t0 = time.time()
p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=CODE)
with open(os.path.join(W, "console.log"), "a") as logf:
    for line in p.stdout:
        logf.write(line)
        logf.flush()
        if line[:4].isdigit() or "Error" in line or "Traceback" in line or "STOP" in line:
            print(line, end="", flush=True)
code = p.wait()
out = os.path.join(W, "out")
dst = f"/content/selfedit_results_{MODE}.tar.gz"
if os.path.isdir(out):
    pack_results(W, dst)
    print(f"packed {dst}: {os.path.getsize(dst) / 1e6:.1f} MB", flush=True)
print(f"exit code {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
# keep the runtime busy for a while so that the packed results can be downloaded (an idle runtime is soon recycled);
# create /content/selfedit_release to stop holding early
hold = float(os.environ.get("SELFEDIT_HOLD_HOURS", "0")) if code == 0 else 0.0
t1 = time.time()
while time.time() - t1 < hold * 3600 and not os.path.exists("/content/selfedit_release"):
    time.sleep(60)
    if int(time.time() - t1) % 1800 < 60:
        print(f"holding the runtime for the download ({(time.time() - t1) / 3600:.1f} of {hold} h)", flush=True)

raise SystemExit(code if code >= 0 else 128 - code)
