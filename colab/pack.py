"""Packs <work>/out and console.log again into /content/selfedit_results_<mode>.tar.gz, by the same rule as
run_in_colab.py (files over 50 MB and the resumable state.pt files stay on the machine). Used after an analysis that
was run on the machine once the main run had finished (analysis/exploratory_direction.py).

  python colab/pack.py full
"""
import os
import sys
from packing import pack_results

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
W = "/content/selfedit_pilot" if MODE == "pilot" else "/content/selfedit"
dst = f"/content/selfedit_results_{MODE}.tar.gz"


pack_results(W, dst)
print(f"packed {dst}: {os.path.getsize(dst) / 1e6:.1f} MB", flush=True)
