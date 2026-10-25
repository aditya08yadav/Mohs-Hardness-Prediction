"""Execute the complete notebook using the active Python environment."""
import os
import sys
from pathlib import Path

# Keep CPU use bounded before starting the kernel.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
notebook = ROOT / "project.ipynb"
book = nbformat.read(notebook, as_version=4)
manager = KernelManager(kernel_name="python3")
# Use this interpreter even if a global python3 kernelspec points elsewhere.
manager.kernel_spec.argv = [
    sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
]
client = NotebookClient(
    book, km=manager, timeout=300,
    resources={"metadata": {"path": str(ROOT)}}
)
print(f"Executing {notebook.name} with {sys.executable}", flush=True)
try:
    client.execute()
except Exception:
    nbformat.write(book, ROOT / "project.failed.ipynb")
    raise
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    manager.cleanup_resources()
nbformat.write(book, notebook)
print("Completed. Saved executed notebook and results/ artifacts.", flush=True)
