"""Execute both notebooks with this interpreter, without rewriting their saved outputs."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root=Path(__file__).resolve().parents[1]
for name in ('phase1','phase2'):
    notebook=nbformat.read(root/f'notebooks/{name}.ipynb',as_version=4)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(notebook,timeout=300,km=manager,resources={'metadata':{'path':str(root/'notebooks')}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f'{name}: all {len(notebook.cells)} cells executed with {sys.executable}')
