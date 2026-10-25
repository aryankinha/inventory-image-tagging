"""Execute all teaching cells using the current uv environment's Python kernel."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/phase1.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name='python3',
                        resources={'metadata':{'path':str(root/'notebooks')}})
client.execute()
nbformat.write(notebook,path)
print('All notebook cells executed successfully.')
