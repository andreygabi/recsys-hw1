"""Execute the notebook in the current Python environment."""

import json
import os
from pathlib import Path
import sys
from time import perf_counter

import nbformat
from nbclient import NotebookClient


root = Path(__file__).resolve().parent
kernel_root = root / '.cache' / 'jupyter'
kernel_dir = kernel_root / 'kernels' / 'recsys-hw1'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Recsys HW1', 'language': 'python',
}))
os.environ['JUPYTER_PATH'] = str(kernel_root) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
notebook = nbformat.read(root / 'experiments.ipynb', as_version=4)
started = perf_counter()


def checkpoint(cell, cell_index, **kwargs):
    nbformat.write(notebook, root / '.cache' / 'executing.ipynb')
    print(f'Cell {cell_index + 1}/{len(notebook.cells)} completed; '
          f'{perf_counter() - started:.0f}s elapsed', flush=True)


client = NotebookClient(notebook, timeout=3600, kernel_name='recsys-hw1',
                        resources={'metadata': {'path': str(root)}}, on_cell_executed=checkpoint)
client.execute()
nbformat.write(notebook, root / 'experiments.ipynb')
print(f'Executed notebook saved; total {perf_counter() - started:.1f}s', flush=True)
