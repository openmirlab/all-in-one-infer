"""Run an upstream diagnostic in an explicit interpreter without modifying its checkout.

Inputs are caller-owned read-only paths; intermediate files use a temporary cwd, and only
an explicitly requested JSON report persists. This reports observations, not numerical parity.
Reads: subprocess · pathlib · tempfile · JSON; the child imports the selected upstream module.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

CHILD = r'''
import importlib
import json
from pathlib import Path
import sys

request = json.loads(sys.argv[1])
package = importlib.import_module(request['module'])
output = Path(request['output']).resolve()


def is_checkout(directory):
    # A real repository has .git/HEAD; a worktree or submodule has a .git file.
    # An empty .git directory (e.g. a sandbox's read-only mount point in /tmp)
    # is not a checkout.
    marker = directory / '.git'
    return marker.is_file() or (marker / 'HEAD').is_file()


for source in (Path(package.__file__).resolve(), Path(sys.executable).absolute()):
    for directory in source.parents:
        if is_checkout(directory) and output.is_relative_to(directory):
            raise RuntimeError('Output must be outside the reference checkout: ' + str(directory))
reports = []
for index, audio in enumerate(request['audio']):
    workspace = Path.cwd() / str(index)
    workspace.mkdir()
    result = package.analyze(
        audio, device='cpu', include_activations=True,
        demix_dir=str(workspace / 'demix'), spec_dir=str(workspace / 'spec'),
        out_dir=str(workspace / 'results'),
    )
    bpm = getattr(result, 'bpm', None)
    beats = list(getattr(result, 'beats', []))
    if bpm is None or not beats:
        raise RuntimeError('Reference returned no BPM or beats for ' + audio)
    activations = getattr(result, 'activations', None)
    summary = None
    if activations is not None and 'beat' in activations:
        values = activations['beat']
        summary = {'maximum': float(values.max()), 'above_0_19': int((values > 0.19).sum())}
    reports.append({
        'audio': audio, 'bpm': float(bpm), 'beats': len(beats),
        'downbeats': len(getattr(result, 'downbeats', [])),
        'segments': len(getattr(result, 'segments', [])), 'beat_activations': summary,
    })
Path('report.json').write_text(json.dumps({
    'module': request['module'], 'module_file': package.__file__,
    'version': getattr(package, '__version__', None), 'observations': reports,
    'claim': 'Diagnostic observations only; no baseline or numerical parity claim.',
}, indent=2) + '\n')
'''


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--reference-python', required=True, type=Path,
                      help='Existing interpreter with the upstream package already installed.')
  parser.add_argument('--module', default='allin1', choices=('allin1', 'allinone'),
                      help='Upstream import name; no automatic fallback or installation.')
  parser.add_argument('--output', required=True, type=Path,
                      help='New JSON report path outside the reference checkout; never overwritten.')
  parser.add_argument('audio', nargs='+', type=Path, help='Existing audio files, read in place.')
  args = parser.parse_args(argv)
  interpreter = args.reference_python.expanduser().absolute()
  if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
    parser.error(f'--reference-python must be an existing executable: {interpreter}')
  audio = [path.expanduser().resolve() for path in args.audio]
  missing = [str(path) for path in audio if not path.is_file()]
  if missing:
    parser.error('Audio input does not exist: ' + ', '.join(missing))
  output = args.output.expanduser().absolute()
  if output.exists() or not output.parent.is_dir():
    parser.error('--output must be a new file in an existing directory')
  environment = os.environ.copy()
  environment.pop('PYTHONPATH', None)
  environment['PYTHONDONTWRITEBYTECODE'] = '1'
  with tempfile.TemporaryDirectory(prefix='allinone-reference-') as cwd:
    cache = Path(cwd, 'cache')
    environment.update({
      'TORCH_HOME': str(cache / 'torch'), 'HF_HOME': str(cache / 'huggingface'),
      'HF_HUB_CACHE': str(cache / 'huggingface/hub'),
      'HUGGINGFACE_HUB_CACHE': str(cache / 'huggingface/hub'),
      'XDG_CACHE_HOME': str(cache), 'NUMBA_CACHE_DIR': str(cache / 'numba'),
      'MPLCONFIGDIR': str(cache / 'matplotlib'),
    })
    completed = subprocess.run(
      [str(interpreter), '-B', '-c', CHILD,
       json.dumps({'module': args.module, 'audio': [str(path) for path in audio],
                   'output': str(output)})],
      cwd=cwd, env=environment, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
      parser.exit(1, 'Reference diagnostic failed:\n' + completed.stdout + completed.stderr)
    report = Path(cwd, 'report.json').read_text()
  with output.open('x') as handle:
    handle.write(report)
  print(f'Diagnostic saved to {output}; no numerical parity claim.')


if __name__ == '__main__':
  main()
