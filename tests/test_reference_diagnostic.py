"""Guard the opt-in diagnostic's read-only reference boundary and failure reporting.

Reads: tools/reference_diagnostic.py · a fake upstream package in a separate interpreter.
"""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/reference_diagnostic.py'))


def _reference(tmp_path, fail=False):
  root = tmp_path / 'reference'
  package = root / 'allin1'
  package.mkdir(parents=True)
  (root / '.git').mkdir()
  (root / '.git' / 'HEAD').write_text('ref: refs/heads/main\n')
  (package / '__init__.py').write_text('''
import os
from pathlib import Path
from types import SimpleNamespace

def analyze(audio, **kwargs):
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    assert Path(os.environ['TORCH_HOME']).is_relative_to(Path.cwd())
    assert Path(audio).is_absolute() and Path(audio).read_bytes() == b'read-only audio'
    for key in ('demix_dir', 'spec_dir', 'out_dir'):
        assert Path(kwargs[key]).is_relative_to(Path.cwd())
    return SimpleNamespace(bpm=120, beats=[0, 0.5], downbeats=[0], segments=[1], activations=None)
''' + ("\nraise RuntimeError('deliberate upstream failure')\n" if fail else ''))
  interpreter = root / 'python'
  interpreter.write_text(
    f'#!{sys.executable}\nimport sys\n'
    f'sys.path.insert(0, {str(root)!r})\n'
    'script, request = sys.argv[-2:]\nsys.argv = [sys.argv[0], request]\n'
    "exec(script, {'__name__': '__main__'})\n"
  )
  interpreter.chmod(0o755)
  return root, interpreter


def _files(root):
  return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in root.rglob('*') if p.is_file()}


def test_reference_diagnostic_preserves_reference_tree_and_audio(tmp_path):
  reference, interpreter = _reference(tmp_path)
  audio = tmp_path / 'input.wav'
  audio.write_bytes(b'read-only audio')
  output = tmp_path / 'observations.json'
  before = _files(reference)
  TOOL['main'](['--reference-python', str(interpreter), '--output', str(output), str(audio)])
  assert _files(reference) == before
  assert audio.read_bytes() == b'read-only audio'
  report = json.loads(output.read_text())
  assert report['observations'][0]['beats'] == 2
  assert 'no baseline or numerical parity claim' in report['claim']


@pytest.mark.parametrize('failure', ['missing_interpreter', 'missing_audio', 'existing_output'])
def test_reference_diagnostic_rejects_invalid_explicit_paths(tmp_path, failure):
  _, interpreter = _reference(tmp_path)
  audio = tmp_path / 'input.wav'
  audio.write_bytes(b'read-only audio')
  output = tmp_path / 'observations.json'
  if failure == 'missing_interpreter':
    interpreter = tmp_path / 'absent-python'
  elif failure == 'missing_audio':
    audio.unlink()
  else:
    output.write_text('preserve me')
  with pytest.raises(SystemExit) as error:
    TOOL['main'](['--reference-python', str(interpreter), '--output', str(output), str(audio)])
  assert error.value.code == 2
  if output.exists():
    assert output.read_text() == 'preserve me'


def test_reference_diagnostic_rejects_output_inside_reference_checkout(tmp_path):
  reference, interpreter = _reference(tmp_path)
  audio = tmp_path / 'input.wav'
  audio.write_bytes(b'read-only audio')
  output = reference / 'report.json'
  before = _files(reference)
  with pytest.raises(SystemExit) as error:
    TOOL['main'](['--reference-python', str(interpreter), '--output', str(output), str(audio)])
  assert error.value.code == 1
  assert _files(reference) == before
  assert not output.exists()


def test_reference_failure_does_not_create_success_report(tmp_path):
  _, interpreter = _reference(tmp_path, fail=True)
  audio = tmp_path / 'input.wav'
  audio.write_bytes(b'read-only audio')
  output = tmp_path / 'observations.json'
  with pytest.raises(SystemExit) as error:
    TOOL['main'](['--reference-python', str(interpreter), '--output', str(output), str(audio)])
  assert error.value.code == 1
  assert not output.exists()


@pytest.mark.parametrize('marker', ['empty_dir', 'worktree_file'])
def test_reference_diagnostic_checkout_detection(tmp_path, marker):
  """An empty .git directory above the output (e.g. a sandbox's mount point in
  /tmp) is not a checkout; a worktree's .git file is."""
  reference, interpreter = _reference(tmp_path)
  audio = tmp_path / 'input.wav'
  audio.write_bytes(b'read-only audio')
  if marker == 'empty_dir':
    (tmp_path / '.git').mkdir()
    output = tmp_path / 'observations.json'
    TOOL['main'](['--reference-python', str(interpreter), '--output', str(output), str(audio)])
    assert json.loads(output.read_text())['observations'][0]['beats'] == 2
  else:
    (reference / '.git' / 'HEAD').unlink()
    (reference / '.git').rmdir()
    (reference / '.git').write_text('gitdir: /elsewhere/.git/worktrees/reference\n')
    with pytest.raises(SystemExit) as error:
      TOOL['main'](['--reference-python', str(interpreter), '--output', str(reference / 'r.json'),
                    str(audio)])
    assert error.value.code == 1
