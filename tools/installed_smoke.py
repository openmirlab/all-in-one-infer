"""Verify installed production bytes, checkpoint data, and all NATTEN golden output fields.

Run outside the checkout with fixture and delivery-baseline JSON paths. No checkpoint
network access or pretrained model load is needed for these package/attention contracts.
Reads: installed allin1_infer API and neighborhood attention · torch · recorded baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import allin1_infer
from allin1_infer.models import neighborhood_attention as na


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('fixture', type=Path)
  parser.add_argument('baseline', type=Path)
  args = parser.parse_args()
  package = Path(allin1_infer.__file__).resolve().parent
  assert package.is_relative_to(Path(sys.prefix).resolve()), package
  assert not package.is_relative_to(Path(__file__).resolve().parents[1] / 'src'), package
  baseline = json.loads(args.baseline.read_text())
  expected = {name.removeprefix('src/allin1_infer/'): digest
              for name, digest in baseline['source_sha256'].items()}
  installed = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in package.rglob('*') if path.is_file() and path.suffix in ('.py', '.toml')}
  # The two approved production changes: postpone annotations for Python 3.9.
  annotation_modules = ('checkpoints.py', 'session.py')
  addition = b'from __future__ import annotations\n\n'
  for name in annotation_modules:
    module_bytes = (package / name).read_bytes()
    assert module_bytes.count(addition) == 1
    final_source = Path(__file__).resolve().parents[1] / 'src/allin1_infer' / name
    assert module_bytes == final_source.read_bytes(), f'Installed {name} differs from final source'
    installed[name] = hashlib.sha256(module_bytes.replace(addition, b'', 1)).hexdigest()
  # Approved post-baseline rewrites: installed bytes must equal the final source exactly.
  # stems.py: TorchAudio-free input loading (issue #7).
  for name in ('stems.py',):
    module_bytes = (package / name).read_bytes()
    final_source = Path(__file__).resolve().parents[1] / 'src/allin1_infer' / name
    assert module_bytes == final_source.read_bytes(), f'Installed {name} differs from final source'
    installed[name] = expected[name]
  assert installed == expected, 'Production files differ beyond the approved changes'
  assert hashlib.sha256(args.fixture.read_bytes()).hexdigest() == baseline['fixture_sha256'][
    'tests/fixtures/natten_0_17_5_golden.pt']
  assert len(allin1_infer.load_checkpoints()['models']) == 9
  session = allin1_infer.AllInOneSession(device='cpu')
  assert session.status == 'new'
  session.close()
  assert session.status == 'closed'
  torch.set_num_threads(1)
  fixture = torch.load(args.fixture, weights_only=True, map_location='cpu')
  fields = 0
  max_abs = 0.0
  max_relative_rms = 0.0
  for dimension in ('1d', '2d'):
    qk = getattr(na, 'na' + dimension + '_qk')
    av = getattr(na, 'na' + dimension + '_av')
    for case in fixture[dimension]:
      k, d = case['kernel_size'], case['dilation']
      actual = {
        'attn': qk(case['q'], case['k'], k, d, rpb=case['rpb']),
        'attn_norpb': qk(case['q'], case['k'], k, d),
        'out': av(torch.softmax(case['attn'], -1), case['v'], k, d),
      }
      for name, output in actual.items():
        reference = case[name]
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
        delta = (output - reference).double()
        rms = reference.double().square().mean().sqrt()
        assert rms > 0
        max_abs = max(max_abs, float(delta.abs().max()))
        max_relative_rms = max(max_relative_rms, float(delta.square().mean().sqrt() / rms))
        fields += 1
  print(json.dumps({
    "package": str(package), "version": allin1_infer.__version__,
    "torch": torch.__version__, "unchanged_production_files": len(installed) - len(annotation_modules),
    "annotation_only_exceptions": annotation_modules,
    "golden_fields": fields, "max_abs": max_abs,
    "max_relative_rms": max_relative_rms,
  }, indent=2))


if __name__ == '__main__':
  main()
