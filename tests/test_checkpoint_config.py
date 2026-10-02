"""Checkpoint config parsing and override contracts across supported Python versions.

Reads: package-owned checkpoint facade · pytest temporary files.
"""

import pytest

from allin1_infer.checkpoints import checkpoint_metadata, load_checkpoints


def test_custom_config_and_metadata_override(tmp_path):
  path = tmp_path / 'checkpoints.toml'
  path.write_text(
    '[schema]\nversion = 1\n[models.custom]\nlicense = "original"\n'
    '[[models.custom.artifacts]]\nurl = "https://example.test/model.pth"\n'
    'sha256 = "' + 'a' * 64 + '"\n'
  )
  assert load_checkpoints(path)['models']['custom']['license'] == 'original'
  result = checkpoint_metadata('custom', path=path, overrides={'license': 'override'})
  assert result['license'] == 'override'
  assert result['artifacts'][0]['sha256'] == 'a' * 64
  assert load_checkpoints(path)['models']['custom']['license'] == 'original'


@pytest.mark.parametrize('contents', ['[invalid', '[schema]\nversion = 2\n'])
def test_malformed_checkpoint_config_fails_clearly(tmp_path, contents):
  path = tmp_path / 'broken.toml'
  path.write_text(contents)
  with pytest.raises(ValueError, match='invalid checkpoint config|schema.version=1'):
    load_checkpoints(path)
