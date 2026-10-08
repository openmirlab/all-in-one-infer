"""Tests for the pure-PyTorch neighborhood attention implementation.

Verifies numerical parity with NATTEN 0.17.5 against golden fixtures recorded
from a real natten 0.17.5 install (tests/fixtures/neighborhood_attention_golden.pt).
NATTEN itself is no longer a dependency, so these fixtures are the reference.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

# Import the module file directly: `import allin1_infer` pulls in heavy runtime
# deps (demucs_infer, madmom_infer) that these unit tests don't need.
_MODULE_PATH = (
  Path(__file__).parent.parent / 'src' / 'allin1_infer' / 'models' / 'neighborhood_attention.py'
)
_spec = importlib.util.spec_from_file_location('neighborhood_attention', _MODULE_PATH)
na = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(na)

_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'neighborhood_attention_golden.pt'


@pytest.fixture(scope='module')
def golden():
  return torch.load(_FIXTURE_PATH, weights_only=True)


def test_golden_1d(golden):
  for case in golden['1d']:
    k, d = case['kernel_size'], case['dilation']
    attn = na.na1d_qk(case['q'], case['k'], k, d, rpb=case['rpb'])
    torch.testing.assert_close(attn, case['attn'], atol=1e-5, rtol=1e-5)
    attn_norpb = na.na1d_qk(case['q'], case['k'], k, d)
    torch.testing.assert_close(attn_norpb, case['attn_norpb'], atol=1e-5, rtol=1e-5)
    out = na.na1d_av(torch.softmax(case['attn'], -1), case['v'], k, d)
    torch.testing.assert_close(out, case['out'], atol=1e-5, rtol=1e-5)


def test_golden_2d(golden):
  for case in golden['2d']:
    k, d = case['kernel_size'], case['dilation']
    attn = na.na2d_qk(case['q'], case['k'], k, d, rpb=case['rpb'])
    torch.testing.assert_close(attn, case['attn'], atol=1e-5, rtol=1e-5)
    attn_norpb = na.na2d_qk(case['q'], case['k'], k, d)
    torch.testing.assert_close(attn_norpb, case['attn_norpb'], atol=1e-5, rtol=1e-5)
    out = na.na2d_av(torch.softmax(case['attn'], -1), case['v'], k, d)
    torch.testing.assert_close(out, case['out'], atol=1e-5, rtol=1e-5)


def test_attention_weights_sum_to_one_after_softmax():
  # Sanity: every query gets exactly kernel_size neighbors, even at boundaries.
  q = torch.randn(1, 2, 40, 8)
  attn = na.na1d_qk(q, q, 5, 4)
  assert attn.shape == (1, 2, 40, 5)
  probs = torch.softmax(attn, -1)
  torch.testing.assert_close(probs.sum(-1), torch.ones(1, 2, 40))


def test_input_shorter_than_window_raises():
  q = torch.randn(1, 2, 19, 8)
  with pytest.raises(ValueError):
    na.na1d_qk(q, q, 5, 4)  # needs T >= 5 * 4 = 20


def test_unsupported_natten_features_raise():
  q = torch.randn(1, 2, 20, 8)
  with pytest.raises(NotImplementedError):
    na.na1d_qk(q, q, 5, 1, is_causal=True)
  with pytest.raises(NotImplementedError):
    na.na1d_av(torch.randn(1, 2, 20, 5), q, 5, 1, additional_values=q)
