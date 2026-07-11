"""Tests for the pure-PyTorch neighborhood attention fallback.

Verifies numerical parity with NATTEN 0.17.5 two ways:
  1. Against golden fixtures recorded from a real natten 0.17.5 install
     (tests/fixtures/natten_0_17_5_golden.pt) — runs everywhere, no natten needed.
  2. Against a live natten install, if one is importable — extra safety net
     for environments that still have natten.
"""

import importlib.util
import itertools
from pathlib import Path

import pytest
import torch

# Import the module file directly: `import allin1fix` pulls in heavy runtime
# deps (torchaudio, madmom) that these unit tests don't need.
_MODULE_PATH = (
  Path(__file__).parent.parent / 'src' / 'allin1fix' / 'models' / 'neighborhood_attention.py'
)
_spec = importlib.util.spec_from_file_location('neighborhood_attention', _MODULE_PATH)
na = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(na)

_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'natten_0_17_5_golden.pt'

try:
  from natten.functional import na1d_qk as ref_na1d_qk  # noqa: F401
  HAS_NATTEN = True
except Exception:
  HAS_NATTEN = False


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


@pytest.mark.skipif(not HAS_NATTEN, reason='natten not installed')
def test_live_parity_1d():
  from natten.functional import na1d_av as ref_av, na1d_qk as ref_qk
  torch.manual_seed(0)
  for k, d in itertools.product([3, 5, 7], [1, 2, 4, 8]):
    T = max(k * d, 60)
    q = torch.randn(2, 2, T, 12, dtype=torch.float64)
    key = torch.randn(2, 2, T, 12, dtype=torch.float64)
    v = torch.randn(2, 2, T, 12, dtype=torch.float64)
    rpb = torch.randn(2, 2 * k - 1, dtype=torch.float64)
    attn_ref = ref_qk(q, key, k, d, rpb=rpb)
    torch.testing.assert_close(na.na1d_qk(q, key, k, d, rpb=rpb), attn_ref)
    probs = torch.softmax(attn_ref, -1)
    torch.testing.assert_close(na.na1d_av(probs, v, k, d), ref_av(probs, v, k, d))


@pytest.mark.skipif(not HAS_NATTEN, reason='natten not installed')
def test_live_parity_2d():
  from natten.functional import na2d_av as ref_av, na2d_qk as ref_qk
  torch.manual_seed(0)
  for k, d in itertools.product([3, 5], [1, 2]):
    X, Y = k * d + 1, 40
    q = torch.randn(2, 2, X, Y, 12, dtype=torch.float64)
    key = torch.randn(2, 2, X, Y, 12, dtype=torch.float64)
    v = torch.randn(2, 2, X, Y, 12, dtype=torch.float64)
    rpb = torch.randn(2, 2 * k - 1, 2 * k - 1, dtype=torch.float64)
    attn_ref = ref_qk(q, key, k, d, rpb=rpb)
    torch.testing.assert_close(na.na2d_qk(q, key, k, d, rpb=rpb), attn_ref)
    probs = torch.softmax(attn_ref, -1)
    torch.testing.assert_close(na.na2d_av(probs, v, k, d), ref_av(probs, v, k, d))
