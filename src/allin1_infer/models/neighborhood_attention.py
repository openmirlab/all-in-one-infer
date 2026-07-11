"""Pure-PyTorch neighborhood attention, drop-in replacement for NATTEN's
legacy functional API (na1d_qk / na1d_av / na2d_qk / na2d_av, NATTEN <=0.17.x).

Why this exists: NATTEN ships source-only builds that must compile against the
installed torch at install time, and NATTEN >=0.20 removed both the legacy
functional API and relative positional bias (RPB) support — which the
pretrained all-in-one checkpoints require. This module reimplements the exact
semantics of NATTEN 0.17.x with plain gather + einsum, so the package installs
anywhere (CPU, CUDA, MPS, any torch >= 2.0) with no compiled extension.

Semantics (must match NATTEN 0.17.x bit-for-bit up to float associativity):
  - Each query attends to exactly `kernel_size` keys at the same residue class
    modulo `dilation`; the window is centered on the query and shifted (not
    zero-padded) at sequence boundaries.
  - RPB is indexed by the relative offset (in dilation steps) between key and
    query, offset by kernel_size - 1 into a table of size 2 * kernel_size - 1.

The model dims in this project are tiny (kernel 5, heads 2, head_dim 12), so
the O(kernel_size) memory overhead of materializing gathered neighbors is
negligible compared to NATTEN's fused kernels' target workloads.
"""

import functools
import torch

__all__ = ['na1d_qk', 'na1d_av', 'na2d_qk', 'na2d_av']


@functools.lru_cache(maxsize=32)
def _na1d_indices(length: int, kernel_size: int, dilation: int, device: str):
  """Neighbor and RPB index tables for one axis.

  Returns:
    idx: (length, kernel_size) int64 — global positions of each query's neighbors.
    rpb_idx: (length, kernel_size) int64 — indices into an RPB table of size
      2 * kernel_size - 1.
  """
  if length < kernel_size * dilation:
    raise ValueError(
      f'Input axis length {length} must be at least kernel_size * dilation '
      f'({kernel_size} * {dilation} = {kernel_size * dilation}).'
    )
  i = torch.arange(length, device=device)
  # Coordinates within the query's residue class modulo `dilation`.
  q_class = i // dilation
  residue = i % dilation
  # Number of positions in each residue class.
  class_len = (length - residue + dilation - 1) // dilation
  # Window start in class coordinates: centered, clamped to stay in bounds.
  start = (q_class - (kernel_size - 1) // 2).clamp(min=0)
  start = torch.minimum(start, class_len - kernel_size)
  j = torch.arange(kernel_size, device=device)
  neighbor_class = start[:, None] + j[None, :]
  idx = neighbor_class * dilation + residue[:, None]
  rpb_idx = neighbor_class - q_class[:, None] + (kernel_size - 1)
  return idx, rpb_idx


def _pair(value):
  if isinstance(value, (tuple, list)):
    return int(value[0]), int(value[1])
  return int(value), int(value)


def na1d_qk(query, key, kernel_size, dilation=1, additional_keys=None,
            is_causal=False, rpb=None):
  """query, key: (B, heads, T, dim) -> attn: (B, heads, T, kernel_size)."""
  if additional_keys is not None or is_causal:
    raise NotImplementedError(
      'additional_keys / is_causal are not supported by the pure-PyTorch '
      'neighborhood attention fallback.'
    )
  kernel_size = int(kernel_size[0]) if isinstance(kernel_size, (tuple, list)) else int(kernel_size)
  dilation = int(dilation[0]) if isinstance(dilation, (tuple, list)) else int(dilation)
  idx, rpb_idx = _na1d_indices(query.shape[2], kernel_size, dilation, str(query.device))
  key_neighbors = key[:, :, idx]  # (B, heads, T, kernel, dim)
  attn = torch.einsum('bhtc,bhtkc->bhtk', query, key_neighbors)
  if rpb is not None:
    attn = attn + rpb[:, rpb_idx]  # (heads, T, kernel) broadcasts over batch
  return attn


def na1d_av(attn, value, kernel_size, dilation=1, additional_values=None,
            is_causal=False):
  """attn: (B, heads, T, kernel_size), value: (B, heads, T, dim) -> (B, heads, T, dim)."""
  if additional_values is not None or is_causal:
    raise NotImplementedError(
      'additional_values / is_causal are not supported by the pure-PyTorch '
      'neighborhood attention fallback.'
    )
  kernel_size = int(kernel_size[0]) if isinstance(kernel_size, (tuple, list)) else int(kernel_size)
  dilation = int(dilation[0]) if isinstance(dilation, (tuple, list)) else int(dilation)
  idx, _ = _na1d_indices(value.shape[2], kernel_size, dilation, str(value.device))
  value_neighbors = value[:, :, idx]  # (B, heads, T, kernel, dim)
  return torch.einsum('bhtk,bhtkc->bhtc', attn, value_neighbors)


def na2d_qk(query, key, kernel_size, dilation=1, additional_keys=None,
            is_causal=False, rpb=None):
  """query, key: (B, heads, H, W, dim) -> attn: (B, heads, H, W, kernel_h * kernel_w)."""
  if additional_keys is not None or is_causal:
    raise NotImplementedError(
      'additional_keys / is_causal are not supported by the pure-PyTorch '
      'neighborhood attention fallback.'
    )
  kernel_h, kernel_w = _pair(kernel_size)
  dilation_h, dilation_w = _pair(dilation)
  B, heads, H, W, dim = query.shape
  idx_h, rpb_idx_h = _na1d_indices(H, kernel_h, dilation_h, str(query.device))
  idx_w, rpb_idx_w = _na1d_indices(W, kernel_w, dilation_w, str(query.device))
  # Broadcast per-axis index tables to a (H, W, kernel_h, kernel_w) grid.
  ih = idx_h[:, None, :, None]  # (H, 1, kernel_h, 1)
  iw = idx_w[None, :, None, :]  # (1, W, 1, kernel_w)
  key_neighbors = key[:, :, ih, iw]  # (B, heads, H, W, kernel_h, kernel_w, dim)
  attn = torch.einsum('bhxyc,bhxyijc->bhxyij', query, key_neighbors)
  if rpb is not None:
    bias = rpb[:, rpb_idx_h[:, None, :, None], rpb_idx_w[None, :, None, :]]
    attn = attn + bias  # (heads, H, W, kernel_h, kernel_w) broadcasts over batch
  return attn.reshape(B, heads, H, W, kernel_h * kernel_w)


def na2d_av(attn, value, kernel_size, dilation=1, additional_values=None,
            is_causal=False):
  """attn: (B, heads, H, W, kernel_h * kernel_w), value: (B, heads, H, W, dim)."""
  if additional_values is not None or is_causal:
    raise NotImplementedError(
      'additional_values / is_causal are not supported by the pure-PyTorch '
      'neighborhood attention fallback.'
    )
  kernel_h, kernel_w = _pair(kernel_size)
  dilation_h, dilation_w = _pair(dilation)
  B, heads, H, W, _ = attn.shape
  idx_h, _ = _na1d_indices(H, kernel_h, dilation_h, str(attn.device))
  idx_w, _ = _na1d_indices(W, kernel_w, dilation_w, str(attn.device))
  ih = idx_h[:, None, :, None]
  iw = idx_w[None, :, None, :]
  value_neighbors = value[:, :, ih, iw]  # (B, heads, H, W, kernel_h, kernel_w, dim)
  attn = attn.reshape(B, heads, H, W, kernel_h, kernel_w)
  return torch.einsum('bhxyij,bhxyijc->bhxyc', attn, value_neighbors)
