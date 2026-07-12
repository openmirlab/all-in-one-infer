"""End-to-end test of allin1_infer.analyze() on the bundled demo track.

Runs the full pipeline (demucs separation -> madmom_infer spectrograms ->
harmonix-all ensemble -> DBN postprocessing) via the shared session-scoped
`analysis_result` fixture (see conftest.py) and sanity-checks the result's
structure: plausible bpm, non-empty ordered beats/downbeats, and a segment
list that covers the track with known labels.
"""

from allin1_infer.config import HARMONIX_LABELS


def test_analyze(analysis_result):
  result = analysis_result

  assert result.bpm > 0

  assert len(result.beats) > 0
  assert result.beats == sorted(result.beats)
  assert len(result.downbeats) > 0
  assert result.downbeats == sorted(result.downbeats)
  # Downbeats are a subset of beats, so there must be fewer of them.
  assert len(result.downbeats) < len(result.beats)
  assert len(result.beat_positions) == len(result.beats)

  assert len(result.segments) > 0
  for segment in result.segments:
    assert segment.start < segment.end
    assert segment.label in HARMONIX_LABELS
