"""End-to-end tests of allin1_infer.visualize().

Uses the shared session-scoped `analysis_result` fixture (one analyze() run
per session, see conftest.py) instead of re-running analyze per test as the
original did. Checks both the in-memory figure return and the save-to-disk
path (a {track stem}.pdf in out_dir); MPLBACKEND=Agg is forced in conftest so
no GUI window is ever opened.
"""

from allin1_infer import visualize


def test_visualize(analysis_result):
  fig = visualize(analysis_result, multiprocess=False)
  assert fig is not None


def test_visualize_save(analysis_result, tmp_path):
  out_dir = tmp_path / 'viz'
  visualize(analysis_result, out_dir=out_dir, multiprocess=False)
  assert (out_dir / f'{analysis_result.path.stem}.pdf').is_file()
