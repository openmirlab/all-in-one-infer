"""End-to-end tests of allin1_infer.sonify().

Preserves the original test's intent (sonify a result loaded back from a
saved JSON) but generates that JSON from the shared session-scoped
`analysis_result` fixture instead of a never-committed tests/test.json:
save_results -> AnalysisResult.from_json round trip, then sonify in-memory
and sonify-to-disk.
"""

from allin1_infer import AnalysisResult, sonify
from allin1_infer.helpers import save_results


def _roundtripped_result(analysis_result, tmp_path):
  save_results(analysis_result, tmp_path)
  json_path = tmp_path / analysis_result.path.with_suffix('.json').name
  assert json_path.is_file()
  return AnalysisResult.from_json(json_path)


def test_sonify(analysis_result, tmp_path):
  result = _roundtripped_result(analysis_result, tmp_path)
  y, sr = sonify(result, multiprocess=False)
  assert sr == 44100
  assert y.size > 0


def test_sonify_save(analysis_result, tmp_path):
  result = _roundtripped_result(analysis_result, tmp_path)
  out_dir = tmp_path / 'sonif'
  sonify(result, out_dir=out_dir, multiprocess=False)
  path = result.path
  assert (out_dir / f'{path.stem}.sonif{path.suffix}').is_file()
