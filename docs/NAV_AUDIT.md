# Navigation audit evidence

Read-only `nav-audit` review completed 2026-07-15 before lifecycle adoption.
The repository is Python, with model, preprocessing, postprocessing, and
cache domains. Load-bearing modules have file-top purpose/Reads headers and
the new session/checkpoint modules expose narrow interfaces over model and
download details.

## Inference-only boundary

The shipped `src/allin1_infer` tree contains no training loops, dataset
builders/loaders, evaluation metrics, Lightning, W&B, or TensorBoard imports.
The only training-related text is historical documentation and checkpoint
configuration fields retained so OmegaConf can deserialize upstream pretrained
checkpoint payloads; no training runtime or CLI surface is shipped.

## Eight-principle findings

- **1 Information hiding:** model loading, checkpoint metadata, lifecycle, and
  cache concerns each have one owner (`models/loaders.py`, `checkpoints.py`,
  `session.py`, `cache.py`). No new cross-domain value duplication found.
- **2 Interface-first:** `allin1_infer.__init__` is the public barrel; the
  additive `AllInOneSession` facade preserves legacy `analyze()` and advanced
  provider APIs.
- **3 Explicit dependencies:** session dependencies are constructor arguments
  or explicit generic overrides; no shared runtime or global model manager.
- **4 Right grain:** existing compatibility orchestrators remain large
  (`analyze.py` ~401 LOC, `stems.py` ~508 LOC) and are retained to preserve
  behavior. They are pre-existing structural warnings, not expanded by the
  lifecycle facade; future behavior-preserving extraction should split them.
- **5 Framework fit:** package follows conventional Python modules and
  context-manager lifecycle semantics.
- **6 Rearrange, don't rewrite:** prior 3.0.0 deep-module refactor established
  the current domain split; this adoption adds only additive facades and
  package metadata, with no legacy algorithm rewrites.
- **7 Confidence:** no uncertain boundary was encountered; checksum metadata
  is sourced from Hugging Face `X-Linked-Etag` values where available.
- **8 Agent navigability:** all load-bearing modules now have concise headers;
  `__about__.py` and `postprocessing/tempo.py` received missing headers.

The two size findings above are warnings with an explicit compatibility
justification. They do not block the additive API adoption, but remain tracked
for a separate structural refactor commit.
