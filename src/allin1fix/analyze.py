import torch

from typing import List, Union, Optional
from tqdm import tqdm
from .demix import demix
from .stems import get_stems, StemProvider, DemucsProvider, PrecomputedStemProvider
from .stems_input import StemsInput, prepare_stems_for_analysis, validate_stems_input
from .spectrogram import extract_spectrograms
from .models import load_pretrained_model
from .visualize import visualize as _visualize
from .sonify import sonify as _sonify
from .helpers import (
  run_inference,
  expand_paths,
  check_paths,
  rmdir_if_empty,
  save_results,
)
from .utils import mkpath, load_result
from .typings import AnalysisResult, PathLike


def analyze(
  paths: Union[PathLike, List[PathLike], StemsInput, List[StemsInput]] = None,
  out_dir: PathLike = None,
  visualize: Union[bool, PathLike] = False,
  sonify: Union[bool, PathLike] = False,
  model: str = 'harmonix-all',
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demix',
  spec_dir: PathLike = './spec',
  keep_byproducts: bool = False,
  overwrite: bool = False,
  multiprocess: bool = True,
  stem_provider: Optional[StemProvider] = None,
  stems_dict: Optional[dict] = None,
  skip_separation: bool = False,
  stems_input: Union[StemsInput, List[StemsInput], List[dict]] = None,
) -> Union[AnalysisResult, List[AnalysisResult]]:
  """
  Analyzes the provided audio files and returns the analysis results.

  Parameters
  ----------
  paths : Union[PathLike, List[PathLike], StemsInput, List[StemsInput]], optional
      List of paths or a single path to the audio files to be analyzed.
      Can also be StemsInput object(s) for direct stems input.
      If None, stems_input parameter must be provided.
  out_dir : PathLike, optional
      Path to the directory where the analysis results will be saved. By default, the results will not be saved.
  visualize : Union[bool, PathLike], optional
      Whether to visualize the analysis results or not. If a path is provided, the visualizations will be saved in that
      directory. Default is False. If True, the visualizations will be saved in './viz'.
  sonify : Union[bool, PathLike], optional
      Whether to sonify the analysis results or not. If a path is provided, the sonifications will be saved in that
      directory. Default is False. If True, the sonifications will be saved in './sonif'.
  model : str, optional
      Name of the pre-trained model to be used for the analysis. Default is 'harmonix-all'. Please refer to the
      documentation for the available models.
  device : str, optional
      Device to be used for computation. Default is 'cuda' if available, otherwise 'cpu'.
  include_activations : bool, optional
      Whether to include activations in the analysis results or not.
  include_embeddings : bool, optional
      Whether to include embeddings in the analysis results or not.
  demix_dir : PathLike, optional
      Path to the directory where the source-separated audio will be saved. Default is './demix'.
  spec_dir : PathLike, optional
      Path to the directory where the spectrograms will be saved. Default is './spec'.
  keep_byproducts : bool, optional
      Whether to keep the source-separated audio and spectrograms or not. Default is False.
  overwrite : bool, optional
      Whether to overwrite the existing analysis results or not. Default is False.
  multiprocess : bool, optional
      Whether to use multiprocessing for spectrogram extraction, visualization, and sonification. Default is True.
  stem_provider : Optional[StemProvider], optional
      Custom stem provider for source separation. If None, uses default DemucsProvider.
  stems_dict : Optional[dict], optional
      Dictionary mapping audio paths to pre-computed stem directories. Alternative to stem_provider.
  skip_separation : bool, optional
      If True, assumes stems are already available and skips separation step. Default is False.
  stems_input : Union[StemsInput, List[StemsInput], List[dict]], optional
      Direct stems input. Provide pre-separated stem files (bass, drums, other, vocals) directly.
      Alternative to paths parameter for stems-only workflow.

  Returns
  -------
  Union[AnalysisResult, List[AnalysisResult]]
      Analysis results for the provided audio files.
  """

  # Handle different input modes
  # TF32 matmul on CUDA: ~25-33% faster forward pass, verified bit-identical
  # beats/downbeats/segments on the harmonix-all ensemble (matmul-heavy attention
  # layers only; conv-heavy demucs separation is unaffected by this flag).
  if 'cuda' in str(device):
    torch.set_float32_matmul_precision('high')

  stems_mode = False

  # Check if we have stems input (either via paths or stems_input parameter)
  if stems_input is not None:
    stems_mode = True
    if paths is not None:
      raise ValueError('Cannot specify both paths and stems_input parameters')
    
    # Convert stems_input to list format
    if not isinstance(stems_input, list):
      stems_input = [stems_input]
    
    # Validate all stems inputs
    validated_stems = [validate_stems_input(s) for s in stems_input]
    
    # Create pseudo-paths for stems (used for result identification)
    paths = [mkpath(f"{s.name}.stems") for s in validated_stems]
    return_list = len(validated_stems) > 1
    
  elif paths is not None:
    # Check if paths contain StemsInput objects
    if isinstance(paths, StemsInput):
      stems_mode = True
      validated_stems = [validate_stems_input(paths)]
      paths = [mkpath(f"{validated_stems[0].name}.stems")]
      return_list = False
    elif isinstance(paths, list) and any(isinstance(p, StemsInput) for p in paths):
      stems_mode = True
      # Mix of StemsInput and regular paths not supported in this version
      if not all(isinstance(p, StemsInput) for p in paths):
        raise ValueError('Cannot mix StemsInput and regular paths in same analysis')
      validated_stems = [validate_stems_input(p) for p in paths]
      paths = [mkpath(f"{s.name}.stems") for s in validated_stems]
      return_list = True
    else:
      # Regular audio file paths
      return_list = True
      if not isinstance(paths, list):
        return_list = False
        paths = [paths]
      paths = [mkpath(p) for p in paths]
      paths = expand_paths(paths)
      check_paths(paths)
  else:
    raise ValueError('Either paths or stems_input must be specified')
  
  demix_dir = mkpath(demix_dir)
  spec_dir = mkpath(spec_dir)

  # Check if the results are already computed.
  if out_dir is None or overwrite:
    todo_paths = paths
    exist_paths = []
  else:
    out_paths = [mkpath(out_dir) / path.with_suffix('.json').name for path in paths]
    todo_paths = [path for path, out_path in zip(paths, out_paths) if not out_path.exists()]
    exist_paths = [out_path for path, out_path in zip(paths, out_paths) if out_path.exists()]

  print(f'=> Found {len(exist_paths)} tracks already analyzed and {len(todo_paths)} tracks to analyze.')
  if exist_paths:
    print(f'=> To re-analyze, please use --overwrite option.')

  # Load the results for the tracks that are already analyzed.
  results = []
  if exist_paths:
    results += [
      load_result(
        exist_path,
        load_activations=include_activations,
        load_embeddings=include_embeddings,
      )
      for exist_path in tqdm(exist_paths, desc='Loading existing results')
    ]

  # Initialize demix_paths and spec_paths as empty lists
  demix_paths = []
  spec_paths = []

  # Analyze the tracks that are not analyzed yet.
  if todo_paths:
    if stems_mode:
      # Direct stems input - prepare stems for analysis
      todo_stems = [validated_stems[i] for i, path in enumerate(paths) if path in todo_paths]
      demix_paths = prepare_stems_for_analysis(
        todo_stems, 
        demix_dir, 
        use_symlinks=True
      )
      print(f'=> Using direct stems input for {len(todo_stems)} track(s).')
    else:
      # Handle source separation based on provided options
      if skip_separation:
        # Assume stems are already in demix_dir with expected structure
        demix_paths = [demix_dir / 'htdemucs' / path.stem for path in todo_paths]
        print(f'=> Skipping source separation, using existing stems.')
      elif stems_dict:
        # Use pre-computed stems from dictionary
        stem_provider = PrecomputedStemProvider(stems_dict)
        demix_paths = get_stems(todo_paths, demix_dir, stem_provider, device)
      elif stem_provider is not None:
        # Use custom stem provider
        demix_paths = get_stems(todo_paths, demix_dir, stem_provider, device)
      else:
        # Default: use HTDemucs (backward compatibility)
        demix_paths = demix(todo_paths, demix_dir, device)

    # Extract spectrograms for the tracks that are not analyzed yet.
    spec_paths = extract_spectrograms(demix_paths, spec_dir, multiprocess)

    # Load the model.
    model = load_pretrained_model(
      model_name=model,
      device=device,
    )

    with torch.no_grad():
      pbar = tqdm(zip(todo_paths, spec_paths), total=len(todo_paths))
      for path, spec_path in pbar:
        pbar.set_description(f'Analyzing {path.name}')

        result = run_inference(
          path=path,
          spec_path=spec_path,
          model=model,
          device=device,
          include_activations=include_activations,
          include_embeddings=include_embeddings,
        )

        # Save the result right after the inference.
        # Checkpointing is always important for this kind of long-running tasks...
        # for my mental health...
        if out_dir is not None:
          save_results(result, out_dir)

        results.append(result)

  # Sort the results by the original order of the tracks.
  results = sorted(results, key=lambda result: paths.index(result.path))

  if visualize:
    if visualize is True:
      visualize = './viz'
    _visualize(results, out_dir=visualize, multiprocess=multiprocess)
    print(f'=> Plots are successfully saved to {visualize}')

  if sonify:
    if sonify is True:
      sonify = './sonif'
    _sonify(results, out_dir=sonify, multiprocess=multiprocess)
    print(f'=> Sonified tracks are successfully saved to {sonify}')

  if not keep_byproducts:
    for path in demix_paths:
      for stem in ['bass', 'drums', 'other', 'vocals']:
        stem_file = path / f'{stem}.wav'
        # Only remove if it's not a symlink (to avoid removing original files)
        if stem_file.exists() and not stem_file.is_symlink():
          stem_file.unlink(missing_ok=True)
        elif stem_file.is_symlink():
          stem_file.unlink(missing_ok=True)  # Remove symlink
      rmdir_if_empty(path)
    
    # Clean up different demix subdirectories
    for subdir in ['htdemucs', 'stems_input', 'custom']:
      subdir_path = demix_dir / subdir
      if subdir_path.exists():
        rmdir_if_empty(subdir_path)
    rmdir_if_empty(demix_dir)

    for path in spec_paths:
      path.unlink(missing_ok=True)
    rmdir_if_empty(spec_dir)

  if not return_list:
    return results[0]
  return results
