# Copyright (c) 2023 Taejun Kim (All-In-One original)
# Copyright (c) 2025 Bo-Yu Chen (Cache management additions)
# SPDX-License-Identifier: MIT

import argparse
import torch
import json

from pathlib import Path
from .analyze import analyze
from .stems import PrecomputedStemProvider
from .stems_input import StemsInput, create_stems_input_from_directory, create_stems_input_from_pattern
from .helpers import print_cache_info, clear_model_cache


def make_parser():
  cwd = Path.cwd()
  parser = argparse.ArgumentParser()
  parser.add_argument('paths', nargs='*', type=Path, default=[], 
                      help='Path to tracks (for single track mode) or omit for stems input mode')
  parser.add_argument('-o', '--out-dir', type=Path, default=cwd / './struct',
                      help='Path to a directory to store analysis results (default: ./struct)')
  parser.add_argument('-v', '--visualize', action='store_true', default=False,
                      help='Save visualizations (default: False)')
  parser.add_argument('--viz-dir', type=str, default=cwd / 'viz',
                      help='Directory to save visualizations if -v is provided (default: ./viz)')
  parser.add_argument('-s', '--sonify', action='store_true', default=False,
                      help='Save sonifications (default: False)')
  parser.add_argument('--sonif-dir', type=str, default=cwd / 'sonif',
                      help='Directory to save sonifications if -s is provided (default: ./sonif)')
  parser.add_argument('-a', '--activ', action='store_true',
                      help='Save frame-level raw activations from sigmoid and softmax (default: False)')
  parser.add_argument('-e', '--embed', action='store_true',
                      help='Save frame-level embeddings (default: False)')
  parser.add_argument('-m', '--model', type=str, default='harmonix-all',
                      help='Name of the pretrained model to use (default: harmonix-all)')
  parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available else cpu)')
  parser.add_argument('-k', '--keep-byproducts', action='store_true',
                      help='Keep demixed audio files and spectrograms (default: False)')
  parser.add_argument('--demix-dir', type=Path, default=cwd / 'demix',
                      help='Path to a directory to store demixed tracks (default: ./demix)')
  parser.add_argument('--spec-dir', type=Path, default=cwd / 'spec',
                      help='Path to a directory to store spectrograms (default: ./spec)')
  parser.add_argument('--overwrite', action='store_true', default=False,
                      help='Overwrite existing files (default: False)')
  parser.add_argument('--no-multiprocess', action='store_true', default=False,
                      help='Disable multiprocessing (default: False)')

  # Performance options
  perf_group = parser.add_argument_group('Performance (opt-in)')
  perf_group.add_argument('--compile-model', action='store_true', default=False,
                          help='EXPERIMENTAL: torch.compile the model(s) (reduce-overhead mode). '
                               '~57s one-time compile cost, ~38%% faster steady-state forward -- '
                               'only worth it for long-lived/batch processing (default: False)')
  perf_group.add_argument('--demucs-overlap', type=float, default=0.25,
                          help='EXPERIMENTAL, accuracy-affecting: overlap fraction for demucs '
                               'separation chunks. Default 0.25 matches demucs\' own default '
                               '(unchanged behavior); other values can shift segment boundaries.')
  perf_group.add_argument('--demucs-fp16', action='store_true', default=False,
                          help='EXPERIMENTAL, accuracy-affecting: run demucs separation under '
                               'fp16 autocast (CUDA only) for faster separation (default: False)')


  # Source separation options
  parser.add_argument('--stems-dict', type=Path, default=None,
                      help='JSON file mapping audio paths to stem directories')
  parser.add_argument('--stems-dir', type=Path, default=None,
                      help='Directory containing pre-computed stems (auto-discovery mode)')
  parser.add_argument('--skip-separation', action='store_true', default=False,
                      help='Skip source separation, use existing stems in demix-dir (default: False)')
  parser.add_argument('--no-demucs', action='store_true', default=False,
                      help='Disable demucs dependency (requires pre-computed stems) (default: False)')
  
  # Stems input mode options
  stems_group = parser.add_argument_group('Direct stems input mode')
  stems_group.add_argument('--stems-bass', type=Path,
                           help='Path to bass stem file (requires all 4 stems)')
  stems_group.add_argument('--stems-drums', type=Path, 
                           help='Path to drums stem file (requires all 4 stems)')
  stems_group.add_argument('--stems-other', type=Path,
                           help='Path to other stem file (requires all 4 stems)')
  stems_group.add_argument('--stems-vocals', type=Path,
                           help='Path to vocals stem file (requires all 4 stems)')
  stems_group.add_argument('--stems-from-dir', type=Path,
                           help='Directory containing bass.wav, drums.wav, other.wav, vocals.wav')
  stems_group.add_argument('--stems-pattern', type=str,
                           help='Pattern for stem files, use with --stems-from-dir (e.g. "song_{stem}.wav")')
  stems_group.add_argument('--stems-id', type=str,
                           help='Identifier for the stem set (defaults to directory name or bass filename)')

  # Cache management options
  cache_group = parser.add_argument_group('Model cache management')
  cache_group.add_argument('--cache-info', action='store_true',
                          help='Show information about cached separation models and exit')
  cache_group.add_argument('--clear-cache', action='store_true',
                          help='Clear all cached separation models and exit')
  cache_group.add_argument('--clear-cache-dry-run', action='store_true',
                          help='Show what would be deleted without actually deleting')

  return parser


def main():
  parser = make_parser()
  args = parser.parse_args()

  # Handle cache management commands (exit after completion)
  if args.cache_info:
    print_cache_info()
    return

  if args.clear_cache or args.clear_cache_dry_run:
    dry_run = args.clear_cache_dry_run
    if dry_run:
      print("\nDry run mode: showing what would be deleted\n")
    else:
      print("\nClearing model cache...\n")

    count = clear_model_cache(dry_run=dry_run)

    if count == 0:
      print("No cached models found")
    else:
      if dry_run:
        print(f"\nWould remove {count} model file(s)")
      else:
        print(f"\nSuccessfully removed {count} model file(s)")
    return

  # Determine input mode: single track or stems
  stems_mode = any([
    args.stems_bass,
    args.stems_drums,
    args.stems_other,
    args.stems_vocals,
    args.stems_from_dir
  ])

  # Validate input arguments
  if not stems_mode and not args.paths:
    raise ValueError('Either provide audio paths or use stems input mode (--stems-bass, etc.)')

  if stems_mode and args.paths:
    raise ValueError('Cannot mix regular audio paths and stems input mode')

  assert args.out_dir is not None, 'Output directory must be specified with --out-dir'

  # Handle stems input mode
  stems_input = None
  if stems_mode:
    if args.stems_from_dir:
      # Create stems input from directory
      if args.stems_pattern:
        stems_input = create_stems_input_from_pattern(
          args.stems_from_dir,
          args.stems_pattern,
          args.stems_id
        )
      else:
        stems_input = create_stems_input_from_directory(
          args.stems_from_dir,
          args.stems_id
        )
    elif all([args.stems_bass, args.stems_drums, args.stems_other, args.stems_vocals]):
      # Create stems input from individual files
      stems_input = StemsInput(
        bass=args.stems_bass,
        drums=args.stems_drums,
        other=args.stems_other,
        vocals=args.stems_vocals,
        identifier=args.stems_id
      )
    else:
      raise ValueError('For stems input mode, either provide --stems-from-dir or all four stem files')

  # Handle stems dictionary
  stems_dict = None
  if args.stems_dict:
    with open(args.stems_dict, 'r') as f:
      stems_dict = json.load(f)

  analyze(
    paths=args.paths if not stems_mode else None,
    stems_input=stems_input,
    out_dir=args.out_dir,
    visualize=args.viz_dir if args.visualize else False,
    sonify=args.sonif_dir if args.sonify else False,
    model=args.model,
    device=args.device,
    include_activations=args.activ,
    include_embeddings=args.embed,
    demix_dir=args.demix_dir,
    spec_dir=args.spec_dir,
    keep_byproducts=args.keep_byproducts,
    overwrite=args.overwrite,
    multiprocess=not args.no_multiprocess,
    stems_dict=stems_dict,
    skip_separation=args.skip_separation,
    compile_model=args.compile_model,
    demucs_overlap=args.demucs_overlap,
    demucs_fp16=args.demucs_fp16,
  )

  print(f'=> Analysis results are successfully saved to {args.out_dir}')


if __name__ == '__main__':
  main()
