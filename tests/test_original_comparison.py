#!/usr/bin/env python3
"""
Test original all-in-one vs our all-in-one-infer on the same three tracks.
"""

import sys
import subprocess
import json
from pathlib import Path

def test_original_allinone():
    """Test original all-in-one on the problematic tracks."""
    
    print("🔬 Testing Original All-in-One Package")
    print("=" * 60)
    
    # Change to original all-in-one directory
    # NOTE: Update this path to your local all-in-one repository
    original_dir = Path('../all-in-one')  # Relative path, update as needed
    
    # Test tracks - update paths to your test audio files
    # NOTE: These are example paths - update to your actual test audio locations
    test_tracks = [
        {
            'name': 'Sunflower 60BPM.mp3',
            'source': 'assets/Sunflower 60BPM.mp3',  # Update to your test file path
            'expected_bpm': 60,
            'description': 'Simple track - should work'
        },
        {
            'name': 'Nujabes - Luv(sic) Part 2 feat.Shing02.wav', 
            'source': 'assets/Nujabes - Luv(sic) Part 2 feat.Shing02.wav',  # Update to your test file path
            'expected_bpm': 85,
            'description': 'Complex jazz - may have DBN issues'
        },
        {
            'name': "NewJeans (뉴진스) 'Super Shy' Official MV.wav",
            'source': "assets/NewJeans (뉴진스) 'Super Shy' Official MV.wav",  # Update to your test file path
            'expected_bpm': 125,
            'description': 'K-pop - activation strength test'
        }
    ]
    
    results = []
    
    for track in test_tracks:
        print(f"\n🎧 Testing: {track['name']}")
        print(f"   Description: {track['description']}")
        print(f"   Expected BPM: ~{track['expected_bpm']}")
        
        source_path = Path(track['source'])
        if not source_path.exists():
            print(f"   ⚠️ Source track not found: {source_path}")
            continue
        
        # Copy track to original directory
        target_path = original_dir / 'assets' / track['name'] 
        target_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            # Copy the file
            import shutil
            shutil.copy2(source_path, target_path)
            print(f"   📁 Copied to: {target_path}")
            
            # Run analysis in original all-in-one environment
            cmd = [
                'uv', 'run', 'python', '-c',
                f"""
import allin1
result = allin1.analyze('{target_path}', include_activations=True)
print(f"BPM: {{getattr(result, 'bpm', None)}}")
print(f"Beats: {{len(getattr(result, 'beats', []))}}")
print(f"Downbeats: {{len(getattr(result, 'downbeats', []))}}")  
print(f"Segments: {{len(getattr(result, 'segments', []))}}")

# Check activations
if hasattr(result, 'activations') and result.activations and 'beat' in result.activations:
    beat_activations = result.activations['beat']
    max_act = float(beat_activations.max())
    above_thresh = int((beat_activations > 0.19).sum())
    print(f"Max_activation: {{max_act:.6f}}")
    print(f"Above_threshold: {{above_thresh}}")
else:
    print("Max_activation: N/A")
    print("Above_threshold: N/A")
"""
            ]
            
            # Run in original directory
            result = subprocess.run(
                cmd,
                cwd=original_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Parse output
                output_lines = result.stdout.strip().split('\n')
                
                bpm = None
                beats = 0
                downbeats = 0
                segments = 0
                max_activation = None
                above_threshold = 0
                
                for line in output_lines:
                    if line.startswith('BPM: '):
                        bmp_str = line.split('BPM: ')[1]
                        if bmp_str != 'None':
                            bpm = float(bpm_str)
                    elif line.startswith('Beats: '):
                        beats = int(line.split('Beats: ')[1])
                    elif line.startswith('Downbeats: '):
                        downbeats = int(line.split('Downbeats: ')[1])
                    elif line.startswith('Segments: '):
                        segments = int(line.split('Segments: ')[1])
                    elif line.startswith('Max_activation: '):
                        act_str = line.split('Max_activation: ')[1]
                        if act_str != 'N/A':
                            max_activation = float(act_str)
                    elif line.startswith('Above_threshold: '):
                        thresh_str = line.split('Above_threshold: ')[1]
                        if thresh_str != 'N/A':
                            above_threshold = int(thresh_str)
                
                print(f"   📊 Results:")
                print(f"      BPM: {bpm}")
                print(f"      Beats: {beats} detected")
                print(f"      Downbeats: {downbeats} detected")
                print(f"      Segments: {segments} detected")
                print(f"      Max activation: {max_activation}")
                print(f"      Above threshold (>0.19): {above_threshold}")
                
                # Determine success
                success = (
                    bpm is not None and 
                    isinstance(bmp, (int, float)) and 
                    beats > 0
                )
                
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"      Status: {status}")
                
                results.append({
                    'track': track['name'],
                    'expected_bpm': track['expected_bpm'],
                    'detected_bpm': bpm,
                    'beats': beats,
                    'downbeats': downbeats,
                    'segments': segments,
                    'max_activation': max_activation,
                    'above_threshold': above_threshold,
                    'success': success
                })
                
            else:
                print(f"   ❌ Error running analysis:")
                print(f"      stdout: {result.stdout}")
                print(f"      stderr: {result.stderr}")
                
                results.append({
                    'track': track['name'],
                    'expected_bpm': track['expected_bpm'],
                    'detected_bpm': None,
                    'beats': 0,
                    'downbeats': 0,
                    'segments': 0,
                    'max_activation': None,
                    'above_threshold': 0,
                    'success': False,
                    'error': result.stderr
                })
                
        except Exception as e:
            print(f"   ❌ Error processing {track['name']}: {e}")
            results.append({
                'track': track['name'],
                'expected_bpm': track['expected_bpm'],
                'detected_bpm': None,
                'beats': 0,
                'downbeats': 0,
                'segments': 0,
                'max_activation': None,
                'above_threshold': 0,
                'success': False,
                'error': str(e)
            })
    
    return results

def compare_results():
    """Compare original vs our fix results."""
    
    print(f"\n📊 ORIGINAL ALL-IN-ONE RESULTS SUMMARY")
    print("=" * 60)
    
    original_results = test_original_allinone()
    
    if not original_results:
        print("❌ No results obtained from original package")
        return
    
    successful = sum(1 for r in original_results if r['success'])
    total = len(original_results)
    
    print(f"Overall Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    print(f"\nDetailed Results:")
    for result in original_results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['track']}")
        print(f"   Expected BPM: {result['expected_bpm']}, Got: {result['detected_bpm']}")
        print(f"   Beats: {result['beats']}, Segments: {result['segments']}")
        if result['max_activation'] is not None:
            print(f"   Max activation: {result['max_activation']:.6f}, Above threshold: {result['above_threshold']}")
        else:
            print(f"   Activations: N/A")
        if 'error' in result:
            print(f"   Error: {result['error']}")
    
    # Compare with our known fix results
    print(f"\n🔍 COMPARISON WITH ALL-IN-ONE-FIX")
    print("=" * 60)
    
    our_results = {
        'Sunflower 60BPM.mp3': {'bpm': 60, 'beats': 272, 'max_activation': 1.0, 'success': True},
        'Nujabes - Luv(sic) Part 2 feat.Shing02.wav': {'bmp': None, 'beats': 0, 'max_activation': 1.0, 'success': False},
        "NewJeans (뉴진스) 'Super Shy' Official MV.wav": {'bpm': None, 'beats': 0, 'max_activation': 0.114, 'success': False}
    }
    
    print(f"Track-by-track comparison:")
    for orig_result in original_results:
        track = orig_result['track']
        our_result = our_results.get(track, {})
        
        print(f"\n🎵 {track}:")
        print(f"   Original:     BPM={orig_result['detected_bpm']}, Beats={orig_result['beats']}, Max_act={orig_result['max_activation']}")
        print(f"   Our Fix:      BPM={our_result.get('bpm', 'N/A')}, Beats={our_result.get('beats', 'N/A')}, Max_act={our_result.get('max_activation', 'N/A')}")
        
        # Analysis
        if orig_result['success'] and our_result.get('success', False):
            print(f"   📊 Both working - compare quality")
        elif orig_result['success'] and not our_result.get('success', False):
            print(f"   ⚠️ Original works, our fix broke it - need adjustment")
        elif not orig_result['success'] and our_result.get('success', False):
            print(f"   ✅ Our fix improved upon original failure")
        else:
            print(f"   ❌ Both failing - inherent model limitation")
    
    return original_results

if __name__ == "__main__":
    print("🔬 Original All-in-One vs All-in-One-Fix Comparison")
    print("=" * 60)
    
    original_results = compare_results()
    
    if original_results:
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"This comparison reveals whether:")
        print(f"• Our scaling fix introduced regressions")
        print(f"• Issues are inherent to the original model")
        print(f"• Our fix provides improvements over baseline")
        print(f"• Different tracks need different approaches")
    else:
        print(f"\n❌ Could not complete comparison")
        print(f"Check original all-in-one environment setup")