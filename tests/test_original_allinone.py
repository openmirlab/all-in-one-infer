#!/usr/bin/env python3
"""
Test the original all-in-one package on the same three problematic tracks.
This will help us understand if our scaling fix introduced new issues
or if the original package also struggles with these tracks.
"""

import sys
import subprocess
from pathlib import Path

def install_original_allinone():
    """Install the original all-in-one package for comparison."""
    
    print("📦 Installing Original All-in-One Package")
    print("=" * 60)
    
    try:
        # Install original all-in-one
        print("🔄 Installing original all-in-one package...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'all-in-one'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully installed original all-in-one")
        else:
            print(f"❌ Failed to install: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error installing original all-in-one: {e}")
        return False

def test_original_on_problematic_tracks():
    """Test original all-in-one on our three problematic tracks."""
    
    print(f"\n🎵 Testing Original All-in-One on Problematic Tracks")
    print("=" * 60)
    
    # Test tracks
    test_tracks = [
        ('assets/audio_samples/Sunflower 60BPM.mp3', 60, 'Simple track - should work'),
        ('assets/audio_samples/Nujabes - Luv(sic) Part 2 feat.Shing02.wav', 85, 'Complex jazz - may have DBN issues'),
        ("assets/audio_samples/NewJeans (뉴진스) 'Super Shy' Official MV.wav", 125, 'K-pop - activation strength test')
    ]
    
    try:
        import allinone  # Original package
        
        results = []
        
        for track_path, expected_bpm, description in test_tracks:
            if not Path(track_path).exists():
                print(f"⚠️ Track not found: {Path(track_path).name}")
                continue
                
            print(f"\n🎧 Testing: {Path(track_path).name}")
            print(f"   Description: {description}")
            print(f"   Expected BPM: ~{expected_bpm}")
            
            try:
                # Analyze with original all-in-one
                result = allinone.analyze(track_path, include_activations=True)
                
                # Extract results
                bpm = getattr(result, 'bpm', None)
                beats = getattr(result, 'beats', [])
                downbeats = getattr(result, 'downbeats', [])
                segments = getattr(result, 'segments', [])
                
                print(f"   📊 Results:")
                print(f"      BPM: {bpm}")
                print(f"      Beats: {len(beats)} detected")
                print(f"      Downbeats: {len(downbeats)} detected")
                print(f"      Segments: {len(segments)} detected")
                
                # Check activations if available
                activations_info = "N/A"
                if hasattr(result, 'activations') and result.activations and 'beat' in result.activations:
                    beat_activations = result.activations['beat']
                    max_activation = beat_activations.max()
                    above_threshold = (beat_activations > 0.19).sum()
                    activations_info = f"Max: {max_activation:.6f}, >0.19: {above_threshold}"
                
                print(f"      Activations: {activations_info}")
                
                # Determine success
                success = (
                    bpm is not None and 
                    isinstance(bpm, (int, float)) and 
                    len(beats) > 0
                )
                
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"      Status: {status}")
                
                results.append({
                    'track': Path(track_path).name,
                    'expected_bpm': expected_bpm,
                    'detected_bpm': bpm,
                    'beats': len(beats),
                    'downbeats': len(downbeats),
                    'segments': len(segments),
                    'success': success,
                    'activations_info': activations_info
                })
                
            except Exception as e:
                print(f"   ❌ Error analyzing {Path(track_path).name}: {e}")
                results.append({
                    'track': Path(track_path).name,
                    'expected_bpm': expected_bpm,
                    'detected_bpm': None,
                    'beats': 0,
                    'downbeats': 0,
                    'segments': 0,
                    'success': False,
                    'activations_info': f"Error: {str(e)}"
                })
        
        # Summary
        print(f"\n📊 ORIGINAL ALL-IN-ONE RESULTS SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"Overall Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
        
        print(f"\nDetailed Results:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['track']}")
            print(f"   Expected BPM: {result['expected_bpm']}, Got: {result['detected_bpm']}")
            print(f"   Beats: {result['beats']}, Downbeats: {result['downbeats']}, Segments: {result['segments']}")
            print(f"   Activations: {result['activations_info']}")
        
        return results
        
    except ImportError:
        print("❌ Original all-in-one package not available")
        print("   This may be due to dependency conflicts or installation issues")
        return None
    except Exception as e:
        print(f"❌ Error testing original package: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def compare_original_vs_fix():
    """Compare original all-in-one results with our fixed version."""
    
    print(f"\n🔍 COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    print("""
This comparison will help us understand:

1. 🎯 BASELINE PERFORMANCE:
   • How does original all-in-one perform on these tracks?
   • Are the issues inherent to the model or caused by our fix?

2. 📊 REGRESSION ANALYSIS:
   • Did our NATTEN fix introduce new problems?
   • Are we solving old issues while creating new ones?

3. 💡 SOLUTION VALIDATION:
   • Should we adjust our scaling factors?
   • Do we need different approaches for different track types?

Expected scenarios:

A) 🟢 Original works on all tracks:
   → Our scaling fix needs adjustment (too aggressive or insufficient)

B) 🟡 Original works on some tracks (same pattern as ours):
   → Issues are inherent to model, our fix is on the right track

C) 🔴 Original fails on all tracks:
   → Model has fundamental limitations, our fix shows improvement

D) 🎯 Mixed pattern:
   → Need track-specific solutions and hybrid approaches
""")

if __name__ == "__main__":
    print("🔬 Original All-in-One vs Our Fix Comparison Test")
    print("=" * 60)
    
    # Try to install and test original
    if install_original_allinone():
        original_results = test_original_on_problematic_tracks()
        compare_original_vs_fix()
        
        if original_results:
            print(f"\n🎯 NEXT STEPS BASED ON RESULTS:")
            print(f"Use this comparison to decide whether to:")
            print(f"• Adjust our scaling factors")
            print(f"• Implement track-type specific scaling") 
            print(f"• Add madmom HMM post-processing")
            print(f"• Pursue hybrid approaches")
    else:
        print(f"\n⚠️ Could not install original all-in-one for comparison")
        print(f"This might be due to:")
        print(f"• Dependency conflicts with all-in-one-infer")
        print(f"• NATTEN version compatibility issues")
        print(f"• Environment conflicts")
        print(f"")
        print(f"💡 Alternative: Test in a separate virtual environment")