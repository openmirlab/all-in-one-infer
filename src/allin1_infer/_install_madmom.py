"""
Post-install hook to automatically install madmom from git.
This is called after the package is installed.
"""
import subprocess
import sys


def install_madmom():
    """Install madmom from git if not already installed."""
    try:
        import madmom
        # Already installed, skip
        return
    except ImportError:
        pass
    
    # Install madmom from git
    print("Installing madmom from git (required dependency)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/CPJKU/madmom"
        ])
        print("✅ madmom installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to install madmom automatically: {e}")
        print("Please install it manually with: pip install git+https://github.com/CPJKU/madmom")
        raise


if __name__ == "__main__":
    install_madmom()
