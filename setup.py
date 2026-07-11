"""
Setup script with post-install hook to install madmom.
This works alongside pyproject.toml for package metadata.
"""
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import sys
from pathlib import Path


def get_version():
    """Read version from __about__.py"""
    version_file = Path(__file__).parent / "src" / "allin1_infer" / "__about__.py"
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Could not find version in __about__.py")


def install_madmom():
    """Install madmom from git if not already installed."""
    try:
        import madmom
        return  # Already installed
    except ImportError:
        pass
    
    # Install madmom from git
    print("\n📦 Installing madmom from git (required dependency)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/CPJKU/madmom", "--quiet"
        ])
        print("✅ madmom installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  Warning: Failed to auto-install madmom: {e}")
        print("Please install it manually with:")
        print("  pip install git+https://github.com/CPJKU/madmom")
        # Don't fail the installation, just warn


class PostInstallCommand(install):
    """Post-installation hook to install madmom from git."""
    def run(self):
        install.run(self)
        install_madmom()


class PostDevelopCommand(develop):
    """Post-installation hook for develop mode."""
    def run(self):
        develop.run(self)
        install_madmom()


# Minimal setup - all metadata comes from pyproject.toml (PEP 621)
# This setup.py is ONLY for the post-install hook to auto-install madmom
# Setuptools will read pyproject.toml for all package metadata
# We need to provide version here since setuptools doesn't read hatch.version
setup(
    version=get_version(),  # Read version from __about__.py
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
)
