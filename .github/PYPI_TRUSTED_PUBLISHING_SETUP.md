# PyPI Trusted Publishing Setup

## Quick Setup Steps

1. **Go to PyPI Account Settings:**
   - Visit: https://pypi.org/manage/account/publishing/
   - Log in to your PyPI account

2. **Add Trusted Publisher:**
   - Click "Add a new pending publisher"
   - Fill in the following:
     - **PyPI project name:** `all-in-one-infer` ⚠️ **Must match exactly** the `name` field in `pyproject.toml`
     - **Owner:** `openmirlab`
     - **Repository name:** `all-in-one-infer` ⚠️ **Must match exactly** the GitHub repository name
     - **Workflow filename:** `publish.yml`
     - **Environment name:** (leave empty)
   
   **Important:** 
   - PyPI project name (`all-in-one-infer`) now matches repository name (`all-in-one-infer`) ✅
   - Python package name is `allin1_infer` (used in imports: `import allin1_infer`)
   - CLI command is `all-in-one-infer` (matches the PyPI project name, following the
     `demucs-infer` sibling package's convention)

3. **Verify Configuration:**
   - The publisher should appear as "pending" until the next workflow run
   - After the first successful run, it will become "active"

4. **Re-run the Workflow:**
   ```bash
   gh workflow run publish.yml
   ```

## Alternative: Manual Publishing

If you prefer to publish manually using twine:

```bash
# Build the package
python -m pip install --upgrade pip build hatchling torch>=2.0.0
python -m build

# Upload to PyPI (requires PyPI API token)
pip install twine
twine upload dist/*
```

## Current Status

- ✅ Package builds successfully
- ✅ Package validation passes (`twine check`)
- ✅ GitHub Actions workflow is configured
- ❌ PyPI trusted publishing not yet configured
- ❌ Package not yet published to PyPI

## Next Steps

1. Configure PyPI trusted publishing (see steps above)
2. Re-run the workflow: `gh workflow run publish.yml`
3. Verify publication: https://pypi.org/project/all-in-one-infer/
