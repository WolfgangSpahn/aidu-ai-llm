# AIDu Package Deployment Guide

## How pip Works

Python packages are distributed through PyPI in two formats:

* **Source Distribution (sdist)**: a `.tar.gz` archive containing the package source code.
* **Wheel**: a pre-built `.whl` archive containing the installable package.

When a user executes:

```bash
pip install aidu-ai-llm
```

pip downloads the latest wheel from PyPI and installs it into the current Python environment.

For framework development, editable installs can be used:

```bash
pip install -e .
```

This does not copy files into `site-packages`. Instead, Python loads modules directly from the local source tree.

---

## Credentials

PyPI credentials are stored in:

```bash
~/.env
```

Typical entries are:

```bash
PYPI_USERNAME=__token__
PYPI_PASSWORD=pypi-...
```

Load them before deployment:

```bash
source ~/.env
```

---

## Building the Package

Always start from a clean state:

```bash
rm -rf dist build *.egg-info
```

Build source distribution and wheel:

```bash
uv build
```

This creates:

```text
dist/
├── aidu_ai_llm-<version>.tar.gz
└── aidu_ai_llm-<version>-py3-none-any.whl
```

---

## Inspecting the Build

Verify that the source distribution contains the expected files:

```bash
tar -tf dist/*.tar.gz | less
```

Verify that the wheel contains the expected packages:

```bash
python -m zipfile --list dist/*.whl
```

Check for important modules:

```bash
python -m zipfile --list dist/*.whl | grep aidu/ai
```

---

## Testing in a Clean Environment

Create a temporary virtual environment:

```bash
python -m venv /tmp/test-aidu
source /tmp/test-aidu/bin/activate
```

Install the wheel:

```bash
pip install dist/*.whl
```

Verify imports:

```bash
python -c "import aidu.ai.llm"
```

Run the application:

```bash
python -m aidu.ai.llm.demo.app
```

Verify that:

* all imports resolve,
* runtime dependencies are installed,
* packaged assets are available,
* the application starts successfully.

When finished:

```bash
deactivate
rm -rf /tmp/test-aidu
```

---

## Publishing to PyPI

Increase the version number in `pyproject.toml`:

```toml
[project]
version = "0.1.3"
```

Rebuild:

```bash
rm -rf dist build *.egg-info
uv build
```

Upload:

```bash
uv publish
```

or with Twine:

```bash
twine upload dist/*
```

After publishing, verify installation from PyPI:

```bash
python -m venv /tmp/test-pypi
source /tmp/test-pypi/bin/activate

pip install aidu-ai-llm

python -c "import aidu.ai.llm"
```

If this succeeds, the release is complete.

---

## Development Installation

For contributors working from source:

```bash
git clone <repository>
cd aidu-ai-llm

pip install -e .
```

For multi-repository development:

```bash
pip install -e ../aidu-support
pip install -e .
```

or simply:

```bash
uv sync
```

if local editable dependencies are configured through `tool.uv.sources`.
