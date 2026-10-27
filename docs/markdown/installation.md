# Installation

## Install `uv`

```bash
pip install uv
```

## Create a new virtual environment

```bash
uv venv
```

## Activate the virtual environment

- **Linux/macOS**

```bash
  source .venv/bin/activate
```

- **Windows (PowerShell)**

```powershell
  .venv\Scripts\Activate.ps1
```

## Install the package

```bash
uv pip install .
```

For development mode:

```bash
uv pip install -e .
```
