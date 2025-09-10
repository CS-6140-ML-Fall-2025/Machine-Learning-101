# Jupyter Notebook Quick Reference

## What is Jupyter?
Interactive notebook environment that runs on your computer (unlike Colab which runs in the cloud).

## Installation (One Time)
```bash
# Install via pip
pip install jupyter notebook

# Or install Jupyter Lab (newer interface)
pip install jupyterlab
```

## Starting Jupyter
```bash
# Start Jupyter Notebook
jupyter notebook

# Or start Jupyter Lab
jupyter lab

# This opens in your browser at http://localhost:8888
```

## Key Concepts
- **Cell**: A box containing code or text
- **Kernel**: The "engine" running your Python code
- **Notebook file**: Saved as `.ipynb` files

## Essential Operations
- **Run cell**: `Shift + Enter`
- **Run cell & stay**: `Ctrl + Enter`
- **Insert cell above**: `A` (in command mode)
- **Insert cell below**: `B` (in command mode)
- **Delete cell**: `DD` (press D twice)
- **Switch to Markdown**: `M` (in command mode)
- **Switch to Code**: `Y` (in command mode)

## Kernel Management
```python
# Restart kernel (clears all variables)
# Kernel menu → Restart

# Common pattern after installing new packages:
# 1. Install package
# 2. Restart kernel
# 3. Import package
```

## Check Your Environment
```python
# Where is Jupyter running?
import os
print(os.getcwd())

# What packages are installed?
!pip list

# Python version
import sys
print(sys.version)
```

## Jupyter vs Colab

| Feature | Jupyter | Colab |
|---------|---------|--------|
| Runs on | Your computer | Google's servers |
| Files saved | Locally | Google Drive |
| Free GPU | ❌ | ✅ |
| Installation | Required | Not needed |
| Internet | Optional | Required |

## Common Issues
- **"Kernel died"** → Restart kernel from Kernel menu
- **Can't find my files** → Check your working directory with `os.getcwd()`
- **Package not found** → Install in current environment with `!pip install`