# Google Colab Reference Guide

## What is Google Colab?
Google Colaboratory is a free cloud-based Jupyter notebook environment that requires no setup and runs entirely in your browser.

## Getting Started

### Essential Buttons & Shortcuts

- Run cell: Click ▶️ or press `Shift + Enter`
- New code cell: `+ Code` button or `Ctrl/Cmd + M + B`
- New text cell: `+ Text` button
- Stop execution: Click ⬛ (square stop button)


### Accessing Colab
```
https://colab.research.google.com
```
- Sign in with your Google account
- Create new notebook: File → New notebook
- Open existing: File → Open notebook

## Interface Basics

### Cell Types
```python
# CODE CELL - Run Python code
print("This is a code cell")
x = 10
```

```markdown
# TEXT CELL (Markdown)
# Heading 1
## Heading 2
**Bold text**
*Italic text*
- Bullet points
1. Numbered lists
```

### Running Cells
- **Run cell**: `Shift + Enter` (runs and moves to next)
- **Run in place**: `Ctrl/Cmd + Enter` (stays in same cell)
- **Run all**: Runtime → Run all
- **Run before**: Runtime → Run before
- **Run after**: Runtime → Run after

## Keyboard Shortcuts

### More Shortcuts
```
Ctrl/Cmd + M + H    : See all keyboard shortcuts
Ctrl/Cmd + M + B    : Insert cell below
Ctrl/Cmd + M + A    : Insert cell above
Ctrl/Cmd + M + D    : Delete cell
Ctrl/Cmd + M + M    : Convert to text cell
Ctrl/Cmd + M + Y    : Convert to code cell
Ctrl/Cmd + S        : Save notebook
Ctrl/Cmd + F        : Find and replace
Ctrl/Cmd + Z        : Undo
```

## Runtime Management

### Checking Runtime Info
```python
# Check Python version
import sys
print(f"Python version: {sys.version}")

# Check available RAM
!cat /proc/meminfo | grep MemTotal

# Check GPU (if enabled)
!nvidia-smi
```

### Changing Runtime Type
1. Runtime → Change runtime type
2. Hardware accelerator options:
   - None (CPU only)
   - GPU (Tesla T4 or similar)
   - TPU (Tensor Processing Unit)

```python
# Verify GPU is available
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Or with PyTorch
import torch
print("GPU Available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name: ", torch.cuda.get_device_name(0))
```

## Package Management

### Installing Packages
```python
# Install a package (note the !)
!pip install numpy pandas matplotlib

# Install specific version
!pip install numpy==1.21.0

# Install from GitHub
!pip install git+https://github.com/username/repository.git

# Show installed packages
!pip list

# Check specific package version
!pip show numpy
```

### Importing Packages
```python
# Standard imports for ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check if import successful
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

## File Management

### Google Drive Integration
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your files
import os
os.listdir('/content/drive/MyDrive')

# Read a file from Drive
df = pd.read_csv('/content/drive/MyDrive/data.csv')
```

### Upload Files Directly
```python
# Upload files from your computer
from google.colab import files
uploaded = files.upload()

# View uploaded files
for filename in uploaded.keys():
    print(f'Uploaded: {filename} ({len(uploaded[filename])} bytes)')
```

### Download Files
```python
# Download a file to your computer
from google.colab import files
files.download('results.csv')
```

### Working with URLs
```python
# Download file from URL
!wget https://example.com/data.csv

# Or using gdown for Google Drive links
!pip install gdown
!gdown https://drive.google.com/uc?id=FILE_ID
```

## System Commands

### Shell Commands (use !)
```python
# List files
!ls -la

# Check current directory
!pwd

# Change directory (note: use %cd for persistent change)
%cd /content/drive/MyDrive

# Create directory
!mkdir new_folder

# Remove file
!rm unwanted_file.txt

# Check disk usage
!df -h
```

## Common Issues
- **"Runtime disconnected"** → Runtime restarts after 90 min idle or 12 hours max
- **"Module not found"** → Install it with `!pip install package_name`
- **Variables undefined** → Run cells in order from top to bottom

## Limitations to Remember

1. **Session Duration**: Maximum 12 hours
2. **Idle Timeout**: ~90 minutes of inactivity
3. **GPU Limits**: Usage quotas apply (varies by usage)
4. **Storage**: ~100GB temporary, clears on disconnect
5. **RAM**: ~12GB (25GB with Colab Pro)
6. **Internet Access**: Required for all operations

## Quick Start Template

```python
# Standard ML Project Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Display settings
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')
%matplotlib inline

# Mount Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

print("Setup complete! 🚀")
```