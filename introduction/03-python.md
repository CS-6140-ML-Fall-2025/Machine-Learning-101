# Python IDE Setup

**Goal:** Write Python code on your computer  
**Time:** 10 minutes  

## Quick Setup

### 1. Get VS Code
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download and install
3. Open VS Code

### 2. Add Python Support
1. Click Extensions (four squares icon)
2. Search "Python" 
3. Install the Python extension by Microsoft

### 3. Get Python
- **Windows:** Download from [python.org](https://python.org)
- **Mac:** Run `brew install python`

### 4. Try It Out
1. Create new file: `my_first_ml.py`
2. Copy this code:

```python
# Simple ML example
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Make prediction
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
print(f"Predicted species: {data.target_names[prediction[0]]}")
```

3. Install packages: `pip install scikit-learn`
4. Run your code: Press F5

## Terminal Basics

**Why learn terminal?** You'll need it to install packages and run Python scripts.

### Opening Terminal
- **Windows:** Search "Command Prompt" or "PowerShell"
- **Mac:** Search "Terminal" or press `Cmd+Space` and type "terminal"

### Essential Commands
```bash
# See where you are
pwd

# List files in current folder
ls          # Mac
dir         # Windows

# Change to a folder
cd my_folder

# Go back one folder
cd ..

# Go to home folder
cd ~        # Mac
cd %USERPROFILE%  # Windows

# Install Python packages
pip install package_name

# Run Python files
python my_script.py
```

### Quick Example
```bash
# Navigate to your project
cd Desktop
cd my_ml_project

# Check what's here
ls

# Install what you need
pip install scikit-learn

# Run your code
python my_first_ml.py
```

💡 **Tip:** Type the first few letters of a folder name and press Tab to auto-complete!

## Virtual Environments (Recommended)

**Why use virtual environments?** Keep your projects organized and prevent package conflicts.

### Create and Use Virtual Environment
```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Mac
.venv\Scripts\activate     # Windows

# Install packages in this environment
pip install scikit-learn numpy pandas matplotlib

# Your packages are now isolated to this project!
```

### When to Use
- **New project:** Always create a new virtual environment
- **Different Python versions:** Each project can use different package versions
- **Clean installs:** No conflicts between projects

💡 **Tip:** Activate your virtual environment every time you work on your project!

## Next Steps

✅ **You're ready!** Try our [complete ML example](04-first-ml-example.md)

**Want something easier?** Try [Google Colab](01-google-colab.md) or [Jupyter](02-jupyter.md) instead.