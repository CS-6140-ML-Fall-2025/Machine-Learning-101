# Python IDE Setup

**Goal:** Write Python code on your computer  
**Time:** 15 minutes  
**Prerequisites:** Willingness to install software

## Quick Setup

### 1. Get VS Code
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download and install
3. Open VS Code

### 2. Add Python Support (if needed)
1. Click Extensions (four squares icon)
2. Search "Python" 
3. Install the Python extension by Microsoft

### 3. Get Python (if needed)
- **Windows:** Download from [python.org](https://python.org)
- **Mac:** Run `brew install python`
- There are other methods to install Python, but these are the most common.

### 4. Virtual Environments (Recommended)

**Why use virtual environments?** Keep your projects organized and prevent package conflicts.

### Create and Use Virtual Environment
```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Mac
.venv\Scripts\activate     # Windows

# Install packages in this environment
pip install numpy pandas scikit-learn

# Your packages are now isolated to this project!
```

## When to Choose Python IDE

**Perfect for you if:**
- You want professional development tools
- You prefer local file control
- You're working on larger projects
- You want debugging and testing features

## Your First Script

1. **Create new file**: `my_first_ml.py`
2. **Download our sample**: [python-sample.py](samples/python-sample.py)
3. **Copy the code** into your file
4. **Install packages**: `pip install package1 package2 ...`
5. **Run your code**: `python my_first_ml.py` from the Terminal (or other preferred methods)
6. **Success!** You've built your first ML model

## Next Steps

✅ **You're ready!** Try the [First ML Example](04-first-ml-example.md)

**Want something easier?** Try [Google Colab](01-google-colab.md) or [JupyterLab](02-jupyter.md).

**Ready for advanced?** Explore the [advanced notebooks](../numpy_pandas_scikit-learn/).