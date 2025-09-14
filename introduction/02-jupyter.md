# JupyterLab Setup

**Goal:** Install JupyterLab and run your first ML model locally  
**Time:** 15 minutes  
**Prerequisites:** Willingness to install software

## Quick Setup

```bash
pip install jupyterlab
pip install numpy pandas matplotlib scikit-learn # and other necessary packages
jupyter lab
```

## When to Choose JupyterLab

**Perfect for you if:**
- You want to work offline
- You prefer local file control
- You're working with sensitive data
- You want modern, professional ML tools

**JupyterLab advantages:**
- **Modern interface** - tabbed notebooks, file browser
- **Better for ML** - variable inspector, debugger
- **More powerful** - extensions, themes, layouts

## Familiarity with Google Colab

- Other than the aspects highlighted above, Jupyter Lab is highly similar to Google Colab.
- You can use JupyterLab on your local machine, but you'll need to install a lot of dependencies.

## Your First Notebook

Once JupyterLab opens in your browser:

1. **Click "Python 3" under "Notebook"**
2. **Download our sample**: [jupyter-sample.ipynb](samples/jupyter-sample.ipynb)
3. **Drag and drop** the file into JupyterLab
4. **Double-click** to open and run the cells
5. **Success!** You've built your first ML model

**Note:** JupyterLab can also open classic Jupyter notebooks (.ipynb files) - they work the same way!