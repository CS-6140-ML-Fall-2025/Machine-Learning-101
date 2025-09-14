# Jupyter Notebook Setup

**Goal:** Install Jupyter and run your first ML model locally
**Time needed:** 15-30 minutes
**Prerequisites:** Willingness to install software

## Quick Start

**Already have Python?** Try this:
```bash
pip install jupyter
jupyter notebook
```
Your browser should open with Jupyter running locally!

**New to Python?** Follow the detailed setup below.

## When to Choose Jupyter Notebook

**Perfect for you if:**
- You want control over your files and data
- You need to work offline sometimes
- You prefer local development over cloud services
- You want to learn professional Python practices
- You're working with sensitive or large datasets

**Jupyter's advantages:**
- **Your computer, your rules** - full control over environment
- **Offline capable** - work anywhere, anytime
- **Local file access** - easy integration with your existing files
- **Customizable** - install any packages, themes, extensions
- **Professional workflow** - industry-standard tool for data science
- **HPC cluster access** - connect to university/research computing resources

## Step-by-Step Setup

### Option 1: Quick Install (If you have Python)

```bash
# Install Jupyter
pip install jupyter notebook

# Install essential ML packages
pip install numpy pandas matplotlib scikit-learn seaborn

# Start Jupyter
jupyter notebook
```

### Option 2: Complete Setup (Recommended for beginners)

#### Windows Setup

1. **Install Python**:
   - Go to [python.org/downloads](https://python.org/downloads)
   - Download Python 3.9+ (click the big yellow button)
   - **Important**: Check "Add Python to PATH" during installation
   - Restart your computer after installation

2. **Open Command Prompt**:
   - Press `Windows + R`, type `cmd`, press Enter
   - Or search "Command Prompt" in Start menu

3. **Install Jupyter**:
   ```bash
   pip install jupyter notebook numpy pandas matplotlib scikit-learn seaborn
   ```

4. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

#### Mac Setup

1. **Install Python** (if not already installed):
   ```bash
   # Install Homebrew (if you don't have it)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Python
   brew install python
   ```

2. **Install Jupyter**:
   ```bash
   pip3 install jupyter notebook numpy pandas matplotlib scikit-learn seaborn
   ```

3. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```





### 🚀 High-Performance Computing (HPC) Access

**Advanced Option**: Many universities provide access to powerful GPU clusters through Jupyter interfaces:

**University Research Clusters**:
- **Northeastern University's Research Cluster (RC)** - Access Tesla V100s and A100s via JupyterHub
- **MIT's Engaging Cluster** - Jupyter notebooks on supercomputing hardware  
- **Stanford's Sherlock** - GPU nodes accessible through Jupyter interface
- **UC Berkeley's Savio** - Research computing with notebook access

**How it typically works**:
```bash
# Connect to your university cluster
ssh username@cluster.university.edu

# Load required modules
module load python jupyter

# Start Jupyter with GPU access
jupyter notebook --no-browser --port=8888

# Create SSH tunnel (in another terminal)
ssh -L 8888:localhost:8888 username@cluster.university.edu
```

**HPC Advantages**:
- 🔥 **Massive GPU power** - Tesla V100, A100, H100 GPUs
- 💾 **Large memory** - 100GB+ RAM for huge datasets  
- ⚡ **High-speed storage** - Fast access to research data
- 🕒 **Extended runtimes** - Train models for days or weeks
- 💰 **Free for students** - Access through university accounts

**Perfect for**: Deep learning, computer vision, large-scale NLP, scientific computing

**Check with your university** if they offer research computing access!

## Your First Example

Once Jupyter opens in your browser:

### 1. Create a New Notebook
- Click **"New"** → **"Python 3"**
- Your notebook opens in a new tab
- Save it: **File** → **Save As** → "my_first_ml_model.ipynb"

### 2. Try This Complete Example

**Cell 1: Setup and Data Loading**
```python
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("✅ All libraries imported successfully!")
print(f"📍 Working directory: {os.getcwd()}")
```

**Cell 2: Load and Explore Data**
```python
# Load the iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target_names[data.target]

print("📊 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Species: {df['species'].unique()}")
print("\nFirst 5 rows:")
print(df.head())
```

**Cell 3: Visualize the Data**
```python
# Create a beautiful plot
plt.figure(figsize=(12, 4))

# Plot 1: Sepal measurements
plt.subplot(1, 2, 1)
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
               species_data['sepal width (cm)'], 
               label=species, alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Measurements by Species')
plt.legend()

# Plot 2: Petal measurements
plt.subplot(1, 2, 2)
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['petal length (cm)'], 
               species_data['petal width (cm)'], 
               label=species, alpha=0.7)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Measurements by Species')
plt.legend()

plt.tight_layout()
plt.show()
```

**Cell 4: Train Your Model**
```python
# Prepare data for training
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {accuracy:.1%}")
print("\n📊 Detailed Results:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

**Cell 5: Make New Predictions**
```python
# Predict new flowers
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
    [6.2, 2.8, 4.8, 1.8],  # Likely Virginica
    [5.7, 2.8, 4.1, 1.3]   # Likely Versicolor
])

predictions = model.predict(new_flowers)
probabilities = model.predict_proba(new_flowers)

print("🔮 New Predictions:")
for i, (flower, pred, prob) in enumerate(zip(new_flowers, predictions, probabilities)):
    species = data.target_names[pred]
    confidence = prob.max()
    print(f"Flower {i+1}: {flower} → {species} (confidence: {confidence:.1%})")
```

## Working with Local Files

One of Jupyter's biggest advantages is easy file access:

### Save Your Data
```python
# Save your results to CSV
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'correct': y_test == y_pred
})
results_df.to_csv('model_results.csv', index=False)
print("💾 Results saved to model_results.csv")
```

### Load Your Own Data
```python
# Load data from your computer
# Put a CSV file in the same folder as your notebook
try:
    my_data = pd.read_csv('my_data.csv')
    print(f"✅ Loaded data with shape: {my_data.shape}")
except FileNotFoundError:
    print("📁 Put your CSV file in the same folder as this notebook")
```

### Organize Your Project
```
my_ml_project/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
├── models/
│   └── trained_model.pkl
└── results/
    └── predictions.csv
```

## Essential Jupyter Features

### Keyboard Shortcuts
- `Shift + Enter`: Run cell and move to next
- `Ctrl+Enter` (Windows) / `Cmd+Enter` (Mac): Run cell and stay

💡 **Tip**: Many more keyboard shortcuts are available for faster productivity! Look for keyboard shortcut options in the interface when you're ready to speed up your workflow.

### Magic Commands
```python
# Time how long code takes to run
%time model.fit(X_train, y_train)

# See all variables in memory
%whos

# Display plots inline
%matplotlib inline

# Load code from external file
%load my_functions.py

# Run shell commands
!pip list
!ls -la
```

### Markdown Cells for Documentation
Click a cell and press `M` to convert to Markdown:

```markdown
# My ML Experiment

## Objective
Predict iris species from flower measurements

## Results
- Accuracy: 97.8%
- Best features: petal length and width

## Next Steps
- Try different algorithms
- Collect more data
- Deploy the model
```

## Package Management

### Install New Packages
```python
# Install from within Jupyter
!pip install package_name

# Install specific versions
!pip install scikit-learn==1.3.0

# Install from GitHub
!pip install git+https://github.com/user/repo.git

# List installed packages
!pip list
```



## Common Issues & Solutions

### Jupyter won't start
```bash
# Try different port
jupyter notebook --port=8889

# Check if Python is in PATH
python --version

# Reinstall if needed
pip uninstall jupyter
pip install jupyter
```

### Kernel keeps dying
```bash
# Update packages
pip install --upgrade jupyter notebook

# Check memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### Can't import packages
```python
# Check which Python Jupyter is using
import sys
print(sys.executable)

# Install in correct environment
!{sys.executable} -m pip install package_name
```

### Notebook won't save
- Check file permissions
- Make sure you have write access to the folder
- Try saving with a different name

## Advanced Tips

### Extensions for Better Experience
```bash
# Install useful extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable table of contents
jupyter nbextension enable toc2/main
```

### Export Your Work
```bash
# Convert to HTML
jupyter nbconvert --to html my_notebook.ipynb

# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf my_notebook.ipynb

# Convert to Python script
jupyter nbconvert --to script my_notebook.ipynb
```

### Version Control with Git
```bash
# Initialize git repository
git init

# Add .gitignore for Jupyter
echo ".venv/" >> .gitignore
echo "*.ipynb_checkpoints" >> .gitignore
echo "__pycache__/" >> .gitignore

# Commit your work
git add .
git commit -m "Initial ML notebook"
```

## Next Steps

🎉 **Congratulations!** You've successfully:
- Installed Jupyter on your computer
- Created and run your first local ML notebook
- Learned to manage files and packages
- Built a complete machine learning pipeline

**Ready for more?** Check out:
- **[First ML Example](04-first-ml-example.md)** - More hands-on practice
- **[Next Steps Guide](05-next-steps.md)** - Your learning pathway
- **Sample notebook**: [Download our complete example](samples/jupyter-sample.ipynb)

**Want to explore other environments?**
- **[Google Colab](01-google-colab.md)** - For cloud-based development
- **[Python IDE](03-python.md)** - For professional workflows

You now have a powerful local development environment that will serve you well throughout your ML journey! 🚀