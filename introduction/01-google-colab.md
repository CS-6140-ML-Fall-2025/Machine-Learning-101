# Google Colab Setup

**Goal:** Get your first ML model running in 5 minutes
**Prerequisites:** Google account (Gmail, etc.)

## Quick Start (5 Minutes to Success!)

1. **Go to [colab.research.google.com](https://colab.research.google.com)**
2. **Sign in** with your Google account
3. **Click "New notebook"** (big orange button)
4. **Copy and paste this code** into the first cell:

```python
# Your first ML model in Colab!
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"🎉 Your model is {accuracy:.1%} accurate!")

# Make a prediction
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
flower_names = ['Setosa', 'Versicolor', 'Virginica']
print(f"🌸 This flower is likely: {flower_names[prediction[0]]}")
```

5. **Press Shift + Enter** to run the code
6. **Celebrate!** You just built your first ML model 🎉

## When to Choose Google Colab

**Perfect for you if:**
- You want to start ML immediately (no installation needed)
- You need free GPU power for training models
- You want to easily share your work with others
- You're comfortable working online
- You want to focus on learning, not setup

**Colab's superpowers:**
- **Zero setup** - works in any web browser
- **Free GPUs** - train models 10-100x faster than your laptop
- **Easy sharing** - send a link, others can run your code instantly
- **Pre-installed packages** - most ML libraries already available
- **Google Drive integration** - save your work automatically

## Step-by-Step Setup

### 1. Access Colab
- Go to [colab.research.google.com](https://colab.research.google.com)
- Sign in with any Google account (Gmail, Google Workspace, etc.)
- No downloads or installations needed!

### 2. Create Your First Notebook
- Click **"New notebook"** or **File → New notebook**
- Your notebook opens immediately - you're ready to code!
- Colab automatically saves to your Google Drive

### 3. Understanding the Interface
- **Code cells**: Where you write Python code (gray background)
- **Text cells**: For explanations and notes (white background)
- **Run button**: ▶️ Click this or press `Shift + Enter` to run code
- **+ Code/+ Text**: Add new cells below the current one

### 4. Essential Operations
```python
# This is a code cell - try running it!
print("Hello, Machine Learning! 🤖")

# Variables persist between cells
my_name = "Future ML Expert"
print(f"Welcome, {my_name}!")
```

## Your First Example

Try this complete ML example - copy each code block into separate cells:

### Cell 1: Import Libraries
```python
# Import the tools we need
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

print("✅ Libraries imported successfully!")
```

### Cell 2: Load and Explore Data
```python
# Load the famous iris flower dataset
data = load_iris()
print(f"📊 Dataset shape: {data.data.shape}")
print(f"🌸 Flower types: {data.target_names}")
print(f"📏 Features: {data.feature_names}")

# Show first few samples
print("\nFirst 5 flowers:")
for i in range(5):
    print(f"Flower {i+1}: {data.data[i]} → {data.target_names[data.target[i]]}")
```

### Cell 3: Train Your Model
```python
# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f"🎯 Model accuracy: {accuracy:.1%}")
print("🎉 Your model is trained and ready!")
```

### Cell 4: Make Predictions
```python
# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Sepal length, width, petal length, width
prediction = model.predict(new_flower)
confidence = model.predict_proba(new_flower).max()

flower_name = data.target_names[prediction[0]]
print(f"🔮 Prediction: {flower_name}")
print(f"🎯 Confidence: {confidence:.1%}")
```

## Collaboration & Sharing

### Share Your Notebook
1. **Click "Share"** (top right)
2. **Set permissions**: 
   - "Viewer" - others can see but not edit
   - "Commenter" - others can add comments
   - "Editor" - others can modify your code
3. **Copy the link** and send it to anyone!

### Save Your Work
- **Auto-save**: Colab saves to Google Drive automatically
- **Manual save**: Use keyboard shortcut or File → Save
- **Download**: File → Download → Download .ipynb

### Work with Files
```python
# Upload files from your computer
from google.colab import files
uploaded = files.upload()  # Click "Choose Files" button

# Mount Google Drive for larger files
from google.colab import drive
drive.mount('/content/drive')

# Access your Drive files
import os
os.listdir('/content/drive/MyDrive')
```

## Common Issues & Solutions

### "Runtime disconnected"
- **Cause**: Idle for 90+ minutes or running for 12+ hours
- **Solution**: Just click "Reconnect" - your code is saved!
- **Prevention**: Keep a cell running or upgrade to Colab Pro

### "Module not found" error
```python
# Install missing packages
!pip install package_name

# Example: Install additional ML tools
!pip install seaborn plotly
```

### Code runs slowly
```python
# Enable free GPU acceleration
# Runtime → Change runtime type → Hardware accelerator → GPU

# Check if GPU is available
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

### Can't find uploaded files
```python
# List all files in current directory
import os
print("Files in current directory:")
for file in os.listdir():
    print(f"  📄 {file}")
```

## Pro Tips for Beginners

### 1. Use Text Cells for Notes
Click **"+ Text"** to add explanations:
```markdown
# My ML Experiment
Today I learned how to:
- Load data with sklearn
- Train a Random Forest model
- Make predictions on new data

**Next steps**: Try different algorithms!
```

### 2. Keyboard Shortcuts
- `Shift + Enter`: Run cell and move to next
- `Ctrl+Enter` (Windows) / `Cmd+Enter` (Mac): Run cell and stay

💡 **Tip**: Many more keyboard shortcuts are available for faster productivity! Look for keyboard shortcut options in the interface when you're ready to speed up your workflow.

### 3. Get Help
```python
# Get help on any function
help(RandomForestClassifier)

# Or use ? for quick help
RandomForestClassifier?
```

### 4. Visualize Your Data
```python
# Create beautiful plots
plt.figure(figsize=(10, 6))
plt.scatter(data.data[:, 0], data.data[:, 1], c=data.target, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Flowers by Species')
plt.colorbar()
plt.show()
```

## Next Steps

🎉 **Congratulations!** You've successfully:
- Created your first Colab notebook
- Trained a machine learning model
- Made predictions on new data
- Learned to share your work

**Ready for more?** Check out:
- **[First ML Example](04-first-ml-example.md)** - More hands-on practice
- **[Next Steps Guide](05-next-steps.md)** - Your learning pathway
- **Sample notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CS-6140-ML-Fall-2025/TA-Classes/blob/main/introduction/samples/colab-sample.ipynb)

**Want to explore other environments?**
- **[Jupyter Notebook](02-jupyter.md)** - For local development
- **[Python IDE](03-python.md)** - For professional workflows

Remember: The ML concepts you learn in Colab transfer completely to any other environment. You're building real, valuable skills! 🚀