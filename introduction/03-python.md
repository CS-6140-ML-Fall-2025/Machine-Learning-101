# Python IDE for Machine Learning Beginners

**What you'll learn:**
- How to set up a professional Python development environment
- The advantages of IDEs for serious ML development
- How to transition from notebooks to production-ready code

**Time needed:** 45 minutes
**Prerequisites:** Comfort with installing software and learning new tools

## Quick Start

**Already comfortable with coding?** Try this:
1. Install [VS Code](https://code.visualstudio.com/) or [PyCharm Community](https://www.jetbrains.com/pycharm/download/)
2. Install Python extension/plugin
3. Create a new Python file: `my_first_ml.py`
4. Copy our sample code and run it!

**New to IDEs?** Follow the detailed setup below.

## When to Choose a Python IDE

**Perfect for you if:**
- You want to learn professional development practices
- You plan to build production ML applications
- You need advanced debugging and code organization tools
- You want to integrate ML into larger software projects
- You're comfortable with more complex setup

**IDE advantages:**
- **Professional workflow** - industry-standard development practices
- **Advanced debugging** - step through code line by line
- **Code organization** - manage large projects with multiple files
- **Version control integration** - built-in Git support
- **Intelligent code completion** - AI-powered suggestions
- **Testing frameworks** - automated testing for your ML code

## Recommended IDEs

### VS Code (Recommended for beginners)
- **Free and lightweight**
- **Excellent Python support**
- **Great for both notebooks and scripts**
- **Huge extension ecosystem**

### PyCharm Community (Recommended for serious development)
- **Built specifically for Python**
- **Powerful debugging tools**
- **Excellent code analysis**
- **Professional project management**

### Other Options
- **Spyder**: Scientific Python IDE (similar to MATLAB/RStudio)
- **Sublime Text**: Fast and customizable
- **Vim/Neovim**: For advanced users who love keyboard shortcuts

## Step-by-Step Setup: VS Code

### 1. Install VS Code
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download for your operating system
3. Install with default settings
4. Launch VS Code

### 2. Install Python Extension
1. Click the **Extensions** icon (four squares) in the sidebar
2. Search for "Python"
3. Install the **Python extension by Microsoft**
4. Restart VS Code

### 3. Install Python (if needed)
```bash
# Windows: Download from python.org
# Mac: 
brew install python

# Linux (Ubuntu/Debian):
sudo apt update
sudo apt install python3 python3-pip
```

### 4. Set Up Your First ML Project
1. **Create a new folder**: `File → Open Folder → Create New Folder → "my_ml_project"`
2. **Create a Python file**: `File → New File → Save as "iris_classifier.py"`
3. **Select Python interpreter**: `Ctrl/Cmd + Shift + P → "Python: Select Interpreter"`

## Your First Example

Create a new file called `iris_classifier.py` and add this code:

```python
"""
Iris Species Classifier
A complete machine learning example using scikit-learn
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_and_explore_data():
    """Load the iris dataset and display basic information."""
    print("🌸 Loading Iris Dataset...")
    
    # Load data
    data = load_iris()
    
    # Create DataFrame for easier handling
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target_names[data.target]
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🏷️  Species: {list(data.target_names)}")
    print(f"📏 Features: {list(data.feature_names)}")
    
    return data, df

def visualize_data(df, data):
    """Create visualizations of the iris dataset."""
    print("📈 Creating visualizations...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Iris Dataset Analysis', fontsize=16)
    
    # Plot 1: Sepal Length vs Width
    axes[0, 0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], 
                      c=data.target, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('Sepal Length (cm)')
    axes[0, 0].set_ylabel('Sepal Width (cm)')
    axes[0, 0].set_title('Sepal Measurements')
    
    # Plot 2: Petal Length vs Width
    axes[0, 1].scatter(df['petal length (cm)'], df['petal width (cm)'], 
                      c=data.target, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('Petal Length (cm)')
    axes[0, 1].set_ylabel('Petal Width (cm)')
    axes[0, 1].set_title('Petal Measurements')
    
    # Plot 3: Feature distributions
    df.hist(bins=20, ax=axes[1, 0], alpha=0.7)
    axes[1, 0].set_title('Feature Distributions')
    
    # Plot 4: Species counts
    species_counts = df['species'].value_counts()
    axes[1, 1].bar(species_counts.index, species_counts.values)
    axes[1, 1].set_title('Species Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def train_model(data):
    """Train a Random Forest classifier on the iris dataset."""
    print("🤖 Training machine learning model...")
    
    # Prepare data
    X = data.data
    y = data.target
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=3
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {accuracy:.1%}")
    
    return model, X_test, y_test, y_pred

def evaluate_model(model, data, y_test, y_pred):
    """Evaluate the trained model and display results."""
    print("📊 Evaluating model performance...")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, 
                yticklabels=data.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': data.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

def make_predictions(model, data):
    """Make predictions on new flower measurements."""
    print("🔮 Making predictions on new flowers...")
    
    # Example new flowers
    new_flowers = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
        [6.2, 2.8, 4.8, 1.8],  # Likely Virginica
        [5.7, 2.8, 4.1, 1.3]   # Likely Versicolor
    ])
    
    # Make predictions
    predictions = model.predict(new_flowers)
    probabilities = model.predict_proba(new_flowers)
    
    print("\nPrediction Results:")
    for i, (flower, pred, prob) in enumerate(zip(new_flowers, predictions, probabilities)):
        species = data.target_names[pred]
        confidence = prob.max()
        print(f"  Flower {i+1}: {flower}")
        print(f"    → Predicted: {species}")
        print(f"    → Confidence: {confidence:.1%}")
        print()

def main():
    """Main function to run the complete ML pipeline."""
    print("🚀 Starting Iris Classification Project")
    print("=" * 50)
    
    try:
        # Load and explore data
        data, df = load_and_explore_data()
        
        # Visualize data
        visualize_data(df, data)
        
        # Train model
        model, X_test, y_test, y_pred = train_model(data)
        
        # Evaluate model
        evaluate_model(model, data, y_test, y_pred)
        
        # Make new predictions
        make_predictions(model, data)
        
        print("✅ Project completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure all required packages are installed:")
        print("pip install numpy pandas matplotlib scikit-learn seaborn")

if __name__ == "__main__":
    main()
```

### Run Your Code
1. **Install required packages**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn
   ```

2. **Run the script**:
   - Press `F5` in VS Code
   - Or use terminal: `python iris_classifier.py`

## Step-by-Step Setup: PyCharm

### 1. Install PyCharm Community
1. Go to [jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download/)
2. Download **Community Edition** (free)
3. Install with default settings
4. Launch PyCharm

### 2. Create Your First Project
1. **New Project** → **Pure Python**
2. **Location**: Choose a folder for your project
3. **Python Interpreter**: Select your Python installation
4. **Create**

### 3. Set Up Virtual Environment
1. **File** → **Settings** → **Project** → **Python Interpreter**
2. **Add Interpreter** → **Virtualenv Environment** → **New**
3. **OK** to create the environment

### 4. Install Packages
1. **File** → **Settings** → **Project** → **Python Interpreter**
2. **+** button to add packages
3. Search and install: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`

## Professional Development Practices

### 1. Project Structure
```
ml_project/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_model.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── README.md
└── main.py
```

### 2. Code Organization
Create separate files for different functions:

**data_loader.py**:
```python
"""Data loading and preprocessing utilities."""
from sklearn.datasets import load_iris
import pandas as pd

def load_iris_data():
    """Load and return the iris dataset."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target_names[data.target]
    return data, df
```

**model.py**:
```python
"""Machine learning model definitions."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class IrisClassifier:
    """Random Forest classifier for iris species."""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state
        )
        self.is_trained = False
    
    def train(self, X, y, test_size=0.3):
        """Train the model on the provided data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Return test accuracy
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
```

### 3. Testing Your Code
Create **test_model.py**:
```python
"""Tests for the iris classifier."""
import unittest
import numpy as np
from src.model import IrisClassifier
from src.data_loader import load_iris_data

class TestIrisClassifier(unittest.TestCase):
    """Test cases for IrisClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data, _ = load_iris_data()
        self.classifier = IrisClassifier()
    
    def test_training(self):
        """Test that the model can be trained."""
        accuracy = self.classifier.train(self.data.data, self.data.target)
        self.assertGreater(accuracy, 0.8)  # Should be at least 80% accurate
        self.assertTrue(self.classifier.is_trained)
    
    def test_prediction(self):
        """Test that the model can make predictions."""
        self.classifier.train(self.data.data, self.data.target)
        
        # Test prediction on a single sample
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = self.classifier.predict(sample)
        
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1, 2])  # Valid class

if __name__ == '__main__':
    unittest.main()
```

Run tests with: `python -m pytest tests/`

### 4. Version Control with Git
```bash
# Initialize repository
git init

# Create .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo "data/raw/*" >> .gitignore

# Add and commit
git add .
git commit -m "Initial ML project setup"
```

### 5. Requirements Management
Create **requirements.txt**:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
pytest>=6.0.0
```

Install with: `pip install -r requirements.txt`

## IDE Features for ML Development

### Debugging
1. **Set breakpoints**: Click left margin next to line numbers
2. **Debug mode**: Press `F5` (VS Code) or `Shift+F9` (PyCharm)
3. **Step through code**: `F10` (step over), `F11` (step into)
4. **Inspect variables**: Hover over variables or use debug console

### Code Completion and Analysis
- **IntelliSense**: Auto-complete as you type
- **Error detection**: Red underlines for syntax errors
- **Code suggestions**: Improve code quality and performance
- **Documentation**: Hover over functions to see documentation

### Refactoring Tools
- **Rename variables**: `F2` to rename across entire project
- **Extract functions**: Select code → right-click → "Extract Method"
- **Organize imports**: Automatically sort and clean up imports
- **Code formatting**: Auto-format with `Shift+Alt+F`

## Transitioning from Notebooks

### Convert Notebooks to Scripts
```bash
# Convert Jupyter notebook to Python script
jupyter nbconvert --to script my_notebook.ipynb

# Clean up the converted script
# Remove notebook-specific code like %matplotlib inline
```

### Best Practices for Scripts
1. **Use functions**: Break code into reusable functions
2. **Add docstrings**: Document what each function does
3. **Handle errors**: Use try/except blocks
4. **Add logging**: Track what your code is doing
5. **Write tests**: Ensure your code works correctly

### Example: Adding Logging
```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(X, y):
    """Train model with logging."""
    logger.info("Starting model training...")
    
    # Training code here
    model = RandomForestClassifier()
    model.fit(X, y)
    
    logger.info("Model training completed successfully")
    return model
```

## Common Issues & Solutions

### Python interpreter not found
- **VS Code**: `Ctrl+Shift+P` → "Python: Select Interpreter"
- **PyCharm**: File → Settings → Project → Python Interpreter

### Packages not installing
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install in user directory if permission issues
pip install --user package_name

# Use virtual environment
python -m venv myenv
source myenv/bin/activate  # Mac/Linux
myenv\Scripts\activate     # Windows
```

### Code not running
- Check for syntax errors (red underlines)
- Ensure proper indentation (Python is sensitive to spaces)
- Make sure you're in the right directory
- Check that all imports are available

### Debugging not working
- Ensure you're running in debug mode (not just running the script)
- Check that breakpoints are set correctly
- Make sure the debugger is attached to the right process

## Advanced IDE Features

### Extensions/Plugins (VS Code)
- **Python Docstring Generator**: Auto-generate docstrings
- **GitLens**: Enhanced Git integration
- **Jupyter**: Run notebooks within VS Code
- **Python Test Explorer**: Visual test runner
- **Pylint**: Advanced code analysis

### Plugins (PyCharm)
- **Matplotlib Support**: Better plot visualization
- **CSV Plugin**: View CSV files in tabular format
- **Database Tools**: Connect to databases
- **Scientific Mode**: Enhanced data science features

### Integrated Terminal
- **VS Code**: `Ctrl+`` (backtick)
- **PyCharm**: View → Tool Windows → Terminal
- Run commands without leaving your IDE
- Multiple terminal sessions for different tasks

## Next Steps

🎉 **Congratulations!** You've successfully:
- Set up a professional Python development environment
- Created a complete ML project with proper structure
- Learned debugging, testing, and version control
- Built production-ready ML code

**Ready for more?** Check out:
- **[First ML Example](04-first-ml-example.md)** - More hands-on practice
- **[Next Steps Guide](05-next-steps.md)** - Your learning pathway
- **Sample script**: [Download our complete example](samples/python-sample.py)

**Want to explore other environments?**
- **[Google Colab](01-google-colab.md)** - For cloud-based development
- **[Jupyter Notebook](02-jupyter.md)** - For interactive analysis

You now have professional-grade tools that will serve you throughout your entire ML career! 🚀