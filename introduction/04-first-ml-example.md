# Your First Machine Learning Example

**What you'll learn:**
- How to build the same ML model in any environment
- The complete machine learning workflow from data to predictions
- How to interpret and visualize your results

**Time needed:** 20 minutes
**Prerequisites:** One of the environments set up (Colab, Jupyter, or Python IDE)

## The Universal ML Example

This example works identically in **Google Colab**, **Jupyter Notebook**, and **Python IDEs**. The code is the same - only the way you run it differs slightly.

### What We're Building

We'll create a **flower species classifier** that can:
- 🌸 Identify iris flower species from measurements
- 📊 Visualize the data patterns
- 🎯 Achieve 95%+ accuracy
- 🔮 Make predictions on new flowers

This is a classic machine learning problem that teaches all the fundamentals!

## Choose Your Environment

### Option 1: Google Colab (Easiest)
1. **Click this link**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CS-6140-ML-Fall-2025/TA-Classes/blob/main/introduction/samples/colab-sample.ipynb)
2. **Save a copy**: File → Save a copy in Drive
3. **Run each cell**: Press Shift + Enter for each code block below

### Option 2: Jupyter Notebook (Local)
1. **Download the notebook**: [jupyter-sample.ipynb](samples/jupyter-sample.ipynb)
2. **Open Jupyter**: Run `jupyter notebook` in your terminal
3. **Open the file**: Navigate to and open the downloaded notebook
4. **Run each cell**: Press Shift + Enter for each code block below

### Option 3: Python IDE (Professional)
1. **Download the script**: [python-sample.py](samples/python-sample.py)
2. **Open in your IDE**: VS Code, PyCharm, etc.
3. **Install packages**: `pip install numpy pandas matplotlib scikit-learn seaborn`
4. **Run the script**: Press F5 or run `python python-sample.py`

## The Complete Example

Follow along by copying each code block into your chosen environment:

### Step 1: Import Libraries
```python
# Import the tools we need for machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print("🚀 Ready to build your first ML model!")
```

**What this does**: Imports all the tools we need for data science and machine learning.

### Step 2: Load and Explore the Data
```python
# Load the famous iris flower dataset
print("🌸 Loading the Iris dataset...")
data = load_iris()

# Convert to a pandas DataFrame for easier handling
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target_names[data.target]

# Display basic information
print(f"📊 Dataset shape: {df.shape[0]} flowers, {df.shape[1]-1} measurements")
print(f"🏷️  Species: {', '.join(data.target_names)}")
print(f"📏 Measurements: {', '.join(data.feature_names)}")

# Show the first few flowers
print("\n🔍 First 5 flowers in our dataset:")
print(df.head())

# Check for any missing data
print(f"\n❓ Missing values: {df.isnull().sum().sum()}")
print("✅ Dataset is clean and ready!")
```

**What this does**: Loads a dataset of 150 iris flowers with 4 measurements each, and displays basic information about the data.

### Step 3: Visualize the Data
```python
# Create beautiful visualizations to understand our data
print("📈 Creating data visualizations...")

# Set up a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🌸 Iris Dataset Analysis', fontsize=16, fontweight='bold')

# Plot 1: Sepal measurements by species
axes[0, 0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], 
                  c=data.target, cmap='viridis', alpha=0.7, s=50)
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Sepal Width (cm)')
axes[0, 0].set_title('Sepal Measurements by Species')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Petal measurements by species  
axes[0, 1].scatter(df['petal length (cm)'], df['petal width (cm)'], 
                  c=data.target, cmap='viridis', alpha=0.7, s=50)
axes[0, 1].set_xlabel('Petal Length (cm)')
axes[0, 1].set_ylabel('Petal Width (cm)')
axes[0, 1].set_title('Petal Measurements by Species')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of measurements
df[data.feature_names].hist(bins=20, ax=axes[1, 0], alpha=0.7, color='skyblue')
axes[1, 0].set_title('Distribution of All Measurements')

# Plot 4: Species count
species_counts = df['species'].value_counts()
bars = axes[1, 1].bar(species_counts.index, species_counts.values, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1, 1].set_title('Number of Flowers per Species')
axes[1, 1].set_ylabel('Count')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("📊 Key observations:")
print("• Each species has distinct petal characteristics")
print("• Setosa has the smallest petals")
print("• Virginica has the largest petals")
print("• This should make classification easier!")
```

**What this does**: Creates four different visualizations to help us understand the patterns in our flower data.

### Step 4: Prepare Data for Machine Learning
```python
# Prepare our data for training a machine learning model
print("🔧 Preparing data for machine learning...")

# Separate features (measurements) from target (species)
X = data.data  # Features: sepal length, sepal width, petal length, petal width
y = data.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,      # Use 30% for testing
    random_state=42,    # For reproducible results
    stratify=y          # Ensure equal representation of each species
)

print(f"📚 Training set: {X_train.shape[0]} flowers")
print(f"🧪 Testing set: {X_test.shape[0]} flowers")
print(f"📊 Features per flower: {X_train.shape[1]}")

# Show the split by species
train_species = pd.Series(y_train).value_counts().sort_index()
test_species = pd.Series(y_test).value_counts().sort_index()

print("\n🌸 Training set by species:")
for i, species in enumerate(data.target_names):
    print(f"  {species}: {train_species[i]} flowers")

print("\n🧪 Testing set by species:")
for i, species in enumerate(data.target_names):
    print(f"  {species}: {test_species[i]} flowers")
```

**What this does**: Splits our data into training (70%) and testing (30%) sets, ensuring we can evaluate our model properly.

### Step 5: Train the Machine Learning Model
```python
# Create and train our machine learning model
print("🤖 Training the machine learning model...")

# Create a Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,    # Use 100 decision trees
    random_state=42,     # For reproducible results
    max_depth=3          # Prevent overfitting
)

# Train the model on our training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model trained successfully!")
print(f"📈 Accuracy on test set: {accuracy:.1%}")

# Show which features are most important
feature_importance = pd.DataFrame({
    'feature': data.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Most important features for classification:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")
```

**What this does**: Creates a Random Forest model (which uses many decision trees) and trains it to recognize flower species from measurements.

### Step 6: Evaluate Model Performance
```python
# Evaluate how well our model performs
print("📊 Evaluating model performance...")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Create a confusion matrix to see where the model makes mistakes
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, 
            yticklabels=data.target_names,
            cbar_kws={'label': 'Number of Flowers'})
plt.title('🎯 Confusion Matrix: Actual vs Predicted Species', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Species', fontsize=12)
plt.ylabel('Actual Species', fontsize=12)
plt.show()

# Calculate per-species accuracy
print("\n🌸 Accuracy by species:")
for i, species in enumerate(data.target_names):
    species_mask = y_test == i
    if species_mask.sum() > 0:
        species_accuracy = (y_pred[species_mask] == i).mean()
        print(f"  {species}: {species_accuracy:.1%}")

# Show any misclassifications
misclassified = X_test[y_test != y_pred]
if len(misclassified) > 0:
    print(f"\n❌ Misclassified flowers: {len(misclassified)}")
    print("These are the flowers our model got wrong - let's learn from them!")
else:
    print("\n🎉 Perfect classification! No mistakes on the test set!")
```

**What this does**: Shows detailed results about how well our model performs, including which species it's best at identifying.

### Step 7: Make Predictions on New Flowers
```python
# Use our trained model to predict new flower species
print("🔮 Making predictions on new flowers...")

# Create some example new flowers to classify
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Small petals - likely Setosa
    [6.2, 2.8, 4.8, 1.8],  # Large petals - likely Virginica  
    [5.7, 2.8, 4.1, 1.3],  # Medium petals - likely Versicolor
    [4.9, 3.1, 1.5, 0.1],  # Very small petals - likely Setosa
    [7.2, 3.0, 5.8, 1.6]   # Very large petals - likely Virginica
])

# Make predictions
predictions = model.predict(new_flowers)
probabilities = model.predict_proba(new_flowers)

print("\n🌸 Prediction Results:")
print("=" * 60)

for i, (flower, pred, prob) in enumerate(zip(new_flowers, predictions, probabilities)):
    species = data.target_names[pred]
    confidence = prob.max()
    
    print(f"\n🌺 Flower #{i+1}:")
    print(f"   Measurements: {flower}")
    print(f"   Predicted species: {species}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Show probability for each species
    print("   Probabilities:")
    for j, (species_name, probability) in enumerate(zip(data.target_names, prob)):
        emoji = "🎯" if j == pred else "  "
        print(f"     {emoji} {species_name}: {probability:.1%}")

print("\n✨ Amazing! Your model can now identify iris species from measurements!")
```

**What this does**: Uses our trained model to predict the species of new flowers we haven't seen before.

### Step 8: Summary and Next Steps
```python
# Summarize what we've accomplished
print("\n" + "="*60)
print("🎉 CONGRATULATIONS! You've built your first ML model!")
print("="*60)

print(f"\n📊 What you accomplished:")
print(f"   ✅ Loaded and explored a dataset of {len(df)} flowers")
print(f"   ✅ Visualized data patterns across {len(data.feature_names)} features")
print(f"   ✅ Trained a Random Forest model with {model.n_estimators} trees")
print(f"   ✅ Achieved {accuracy:.1%} accuracy on unseen data")
print(f"   ✅ Made predictions on new flower measurements")

print(f"\n🧠 Key machine learning concepts you learned:")
print(f"   • Data loading and exploration")
print(f"   • Data visualization and pattern recognition")
print(f"   • Train/test split for model evaluation")
print(f"   • Model training and prediction")
print(f"   • Performance evaluation and interpretation")

print(f"\n🚀 You're ready for the next level!")
print(f"   • Try different algorithms (SVM, Neural Networks)")
print(f"   • Work with larger, more complex datasets")
print(f"   • Learn feature engineering and data preprocessing")
print(f"   • Build models for regression and clustering")

print(f"\n💡 Remember: The same code works in Colab, Jupyter, and Python IDEs!")
print(f"   You now have transferable skills across all ML environments.")
```

**What this does**: Celebrates your success and outlines what you've learned and where to go next!

## Environment-Specific Features

### Google Colab Advantages
- **Free GPU**: Add this cell to use GPU acceleration:
```python
# Check if GPU is available
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

- **Easy Sharing**: Click "Share" button to send your notebook to others
- **Drive Integration**: Save large datasets to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Jupyter Notebook Advantages
- **Local Files**: Easy access to files on your computer:
```python
# Save your model for later use
import joblib
joblib.dump(model, 'iris_classifier.pkl')
print("Model saved to iris_classifier.pkl")
```

- **Offline Work**: No internet required after setup
- **Custom Extensions**: Install additional tools:
```bash
pip install jupyter_contrib_nbextensions
```

### Python IDE Advantages
- **Debugging**: Set breakpoints and step through code
- **Code Organization**: Split into multiple files:
```python
# data_loader.py
def load_iris_data():
    from sklearn.datasets import load_iris
    return load_iris()

# model.py  
class IrisClassifier:
    def __init__(self):
        self.model = RandomForestClassifier()
    # ... rest of class
```

- **Testing**: Write automated tests:
```python
def test_model_accuracy():
    # Test that model achieves minimum accuracy
    assert accuracy > 0.9
```

## Troubleshooting

### Common Issues

**"Module not found" error**:
```python
# Install missing packages
!pip install scikit-learn  # In notebooks
# or
pip install scikit-learn   # In terminal
```

**Plots not showing**:
```python
# Add this for notebooks
%matplotlib inline

# Or this for better plots
%matplotlib widget
```

**Low accuracy**:
- Check that you're using the same random_state (42)
- Ensure data is loaded correctly
- Verify train/test split is working

**Code runs slowly**:
- Use smaller datasets for learning
- Reduce n_estimators in RandomForestClassifier
- Consider using simpler algorithms first

## What You've Learned

🎯 **Core ML Concepts**:
- **Supervised Learning**: Learning from labeled examples
- **Classification**: Predicting categories (flower species)
- **Train/Test Split**: Evaluating model on unseen data
- **Feature Importance**: Which measurements matter most
- **Model Evaluation**: Understanding accuracy and errors

🛠️ **Technical Skills**:
- **Data Loading**: Using sklearn datasets
- **Data Exploration**: Pandas DataFrames and basic statistics
- **Visualization**: Creating informative plots with matplotlib
- **Model Training**: Using RandomForestClassifier
- **Prediction**: Making predictions on new data

🚀 **Professional Practices**:
- **Reproducible Results**: Using random_state for consistency
- **Code Organization**: Breaking complex tasks into steps
- **Documentation**: Adding comments and explanations
- **Evaluation**: Properly testing model performance

## Next Steps

**Ready to continue your ML journey?**

1. **[Next Steps Guide](05-next-steps.md)** - Your complete learning roadmap
2. **Try Different Algorithms**: Experiment with SVM, Neural Networks, etc.
3. **New Datasets**: Work with different types of data (text, images, etc.)
4. **Advanced Topics**: Feature engineering, hyperparameter tuning, ensemble methods

**Want to explore other environments?**
- **[Google Colab Guide](01-google-colab.md)** - Master cloud-based development
- **[Jupyter Guide](02-jupyter.md)** - Become a local development pro  
- **[Python IDE Guide](03-python.md)** - Learn professional workflows

🎉 **Congratulations!** You've successfully built your first machine learning model. The journey from here only gets more exciting! 🚀