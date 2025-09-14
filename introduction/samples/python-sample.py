#!/usr/bin/env python3
"""
🌸 Your First ML Model in Python IDE

This script demonstrates building a complete machine learning model
using professional Python development practices.

What you'll build: A flower species classifier that can identify 
iris flowers from their measurements.

Advantages of IDE development:
- 💼 Professional development practices
- 🐛 Advanced debugging capabilities  
- 📁 Excellent project organization
- 🔧 Powerful refactoring tools
- 🧪 Integrated testing frameworks

How to use this script:
1. Make sure you have required packages: pip install numpy pandas matplotlib scikit-learn seaborn
2. Run the script: python python-sample.py
3. Check the output files created in the same directory

Time needed: 15-20 minutes

Author: ML Beginner
Date: 2024
"""

# Standard library imports
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3
N_ESTIMATORS = 100
MAX_DEPTH = 3

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def print_header():
    """Print a welcome header for the script."""
    print("=" * 60)
    print("🌸 IRIS SPECIES CLASSIFIER")
    print("=" * 60)
    print("🚀 Building your first ML model with Python!")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"📁 Working directory: {os.getcwd()}")
    print("=" * 60)


def verify_setup():
    """Verify that all required packages are installed."""
    print("\n🔍 Verifying setup...")
    
    required_packages = {
        'numpy': np,
        'pandas': pd,
        'matplotlib': plt,
        'sklearn': 'sklearn',
        'seaborn': sns
    }
    
    missing_packages = []
    
    for package_name, package_obj in required_packages.items():
        try:
            if package_name == 'sklearn':
                import sklearn
                print(f"✅ {package_name} v{sklearn.__version__}")
            else:
                print(f"✅ {package_name} v{package_obj.__version__}")
        except (ImportError, AttributeError):
            print(f"❌ {package_name} is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n🚨 Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    else:
        print("✅ All packages are installed!")


def load_and_explore_data():
    """
    Load the iris dataset and perform initial exploration.
    
    Returns:
        tuple: (sklearn dataset object, pandas DataFrame)
    """
    print("\n🌸 Loading the Iris dataset...")
    
    # Load the famous iris flower dataset
    data = load_iris()
    
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target_names[data.target]
    
    # Display basic information
    print(f"📊 Dataset shape: {df.shape[0]} flowers, {df.shape[1]-1} measurements")
    print(f"🏷️  Species: {', '.join(data.target_names)}")
    print(f"📏 Measurements: {', '.join(data.feature_names)}")
    
    # Show basic statistics
    print("\n📈 Dataset statistics:")
    print(df.describe())
    
    # Check for missing data
    missing_values = df.isnull().sum().sum()
    print(f"\n❓ Missing values: {missing_values}")
    
    if missing_values == 0:
        print("✅ Dataset is clean and ready!")
    else:
        print("⚠️  Dataset has missing values that need attention")
    
    # Save the full dataset
    df.to_csv('iris_dataset.csv', index=False)
    print("💾 Full dataset saved to iris_dataset.csv")
    
    return data, df


def visualize_data(data, df):
    """
    Create comprehensive visualizations of the iris dataset.
    
    Args:
        data: sklearn dataset object
        df: pandas DataFrame with the data
    """
    print("\n📈 Creating data visualizations...")
    
    # Create output directory for plots
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Set up a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🌸 Iris Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Sepal measurements by species
    for i, species in enumerate(data.target_names):
        species_data = df[df['species'] == species]
        axes[0, 0].scatter(
            species_data['sepal length (cm)'], 
            species_data['sepal width (cm)'], 
            label=species, alpha=0.7, s=50
        )
    axes[0, 0].set_xlabel('Sepal Length (cm)')
    axes[0, 0].set_ylabel('Sepal Width (cm)')
    axes[0, 0].set_title('Sepal Measurements by Species')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Petal measurements by species  
    for i, species in enumerate(data.target_names):
        species_data = df[df['species'] == species]
        axes[0, 1].scatter(
            species_data['petal length (cm)'], 
            species_data['petal width (cm)'], 
            label=species, alpha=0.7, s=50
        )
    axes[0, 1].set_xlabel('Petal Length (cm)')
    axes[0, 1].set_ylabel('Petal Width (cm)')
    axes[0, 1].set_title('Petal Measurements by Species')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distribution of measurements
    df[data.feature_names].hist(bins=20, ax=axes[1, 0], alpha=0.7, color='skyblue')
    axes[1, 0].set_title('Distribution of All Measurements')
    
    # Plot 4: Species count
    species_counts = df['species'].value_counts()
    bars = axes[1, 1].bar(
        species_counts.index, 
        species_counts.values, 
        color=['#FF6B6B', '#4ECDC4', '#45B7D1']
    )
    axes[1, 1].set_title('Number of Flowers per Species')
    axes[1, 1].set_ylabel('Count')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom'
        )
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = plots_dir / 'iris_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Analysis plot saved to {plot_filename}")
    
    # Show plot if running interactively
    if hasattr(sys, 'ps1'):  # Check if running interactively
        plt.show()
    else:
        plt.close()  # Close to save memory in script mode
    
    print("📊 Key observations:")
    print("• Each species has distinct petal characteristics")
    print("• Setosa has the smallest petals")
    print("• Virginica has the largest petals")
    print("• This should make classification easier!")


def prepare_data(data):
    """
    Prepare data for machine learning by splitting into train/test sets.
    
    Args:
        data: sklearn dataset object
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n🔧 Preparing data for machine learning...")
    
    # Separate features from target
    X = data.data  # Features: measurements
    y = data.target  # Target: species
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Ensure equal representation
    )
    
    print(f"📚 Training set: {X_train.shape[0]} flowers")
    print(f"🧪 Testing set: {X_test.shape[0]} flowers")
    print(f"📊 Features per flower: {X_train.shape[1]}")
    
    # Show split by species
    train_species = pd.Series(y_train).value_counts().sort_index()
    test_species = pd.Series(y_test).value_counts().sort_index()
    
    print("\n🌸 Training set by species:")
    for i, species in enumerate(data.target_names):
        print(f"  {species}: {train_species[i]} flowers")
    
    print("\n🧪 Testing set by species:")
    for i, species in enumerate(data.target_names):
        print(f"  {species}: {test_species[i]} flowers")
    
    # Save data splits
    train_df = pd.DataFrame(X_train, columns=data.feature_names)
    train_df['species'] = y_train
    test_df = pd.DataFrame(X_test, columns=data.feature_names)
    test_df['species'] = y_test
    
    train_df.to_csv('iris_train.csv', index=False)
    test_df.to_csv('iris_test.csv', index=False)
    print("💾 Data splits saved to iris_train.csv and iris_test.csv")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        tuple: (trained model, training time in seconds)
    """
    print("\n🤖 Training the machine learning model...")
    
    # Create Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH
    )
    
    # Train the model and measure time
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"🎯 Model trained successfully!")
    print(f"⏱️  Training time: {training_time:.3f} seconds")
    print(f"🌳 Number of trees: {model.n_estimators}")
    print(f"📏 Max depth: {model.max_depth}")
    
    return model, training_time


def evaluate_model(model, data, X_test, y_test):
    """
    Evaluate the trained model and create detailed reports.
    
    Args:
        model: Trained classifier
        data: sklearn dataset object
        X_test: Test features
        y_test: Test targets
        
    Returns:
        tuple: (predictions, accuracy)
    """
    print("\n📊 Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📈 Accuracy on test set: {accuracy:.1%}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': data.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Most important features for classification:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("💾 Feature importance saved to feature_importance.csv")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    report = classification_report(
        y_test, y_pred, 
        target_names=data.target_names, 
        output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv')
    print("💾 Classification report saved to classification_report.csv")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=data.target_names,
        yticklabels=data.target_names,
        cbar_kws={'label': 'Number of Flowers'}
    )
    plt.title('🎯 Confusion Matrix: Actual vs Predicted Species', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Species', fontsize=12)
    plt.ylabel('Actual Species', fontsize=12)
    
    # Save confusion matrix
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    confusion_matrix_file = plots_dir / 'confusion_matrix.png'
    plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to {confusion_matrix_file}")
    
    if hasattr(sys, 'ps1'):
        plt.show()
    else:
        plt.close()
    
    # Per-species accuracy
    print("\n🌸 Accuracy by species:")
    for i, species in enumerate(data.target_names):
        species_mask = y_test == i
        if species_mask.sum() > 0:
            species_accuracy = (y_pred[species_mask] == i).mean()
            print(f"  {species}: {species_accuracy:.1%}")
    
    # Check for misclassifications
    misclassified_mask = y_test != y_pred
    misclassified_count = misclassified_mask.sum()
    
    if misclassified_count > 0:
        print(f"\n❌ Misclassified flowers: {misclassified_count}")
        
        # Save misclassified examples
        misclassified_df = pd.DataFrame(
            X_test[misclassified_mask], 
            columns=data.feature_names
        )
        misclassified_df['actual'] = data.target_names[y_test[misclassified_mask]]
        misclassified_df['predicted'] = data.target_names[y_pred[misclassified_mask]]
        misclassified_df.to_csv('misclassified_flowers.csv', index=False)
        print("💾 Misclassified examples saved to misclassified_flowers.csv")
    else:
        print("\n🎉 Perfect classification! No mistakes on the test set!")
    
    return y_pred, accuracy


def make_predictions(model, data):
    """
    Make predictions on new flower samples.
    
    Args:
        model: Trained classifier
        data: sklearn dataset object
        
    Returns:
        pandas.DataFrame: Prediction results
    """
    print("\n🔮 Making predictions on new flowers...")
    
    # Create example new flowers
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
    
    # Store results for CSV
    results_data = []
    
    for i, (flower, pred, prob) in enumerate(zip(new_flowers, predictions, probabilities)):
        species = data.target_names[pred]
        confidence = prob.max()
        
        print(f"\n🌺 Flower #{i+1}:")
        print(f"   Measurements: {flower}")
        print(f"   Predicted species: {species}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Store results
        result_row = {
            'flower_id': i+1,
            'sepal_length': flower[0],
            'sepal_width': flower[1],
            'petal_length': flower[2],
            'petal_width': flower[3],
            'predicted_species': species,
            'confidence': confidence
        }
        
        # Add probabilities for each species
        for j, (species_name, probability) in enumerate(zip(data.target_names, prob)):
            result_row[f'prob_{species_name.lower()}'] = probability
        
        results_data.append(result_row)
        
        # Show probabilities
        print("   Probabilities:")
        for j, (species_name, probability) in enumerate(zip(data.target_names, prob)):
            emoji = "🎯" if j == pred else "  "
            print(f"     {emoji} {species_name}: {probability:.1%}")
    
    # Save predictions
    predictions_df = pd.DataFrame(results_data)
    predictions_df.to_csv('new_flower_predictions.csv', index=False)
    print("\n💾 Predictions saved to new_flower_predictions.csv")
    print("✨ Amazing! Your model can now identify iris species from measurements!")
    
    return predictions_df


def save_model_and_summary(model, accuracy, training_time):
    """
    Save the trained model and create a comprehensive summary.
    
    Args:
        model: Trained classifier
        accuracy: Model accuracy
        training_time: Time taken to train the model
        
    Returns:
        str: Filename of saved model
    """
    print("\n💾 Saving model and creating summary...")
    
    # Save the model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'iris_classifier_{timestamp}.pkl'
    joblib.dump(model, model_filename)
    
    model_size = os.path.getsize(model_filename) / 1024  # Size in KB
    print(f"✅ Model saved as: {model_filename}")
    print(f"📁 File size: {model_size:.1f} KB")
    
    # Create comprehensive summary
    summary = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'script_name': __file__,
        'python_version': sys.version.split()[0],
        'working_directory': os.getcwd(),
        'dataset_size': 150,  # Iris dataset size
        'training_size': int(150 * (1 - TEST_SIZE)),
        'test_size': int(150 * TEST_SIZE),
        'model_type': 'RandomForestClassifier',
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'random_state': RANDOM_STATE,
        'accuracy': accuracy,
        'training_time_seconds': training_time,
        'model_filename': model_filename,
        'model_size_kb': model_size
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('experiment_summary.csv', index=False)
    
    print("\n📊 Experiment Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n💾 Summary saved to experiment_summary.csv")
    
    return model_filename


def test_model_persistence(model_filename, data):
    """
    Test that the saved model can be loaded and used.
    
    Args:
        model_filename: Path to saved model
        data: sklearn dataset object
    """
    print("\n🔄 Testing model persistence...")
    
    try:
        # Load the model
        loaded_model = joblib.load(model_filename)
        
        # Test on a sample flower
        test_flower = np.array([[5.0, 3.0, 1.6, 0.2]])
        prediction = loaded_model.predict(test_flower)
        probability = loaded_model.predict_proba(test_flower)
        
        print(f"✅ Model loaded successfully from {model_filename}")
        print(f"🌸 Test prediction: {data.target_names[prediction[0]]}")
        print(f"🎯 Confidence: {probability.max():.1%}")
        print("💡 Your model is now saved and can be used anytime!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")


def show_created_files():
    """Display all files created during this session."""
    print("\n📂 Files created in this session:")
    
    expected_files = [
        'iris_dataset.csv',
        'iris_train.csv', 
        'iris_test.csv',
        'feature_importance.csv',
        'classification_report.csv',
        'new_flower_predictions.csv',
        'experiment_summary.csv',
        'plots/iris_analysis.png',
        'plots/confusion_matrix.png'
    ]
    
    # Add model file (find the most recent one)
    model_files = [f for f in os.listdir('.') if f.startswith('iris_classifier_') and f.endswith('.pkl')]
    if model_files:
        expected_files.append(max(model_files))  # Most recent model file
    
    total_size = 0
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            print(f"  📄 {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} (not found)")
    
    print(f"\n📊 Total files size: {total_size / 1024:.1f} KB")
    
    print("\n💡 Advantages of Python IDE development:")
    print("  ✅ Professional code organization")
    print("  ✅ Advanced debugging capabilities")
    print("  ✅ Excellent refactoring tools")
    print("  ✅ Integrated testing frameworks")
    print("  ✅ Version control integration")
    print("  ✅ Code analysis and suggestions")


def main():
    """
    Main function that orchestrates the entire ML workflow.
    
    This function demonstrates professional Python practices:
    - Clear function organization
    - Error handling
    - Comprehensive logging
    - File management
    - Code documentation
    """
    try:
        # Initialize
        print_header()
        verify_setup()
        
        # Data pipeline
        data, df = load_and_explore_data()
        visualize_data(data, df)
        X_train, X_test, y_train, y_test = prepare_data(data)
        
        # Model pipeline
        model, training_time = train_model(X_train, y_train)
        y_pred, accuracy = evaluate_model(model, data, X_test, y_test)
        predictions_df = make_predictions(model, data)
        
        # Persistence and summary
        model_filename = save_model_and_summary(model, accuracy, training_time)
        test_model_persistence(model_filename, data)
        show_created_files()
        
        # Success message
        print("\n" + "=" * 60)
        print("🎉 CONGRATULATIONS!")
        print("=" * 60)
        print("✅ You've successfully built your first ML model using Python!")
        print(f"🎯 Final accuracy: {accuracy:.1%}")
        print(f"⏱️  Total runtime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🚀 Next steps:")
        print("  • Try different algorithms (SVM, Neural Networks)")
        print("  • Work with your own datasets")
        print("  • Learn feature engineering techniques")
        print("  • Build models for regression and clustering")
        print("  • Explore advanced Python ML libraries")
        print("\n💡 You now have professional ML development skills!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("🔧 Check that all required packages are installed:")
        print("pip install numpy pandas matplotlib scikit-learn seaborn")
        sys.exit(1)


if __name__ == "__main__":
    """
    Entry point for the script.
    
    This is a Python best practice - code in this block only runs
    when the script is executed directly, not when imported as a module.
    """
    main()