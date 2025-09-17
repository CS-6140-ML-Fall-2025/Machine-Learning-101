from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    print("🌸 Building your first ML model!")

    # Load flower flower_data
    flower_data = load_iris()
    X, y = flower_data.data, flower_data.target

    print(f"📊 Loaded {len(X)} flower samples")
    print(f"🏷️ Species: {list(flower_data.target_names)}")

    # Split flower_data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"🎯 Model accuracy: {accuracy:.1%}")

    # Try predicting a new flower
    new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Small petals (close to 'setosa')
    prediction = model.predict(new_flower)
    species = flower_data.target_names[prediction[0]]

    print(f"🔮 New flower prediction: {species}")
    print("✅ Success! Your ML model is working!")


if __name__ == "__main__":
    main()
