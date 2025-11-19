#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

# The exploratory analysis was first carried out in a Jupyter Notebook, where the dataset was visualised, the features were inspected, and a baseline understanding of the Iris data was established.
# The final implementation was then reproduced in this Python script following the same workflow, but with additional evaluation metrics and programmatic structure.
# While the notebook focused more on interactive analysis, the Python script formalised the machine learning pipeline by including accuracy, precision, recall, F1-score, and a confusion matrix.
# Feature importance was also calculated to provide insight into which attributes contributed most to the classifier’s decisions.


def load_iris_data():
    """ Load and return the iris dataset """
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {iris.target_names}")
    print(f"Features: {list(X.columns)}")

    return X, y, iris.target_names


def train_model(X, y, test_size=0.2, random_state=42):
    """ Train the iris classification model """
    print(f"\nSplitting data with test_size={test_size}, random_state={random_state}")

    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Train the model
    print("\nTraining DecisionTree model...")
    model = DecisionTreeClassifier(
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    print("Predictions:", y_pred[:5])
    print("True labels:", y_test[:5])

    return model, X_test, y_test, y_pred


# In the notebook I made use of 2 evaluation metrics(Accuracy and Confusion matrix)
# In the code I made use of 5 evaluation metrics(Accuracy, Confusion matrix, Precision, Recall and F1-score)
def evaluate_model(y_test, y_pred, target_names):
    """ Evaluate and print model performance"""
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return accuracy


def save_confusion_matrix(y_test, y_pred, target_names):
    """ Create and save confusion matrix figure"""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to master folder
    master_dir = os.path.dirname(script_dir)

    # Create output directory path in master folder
    output_dir = os.path.join(master_dir, 'output')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create figure with better styling
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                cbar_kws={'label': 'Count'},
                square=True,
                linewidths=0.5)

    plt.title('Confusion Matrix - Iris Classification', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=30, fontweight='bold')

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    # Close the figure to free memory
    plt.close()

    # Print confusion matrix to console as well
    print(f"\nConfusion Matrix:")
    print(f"{'':>12}", end="")
    for name in target_names:
        print(f"{name:>12}", end="")
    print()

    for i, true_name in enumerate(target_names):
        print(f"{true_name:>12}", end="")
        for j in range(len(target_names)):
            print(f"{cm[i,j]:>12}", end="")
        print()


def save_model(model, accuracy, test_size, random_state, output_dir=None):
    """ Save the trained model with metadata """
    if output_dir is None:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one level to master folder
        master_dir = os.path.dirname(script_dir)

        # Create output directory path in master folder
        output_dir = os.path.join(master_dir, 'output')

    # Generate filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"iris_model_acc{accuracy:.3f}_ts{test_size}_rs{random_state}_{timestamp}.pkl")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    joblib.dump(model, filename)

    return filename


def main():
    parser = argparse.ArgumentParser(
        description='Train iris Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (0.0 to 1.0)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducible results'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information during training'
    )

    args = parser.parse_args()

    print("IRIS CLASSIFICATION MODEL TRAINING")
    print("="*50)
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")

    try:
        # Load data
        X, y, target_names = load_iris_data()

        # Train model
        model, X_test, y_test, y_pred = train_model(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # Evaluate model
        accuracy = evaluate_model(y_test, y_pred, target_names)

        # Save confusion matrix figure
        save_confusion_matrix(y_test, y_pred, target_names)

        # Save model
        model_path = save_model(model, accuracy, args.test_size, args.random_state)

        # Feature importance
        if args.verbose:
            print(f"\nFeature Importance:")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance)

        print(f"\n{'='*50}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Confusion matrix saved to: output/confusion_matrix.png")
        print(f"Model saved to: output/{os.path.basename(model_path)}")
        print(f"\n{'='*50}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
