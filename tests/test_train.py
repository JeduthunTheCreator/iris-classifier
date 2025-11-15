import pytest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import sys
from unittest.mock import patch

# Variable to store the test, project and source directories
# This allows importing modules from the source folder during testing
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
src_dir = os.path.join(project_root, 'src')

sys.path.insert(0, src_dir)

# Importing the main training module with error handling to check for any import errors
try:
    import train
    from train import (
        load_iris_data,
        train_model,
        evaluate_model,
        save_confusion_matrix,
        save_model
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Looking for train.py in: {src_dir}")
    raise


# Main class that contains all the individual test components and a full integration test
class TestIrisTraining:
    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """ Configure matplotlib for headless testing"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()
        yield
        plt.close('all')

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for testing outputs"""
        temp_dir = tempfile.mkdtemp()

        # Create a mock project structure
        src_dir = os.path.join(temp_dir, 'src')
        os.makedirs(src_dir, exist_ok=True)

        # Creating a dummy train.py file for __file__ reference
        dummy_train_path = os.path.join(src_dir, 'train.py')
        with open(dummy_train_path, 'w') as f:
            f.write(' # Dummy train.py for testing\n')

        yield temp_dir, dummy_train_path

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Load sample iris data for testing"""
        return load_iris_data()

    def test_load_iris_data(self, sample_data):
        """Test that iris data loads correctly"""
        X, y, target_names = sample_data

        # Check data shapes and types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        # Check expected dimensions
        assert X.shape == (150, 4)
        assert len(y) == 150
        assert len(target_names) == 3

        # Check column names
        expected_columns = [
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
        ]
        assert list(X.columns) == expected_columns

        # Check target names
        expected_targets = ['setosa', 'versicolor', 'virginica']
        assert list(target_names) == expected_targets

    def test_train_model(self, sample_data):
        """Test that model training works correctly"""
        X, y, target_names = sample_data

        # Train model with test parameters
        model, X_test, y_test, y_pred = train_model(
            X, y, test_size=0.2, random_state=42
        )

        # Check model type
        assert isinstance(model, DecisionTreeClassifier)

        # Check data shapes
        assert X_test.shape[0] == int(0.2 * 150)  # 20% of 150 = 30
        assert len(y_test) == len(y_pred)

        # Check predictions are valid
        assert all(pred in [0, 1, 2] for pred in y_pred)

    def test_evaluate_model(self, sample_data):
        """Test model evaluation function"""
        X, y, target_names = sample_data

        # Train a simple model
        model, X_test, y_test, y_pred = train_model(
            X, y, test_size=0.2, random_state=42
        )

        # Evaluate model
        accuracy = evaluate_model(y_test, y_pred, target_names)

        # Check accuracy is reasonable
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy > 0.8  # Iris dataset should have high accuracy

    def test_save_confusion_matrix(self, temp_output_dir, sample_data):
        """Test confusion matrix saving"""
        temp_dir, dummy_train_path = temp_output_dir
        X, y, target_names = sample_data

        # Train model
        model, X_test, y_test, y_pred = train_model(
            X, y, test_size=0.2, random_state=42
        )

        # Mock os.path.abspath
        with patch('train.os.path.abspath') as mock_abspath:
            # Make abspath return a path in temp_dir
            mock_abspath.return_value = os.path.join(temp_dir, 'src', 'train.py')

            # Save confusion matrix
            save_confusion_matrix(y_test, y_pred, target_names)

            # Check if file was created
            expected_path = os.path.join(temp_dir, 'output', 'confusion_matrix.png')
            assert os.path.exists(expected_path), f"Confusion matrix not found at {expected_path}"
            assert os.path.getsize(expected_path) > 0, "Confusion matrix file is empty"

    def test_save_model(self, temp_output_dir, sample_data):
        """Test model saving"""
        temp_dir, dummy_train_path = temp_output_dir
        X, y, target_names = sample_data

        # Train model
        model, X_test, y_test, y_pred = train_model(
            X, y, test_size=0.2, random_state=42
        )

        accuracy = evaluate_model(y_test, y_pred, target_names)

        # Mock os.path.abspath like other tests
        with patch('train.os.path.abspath') as mock_abspath:
            mock_abspath.return_value = os.path.join(temp_dir, 'src', 'train.py')

            # Save model
            model_path = save_model(model, accuracy, 0.2, 42)

            # Check if file was created
            assert os.path.exists(model_path), f"Model file not found at {model_path}"
            assert os.path.getsize(model_path) > 0, "Model file is empty"

            # Check if model can be loaded
            loaded_model = joblib.load(model_path)
            assert isinstance(loaded_model, DecisionTreeClassifier)

            # Check if loaded model works
            test_predictions = loaded_model.predict(X_test)
            assert len(test_predictions) == len(y_test)

    @pytest.mark.integration
    def test_full_pipeline(self, temp_output_dir):
        """Integration test for the full training pipeline"""
        temp_dir, dummy_train_path = temp_output_dir

        # Mock os.path.abspath for both functions
        with patch('train.os.path.abspath') as mock_abspath:
            # Make abspath return a path in temp_dir
            mock_abspath.return_value = os.path.join(temp_dir, 'src', 'train.py')

            # Load data
            X, y, target_names = load_iris_data()

            # Train model
            model, X_test, y_test, y_pred = train_model(
                X, y, test_size=0.2, random_state=42
            )

            # Evaluate
            accuracy = evaluate_model(y_test, y_pred, target_names)

            # Save outputs
            save_confusion_matrix(y_test, y_pred, target_names)
            model_path = save_model(model, accuracy, 0.2, 42)

            # Create expected paths
            output_dir = os.path.join(temp_dir, 'output')
            confusion_path = os.path.join(output_dir, 'confusion_matrix.png')

            # Check all outputs exist
            assert os.path.exists(confusion_path), f"Confusion matrix not created at {confusion_path}"
            assert os.path.exists(model_path), f"Model not created at {model_path}"

            # Check accuracy is reasonable
            assert accuracy > 0.8

            # Integration-specific checks
            assert len(y_pred) == len(y_test), "Prediction and test set size mismatch"

            # Verify model can make predictions on original data
            full_predictions = model.predict(X)
            assert len(full_predictions) == len(y), "Model can't predict on full dataset"


if __name__ == '__main__':
    pytest.main([__file__])
