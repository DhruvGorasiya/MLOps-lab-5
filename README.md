# Ray Distributed Machine Learning - Enhanced Notebook

This notebook demonstrates the power of distributed computing using Ray for parallel machine learning model training. It compares sequential vs parallel training approaches on the California Housing dataset using Random Forest Regressors.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Key Enhancements](#key-enhancements)
- [Performance Metrics](#performance-metrics)
- [Visualizations](#visualizations)
- [Model Persistence](#model-persistence)
- [Results Export](#results-export)

## 🎯 Overview

This notebook trains multiple Random Forest models with varying hyperparameters (number of estimators) to find the best performing model. The key demonstration is comparing:

- **Sequential Training**: Models trained one after another
- **Parallel Training**: Models trained simultaneously using Ray's distributed computing

The notebook shows how Ray can significantly speed up hyperparameter exploration by utilizing all available CPU cores.

## ✨ Features

### Core Functionality

1. **Sequential Model Training**

   - Trains 20 Random Forest models sequentially
   - Each model uses increasing number of estimators (8, 12, 16, ..., 84)
   - Measures training time for each model

2. **Parallel Model Training with Ray**
   - Distributes model training across available CPU cores
   - Uses Ray's object store for efficient data sharing
   - Demonstrates asynchronous task execution

### Enhanced Features Added

#### 1. **Comprehensive Metrics**

- **MSE (Mean Squared Error)**: Primary metric for model selection
- **RMSE (Root Mean Squared Error)**: Interpretable error metric
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Coefficient of determination

#### 2. **Performance Visualizations**

- **MSE vs Number of Estimators**: Compare model performance across configurations
- **R² Score Trends**: Visualize model quality improvements
- **Training Time Comparison**: Bar chart showing total time savings
- **Individual Model Training Times**: Line plot comparing per-model execution times
- **RMSE and MAE Comparisons**: Additional error metric visualizations
- **Metrics Distribution**: Box plots showing distribution of all metrics
- **Time Saved per Model**: Bar chart showing time savings breakdown

#### 3. **Model Persistence**

- Save the best trained model to disk (`.pkl` format)
- Export model metadata (JSON format) including:
  - Hyperparameters
  - Performance metrics
  - Training time
- Demonstrate model loading and prediction

#### 4. **Results Export**

- **CSV Exports**:
  - Sequential results with all metrics
  - Parallel results with all metrics
  - Detailed side-by-side comparison table
- **JSON Exports**:
  - Comparison summary with best models and performance metrics
  - Model metadata

#### 5. **Resource Monitoring**

- Display Ray cluster information
- Show available resources (CPU, memory)
- Node information and status

#### 6. **Detailed Comparison Analysis**

- Side-by-side metrics comparison table
- Summary statistics:
  - Average metric differences
  - Total time saved
  - Average time saved per model

#### 7. **Prediction Analysis**

- **Actual vs Predicted Plots**: Scatter plots for both sequential and parallel best models
- **Residual Analysis**: Residual plots to check model assumptions
- **Final Performance Metrics**: Comprehensive test set evaluation

#### 8. **Performance Summary**

- Speedup calculation (sequential time / parallel time)
- Time saved in seconds and percentage
- Best model identification for both approaches

## 📦 Requirements

### Python Packages

```python
- pandas
- numpy
- scikit-learn
- ray
- matplotlib
- seaborn
- joblib
```

### System Requirements

- Python 3.8+
- Multiple CPU cores (for parallel execution benefits)
- Sufficient RAM for dataset and model storage

## 🚀 Installation

1. **Install Ray**:

   ```bash
   pip install ray
   ```

2. **Install other dependencies**:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

3. **For Jupyter Notebook**:
   ```bash
   pip install jupyter ipywidgets
   ```

## 💻 Usage

1. **Open the notebook**:

   ```bash
   jupyter notebook Ray.ipynb
   ```

2. **Run cells sequentially**:

   - The notebook is designed to be run from top to bottom
   - Each section builds on previous results

3. **Key Execution Steps**:
   - **Cell 1**: Import libraries and set up environment
   - **Cell 3**: Load and prepare California Housing dataset
   - **Cell 8**: Run sequential training (takes ~2 minutes)
   - **Cell 14**: Initialize Ray cluster
   - **Cell 27**: Run parallel training (takes ~30 seconds)
   - **Cells 32-44**: Enhanced features and analysis

## 📁 Output Structure

After running the notebook, the following directory structure will be created:

```
Ray/
├── Ray.ipynb
├── README.md
├── models/
│   ├── best_random_forest_model.pkl      # Saved best model
│   └── model_metadata.json                # Model metadata
├── results/
│   ├── sequential_results.csv            # Sequential training results
│   ├── parallel_results.csv              # Parallel training results
│   ├── detailed_comparison.csv           # Side-by-side comparison
│   └── comparison_summary.json           # Summary statistics
└── figures/
    └── additional_metrics_comparison.png  # Saved visualizations
```

## 🔧 Key Enhancements

### 1. Enhanced Training Functions

**Before**: Functions returned simple tuples `(n_estimators, mse)`

**After**: Functions return comprehensive dictionaries:

```python
{
    'n_estimators': int,
    'mse': float,
    'rmse': float,
    'mae': float,
    'r2': float,
    'training_time': float,
    'model': RandomForestRegressor  # (sequential only)
}
```

### 2. Improved Output Formatting

- Formatted print statements with clear sections
- Summary boxes with performance metrics
- Progress indicators for parallel execution

### 3. Data Organization

- Results stored in pandas DataFrames for easy manipulation
- Structured data export formats
- Organized file structure

## 📊 Performance Metrics

The notebook tracks and compares:

- **Training Time**: Wall-clock time for each model
- **Model Accuracy**: MSE, RMSE, MAE, R² scores
- **Speedup**: Ratio of sequential to parallel execution time
- **Resource Utilization**: Ray cluster resource information

### Expected Performance

On a typical multi-core system:

- **Sequential**: ~2 minutes (120 seconds)
- **Parallel**: ~30-35 seconds
- **Speedup**: ~3.5-4x faster

_Note: Actual performance depends on CPU cores, dataset size, and system load_

## 📈 Visualizations

The notebook generates multiple visualization types:

1. **Line Plots**: Show trends across different n_estimators values
2. **Bar Charts**: Compare total execution times
3. **Scatter Plots**: Actual vs predicted values and residuals
4. **Box Plots**: Distribution of metrics across all models

All plots are saved as high-resolution PNG files (300 DPI) in the `figures/` directory.

## 💾 Model Persistence

### Saving Models

The best model from parallel training is automatically saved:

- **Format**: Pickle (`.pkl`) using `joblib`
- **Location**: `models/best_random_forest_model.pkl`
- **Metadata**: `models/model_metadata.json`

### Loading Models

Example usage:

```python
import joblib

# Load model
model = joblib.load('models/best_random_forest_model.pkl')

# Make predictions
predictions = model.predict(X_test)
```

## 📤 Results Export

### CSV Files

1. **sequential_results.csv**: All sequential training results
2. **parallel_results.csv**: All parallel training results
3. **detailed_comparison.csv**: Side-by-side comparison with differences

### JSON Files

1. **comparison_summary.json**: Structured summary including:

   - Best models for each approach
   - Performance metrics
   - Speedup information
   - Time savings

2. **model_metadata.json**: Best model information:
   - Hyperparameters
   - Performance metrics
   - Training configuration

## 🎓 Learning Objectives

This notebook demonstrates:

1. **Distributed Computing**: How to use Ray for parallel task execution
2. **Object References**: Understanding Ray's object store and references
3. **Performance Optimization**: Measuring and comparing execution times
4. **Model Evaluation**: Comprehensive metrics for regression tasks
5. **Data Persistence**: Saving models and results for later use
6. **Visualization**: Creating informative plots for analysis

## 🔍 Key Concepts

### Ray Fundamentals

- **`@ray.remote`**: Decorator to make functions distributable
- **`.remote()`**: Method to call remote functions asynchronously
- **`ray.put()`**: Store objects in Ray's distributed object store
- **`ray.get()`**: Retrieve objects from object store (blocking)
- **`ray.init()`**: Initialize Ray cluster
- **`ray.shutdown()`**: Clean shutdown of Ray cluster

### Performance Analysis

- **Wall Time**: Actual elapsed time (what users experience)
- **CPU Time**: Total CPU time across all cores
- **Speedup**: Sequential time / Parallel time
- **Efficiency**: Speedup / Number of cores

## 🐛 Troubleshooting

### Common Issues

1. **Ray initialization fails**:

   - Check if Ray is properly installed: `pip install ray`
   - Ensure sufficient system resources

2. **Import errors**:

   - Install missing packages: `pip install <package_name>`
   - Restart Jupyter kernel after installation

3. **Memory issues**:

   - Reduce `NUM_MODELS` if running out of memory
   - Close other applications to free up RAM

4. **Slow parallel execution**:
   - Check number of available CPU cores
   - Ensure Ray is using multiple cores (check Ray dashboard)

## 📝 Notes

- The notebook uses a fixed random seed (201) for reproducibility
- Results may vary slightly between runs due to system load
- Parallel speedup depends on number of CPU cores available
- Model objects cannot be returned through Ray (serialization limitations)

## 🔗 Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray Dashboard](http://127.0.0.1:8265) (when Ray is running)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

## 📄 License

This notebook is provided for educational purposes as part of MLOps coursework.

## 👤 Author

Enhanced for MLOps Lab - Model Development with Ray

---

**Last Updated**: 2024
**Version**: 2.0 (Enhanced)
