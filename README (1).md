
# Machine Learning Models for Supply Chain Sales Prediction

## Overview

This repository contains code to implement and evaluate different machine learning models for predicting sales in a supply chain context. The models used include **Linear Regression**, **Decision Tree**, and **Random Forest**, and their performance is evaluated using metrics such as **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² (R-squared)**.

### Key Components:
1. **Data Preprocessing**: Prepares the dataset by encoding categorical variables and normalizing continuous features.
2. **Model Training and Evaluation**: Implements and trains Linear Regression, Decision Tree, and Random Forest models, and evaluates their performance on test data.
3. **Visualizations**: Generates figures comparing model performance and residuals.

## Files
- `model_comparison.py`: Contains all the code for model training, evaluation, and visualizations.
- `requirements.txt`: Lists all the Python packages required to run the code.
- `README.md`: This file, providing an overview of the repository.

## Code Description

### 1. Data Preprocessing
The data is preprocessed by encoding categorical variables (like shipping mode and customer feedback) and normalizing continuous features (like car prices).

### 2. Model Training and Evaluation
Three models are implemented:
- **Linear Regression**
- **Decision Tree**
- **Random Forest**

Each model is evaluated using the following metrics:
- **MAE**: Measures the average error between actual and predicted values.
- **RMSE**: Similar to MAE but penalizes larger errors more.
- **R²**: Measures how well the model explains the variance in the data.

### 3. Visualizations
The following visualizations are generated:
- **Actual vs. Predicted Sales (Linear Regression)**: Shows how well the Linear Regression model predicted sales.
- **Residual Distribution (Linear Regression)**: Displays the distribution of residuals for the Linear Regression model.
- **Model Comparison Table**: A table comparing the performance of Linear Regression, Decision Tree, and Random Forest models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/supply-chain-sales-prediction.git
   ```
2. Navigate to the directory:
   ```bash
   cd supply-chain-sales-prediction
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the `model_comparison.py` file to train the models and generate visualizations:

```bash
python model_comparison.py
```

## Results
The performance of the models is summarized as follows:

| Model             | MAE       | RMSE      | R²        |
|-------------------|-----------|-----------|-----------|
| Linear Regression | 0.250195  | 0.289483  | -0.013400 |
| Decision Tree     | 0.360898  | 0.426117  | -1.195796 |
| Random Forest     | 0.288490  | 0.337750  | -0.379508 |

## License
This project is licensed under the MIT License - see the LICENSE file for details.

