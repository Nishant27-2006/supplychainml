import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Data preprocessing (encoding and normalization)
# Assuming columns: CarPrice, ShipMode, CustomerFeedback, Sales
data['ShipMode'] = data['ShipMode'].astype('category').cat.codes
data['CustomerFeedback'] = data['CustomerFeedback'].astype('category').cat.codes

# Splitting into features and target
X = data[['CarPrice', 'ShipMode', 'CustomerFeedback']]
y = data['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
y_pred_lr = linear_reg_model.predict(X_test)

# Decision Tree Model
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred_tree = decision_tree_model.predict(X_test)

# Random Forest Model
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Metrics for all models
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
mae = [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_tree), mean_absolute_error(y_test, y_pred_rf)]
rmse = [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_tree)), np.sqrt(mean_squared_error(y_test, y_pred_rf))]
r2 = [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_tree), r2_score(y_test, y_pred_rf)]

# Visualizing: Actual vs Predicted (Linear Regression)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Linear Regression: Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid(True)
plt.savefig('lr_actual_vs_predicted.png')

# Residuals Distribution (Linear Regression)
residuals = y_test - y_pred_lr
plt.figure(figsize=(6, 6))
plt.hist(residuals, bins=30, color='purple')
plt.title('Linear Regression: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('lr_residuals.png')

# Model Comparison Table
comparison = pd.DataFrame({'Model': models, 'MAE': mae, 'RMSE': rmse, 'R²': r2})
plt.figure(figsize=(8, 3))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table(ax, comparison, loc='center', cellLoc='center')
plt.savefig('model_comparison_table.png')
