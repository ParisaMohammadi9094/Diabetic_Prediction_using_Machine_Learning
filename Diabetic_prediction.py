# Diabetes Progression Prediction: ML & Deep Learning Comparison
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# 1. Load Dataset
# ---------------------------
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

print("Dataset shape:", X.shape)
print("Target range:", y.min(), "-", y.max())

# ---------------------------
# 2. Train/Test Split & Scaling
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep unscaled for tree-based models
X_train_raw = X_train.values
X_test_raw = X_test.values

# ---------------------------
# 3. Define Models
# ---------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# ---------------------------
# 4. Train & Evaluate
# ---------------------------
results = []

for name, model in models.items():
    print(f"Training {name}...")
    
    # Use scaled data for distance-based models, raw for tree-based
    if name in ['KNN Regressor', 'SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_raw, y_train)
        y_pred = model.predict(X_test_raw)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation (CV) on scaled data for fairness
    if name in ['KNN Regressor', 'SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train_raw, y_train, cv=5, scoring='r2')
    
    results.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std()
    })

# ---------------------------
# 5. Deep Learning (ANN)
# ---------------------------
print("\nTraining Deep Learning Model (ANN)...")

# Build model
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train
history = ann_model.fit(X_train_scaled, y_train, 
                        validation_split=0.2, 
                        epochs=200, 
                        batch_size=16, 
                        callbacks=[early_stop], 
                        verbose=0)

# Predict & evaluate
y_pred_ann = ann_model.predict(X_test_scaled).flatten()
mse_ann = mean_squared_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mse_ann)
mae_ann = mean_absolute_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)

# Cross-validation for ANN (manual)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = []
for train_idx, val_idx in kf.split(X_train_scaled):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    temp_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2), Dense(64, activation='relu'), Dropout(0.2),
        Dense(32, activation='relu'), Dense(1)
    ])
    temp_model.compile(optimizer='adam', loss='mse')
    temp_model.fit(X_tr, y_tr, epochs=50, batch_size=16, verbose=0)
    cv_r2_scores.append(r2_score(y_val, temp_model.predict(X_val).flatten()))

results.append({
    'Model': 'Deep Learning (ANN)',
    'MSE': mse_ann,
    'RMSE': rmse_ann,
    'MAE': mae_ann,
    'R2': r2_ann,
    'CV_R2_Mean': np.mean(cv_r2_scores),
    'CV_R2_Std': np.std(cv_r2_scores)
})

# ---------------------------
# 6. Results DataFrame & Sorting
# ---------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R2', ascending=False).round(4)

print("\n" + "="*80)
print("FINAL COMPARISON TABLE (Sorted by R2 Score)")
print("="*80)
print(results_df.to_string(index=False))

# ---------------------------
# 7. Visualization
# ---------------------------
# Bar plot of R2 scores
plt.figure(figsize=(12, 6))
bars = plt.barh(results_df['Model'], results_df['R2'], color='skyblue')
plt.xlabel('R2 Score')
plt.title('Model Comparison: R2 Score (Higher is Better)')
plt.gca().invert_yaxis()
for bar, r2_val in zip(bars, results_df['R2']):
    plt.text(bar.get_width() - 0.03, bar.get_y() + bar.get_height()/2, f'{r2_val:.3f}', 
             va='center', ha='right', fontsize=9, color='black')
plt.tight_layout()
plt.show()

# Performance comparison across metrics
metrics_melt = results_df.melt(id_vars='Model', value_vars=['MSE', 'MAE', 'RMSE'], 
                               var_name='Metric', value_name='Value')
plt.figure(figsize=(14, 6))
sns.barplot(data=metrics_melt, x='Model', y='Value', hue='Metric')
plt.xticks(rotation=45, ha='right')
plt.title('Error Metrics Comparison (Lower is Better)')
plt.tight_layout()
plt.show()

# Actual vs Predicted for best model (Gradient Boosting / XGBoost / ANN)
best_model_name = results_df.iloc[0]['Model']
print(f"\nBest model based on R2: {best_model_name}")

# Retrain best model for plotting
if best_model_name in ['KNN Regressor', 'SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
    best_clf = models[best_model_name]
    best_clf.fit(X_train_scaled, y_train)
    y_pred_best = best_clf.predict(X_test_scaled)
else:
    best_clf = models[best_model_name]
    best_clf.fit(X_train_raw, y_train)
    y_pred_best = best_clf.predict(X_test_raw)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Best Model: {best_model_name} (R2={results_df.iloc[0]["R2"]:.3f})')
plt.show()

# ---------------------------
# 8. Save results to CSV
# ---------------------------
results_df.to_csv('diabetes_model_comparison.csv', index=False)
print("\nComparison saved to 'diabetes_model_comparison.csv'")
