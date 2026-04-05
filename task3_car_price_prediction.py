# =============================================================================
# TASK 3: Car Price Prediction with Machine Learning
# Dataset: https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("TASK 3: Car Price Prediction with Machine Learning")
print("=" * 60)


df = pd.read_csv("car_data.csv")

print("\n[STEP 1] Dataset Loaded Successfully")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn Data Types:\n{df.dtypes}")


print("\n[STEP 2] Exploratory Data Analysis")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nUnique Fuel Types: {df['Fuel_Type'].unique()}")
print(f"Unique Seller Types: {df['Selling_type'].unique()}")
print(f"Unique Transmissions: {df['Transmission'].unique()}")


print("\n[STEP 3] Data Preprocessing & Feature Engineering")

current_year = 2024
df["Car_Age"] = current_year - df["Year"]
print(f"  ✔ Created 'Car_Age' feature (current year - manufacture year)")


df.drop(columns=["Car_Name", "Year"], inplace=True)


le = LabelEncoder()
for col in ["Fuel_Type", "Selling_type", "Transmission"]:
    df[col] = le.fit_transform(df[col])
    print(f"  ✔ Label-encoded '{col}'")


X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

print(f"\n  Features used: {list(X.columns)}")
print(f"  Target: Selling_Price")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")


print("\n[STEP 4] Training Regression Models")

models = {
    "Linear Regression":      LinearRegression(),
    "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_scaled, y, cv=5, scoring="r2").mean()

    results[name] = {
        "model": model, "y_pred": y_pred,
        "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv
    }
    print(f"\n  → {name}")
    print(f"     MAE  = {mae:.4f} | RMSE = {rmse:.4f} | R² = {r2:.4f} | CV R² = {cv:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["R2"])
print(f"\n  ★ Best Model: {best_name} (R² = {results[best_name]['R2']:.4f})")


rf_model       = results["Random Forest"]["model"]
feature_names  = list(df.drop(columns=["Selling_Price"]).columns)
importances    = rf_model.feature_importances_
importance_df  = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df  = importance_df.sort_values("Importance", ascending=False)

print("\n[STEP 5] Feature Importances (Random Forest):")
print(importance_df.to_string(index=False))


print("\n[STEP 6] Generating Visualizations...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Task 3: Car Price Prediction — Model Evaluation", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = ["#4C72B0", "#DD8452", "#55A868"]


for idx, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, idx])
    ax.scatter(y_test, res["y_pred"], alpha=0.6, color=colors[idx], edgecolors="white", s=50)
    mn = min(y_test.min(), res["y_pred"].min())
    mx = max(y_test.max(), res["y_pred"].max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_title(f"{name}\nActual vs Predicted", fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual Price (Lakhs)")
    ax.set_ylabel("Predicted Price (Lakhs)")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.92, f"R² = {res['R2']:.3f}", transform=ax.transAxes,
            fontsize=9, color="darkred", fontweight="bold")

ax2 = fig.add_subplot(gs[1, :2])
model_names = list(results.keys())
r2_scores   = [results[n]["R2"]   for n in model_names]
cv_scores   = [results[n]["CV_R2"] for n in model_names]
x = np.arange(len(model_names))
w = 0.35
bars1 = ax2.bar(x - w/2, r2_scores, w, label="Test R²",    color="#4C72B0", alpha=0.85)
bars2 = ax2.bar(x + w/2, cv_scores,  w, label="CV R² (5-fold)", color="#DD8452", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.set_title("Model Comparison: R² Scores", fontweight="bold")
ax2.set_ylabel("R² Score")
ax2.legend()
for bar in bars1: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", fontsize=8)
for bar in bars2: ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", fontsize=8)


ax3 = fig.add_subplot(gs[1, 2])
bars = ax3.barh(importance_df["Feature"], importance_df["Importance"], color="#55A868", alpha=0.85)
ax3.set_title("Feature Importances\n(Random Forest)", fontweight="bold")
ax3.set_xlabel("Importance Score")
ax3.invert_yaxis()
for bar in bars:
    ax3.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():.3f}", va="center", fontsize=8)

ax4 = fig.add_subplot(gs[2, 0])
best_res = results[best_name]
residuals = y_test.values - best_res["y_pred"]
ax4.scatter(best_res["y_pred"], residuals, alpha=0.6, color="#9B59B6", edgecolors="white", s=50)
ax4.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax4.set_title(f"Residual Plot\n({best_name})", fontweight="bold")
ax4.set_xlabel("Predicted Price")
ax4.set_ylabel("Residuals")


ax5 = fig.add_subplot(gs[2, 1])
mae_scores  = [results[n]["MAE"]  for n in model_names]
rmse_scores = [results[n]["RMSE"] for n in model_names]
ax5.bar(x - w/2, mae_scores,  w, label="MAE",  color="#E74C3C", alpha=0.85)
ax5.bar(x + w/2, rmse_scores, w, label="RMSE", color="#F39C12", alpha=0.85)
ax5.set_xticks(x)
ax5.set_xticklabels(model_names, fontsize=9)
ax5.set_title("MAE & RMSE Comparison\n(Lower = Better)", fontweight="bold")
ax5.set_ylabel("Error (Lakhs)")
ax5.legend()

ax6 = fig.add_subplot(gs[2, 2])
ax6.hist(y, bins=30, color="#3498DB", alpha=0.8, edgecolor="white")
ax6.set_title("Selling Price Distribution", fontweight="bold")
ax6.set_xlabel("Price (Lakhs)")
ax6.set_ylabel("Frequency")

plt.savefig("task3_car_price_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔ Plot saved as 'task3_car_price_results.png'")


print("\n[STEP 7] Sample Prediction with Best Model")
sample = pd.DataFrame({
    "Present_Price": [6.5],
    "Driven_kms":    [40000],
    "Fuel_Type":     [0],        
    "Selling_type":   [0],        
    "Transmission":  [0],        
    "Owner":         [0],
    "Car_Age":       [5],
})
sample_scaled   = scaler.transform(sample)
predicted_price = results[best_name]["model"].predict(sample_scaled)[0]
print(f"  Input → Present Price: ₹6.5L | 40,000 km | 5 years old")
print(f"  Predicted Selling Price: ₹{predicted_price:.2f} Lakhs")

print("\n" + "=" * 60)
print("TASK 3 COMPLETE ✓")
print("=" * 60)
