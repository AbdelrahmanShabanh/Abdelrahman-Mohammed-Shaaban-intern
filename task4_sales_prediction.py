
# TASK 4: Sales Prediction using Python
# Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("TASK 4: Sales Prediction using Python")
print("=" * 60)


df = pd.read_csv("advertising.csv")

# Drop unnamed index column if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\n[STEP 1] Dataset Loaded Successfully")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:\n{df.head()}")


print("\n[STEP 2] Exploratory Data Analysis")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nCorrelation with Sales:")
print(df.corr()["Sales"].sort_values(ascending=False))


print("\n[STEP 3] Data Cleaning, Transformation & Feature Selection")


dupes = df.duplicated().sum()
print(f"  Duplicate rows: {dupes} → {'removed' if dupes > 0 else 'none found'}")
df.drop_duplicates(inplace=True)

df["TV_Radio"]       = df["TV"] * df["Radio"]           
df["Total_Spend"]    = df["TV"] + df["Radio"] + df["Newspaper"]
df["Digital_Spend"]  = df["TV"] + df["Radio"]          
df["TV_sq"]          = df["TV"] ** 2                 

print("  ✔ Created interaction feature: TV × Radio")
print("  ✔ Created Total_Spend, Digital_Spend, TV_sq")


feature_sets = {
    "Original (TV, Radio, Newspaper)": ["TV", "Radio", "Newspaper"],
    "Engineered Features":             ["TV", "Radio", "Newspaper", "TV_Radio",
                                        "Total_Spend", "Digital_Spend", "TV_sq"],
}


X = df[feature_sets["Engineered Features"]]
y = df["Sales"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n  Training samples: {len(X_train)}")
print(f"  Testing  samples: {len(X_test)}")


print("\n[STEP 4] Training Regression Models for Sales Forecasting")

models = {
    "Linear Regression":    LinearRegression(),
    "Ridge Regression":     Ridge(alpha=1.0),
    "Lasso Regression":     Lasso(alpha=0.1),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
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
    print(f"     MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | CV R²={cv:.4f}")

best_name = max(results, key=lambda k: results[k]["R2"])
print(f"\n  ★ Best Model: {best_name} (R² = {results[best_name]['R2']:.4f})")


print("\n[STEP 5] Business Insights — Advertising Impact on Sales")


X_orig = df[["TV", "Radio", "Newspaper"]]
lr_orig = LinearRegression()
lr_orig.fit(X_orig, y)
coef_df = pd.DataFrame({
    "Platform":    ["TV", "Radio", "Newspaper"],
    "Coefficient": lr_orig.coef_
}).sort_values("Coefficient", ascending=False)

print("\n  Linear Regression Coefficients (Original Features):")
print("  (Every $1 increase in spend → how much does Sales increase?)")
for _, row in coef_df.iterrows():
    print(f"    {row['Platform']:12s}: +{row['Coefficient']:.4f} units of Sales per $1 spent")
print(f"    Intercept:      {lr_orig.intercept_:.4f}")

rf_model      = results["Random Forest"]["model"]
feat_names    = feature_sets["Engineered Features"]
importance_df = pd.DataFrame({
    "Feature":    feat_names,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n  Feature Importances (Random Forest):")
print(importance_df.to_string(index=False))


print("\n[STEP 6] Generating Visualizations...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Task 4: Sales Prediction — Advertising Impact Analysis",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


platforms = ["TV", "Radio", "Newspaper"]
plat_colors = ["#1976D2", "#388E3C", "#F57C00"]
for i, (plat, col) in enumerate(zip(platforms, plat_colors)):
    ax = fig.add_subplot(gs[0, i])
    ax.scatter(df[plat], df["Sales"], alpha=0.6, color=col, edgecolors="white", s=50)
    # Regression line
    m, b = np.polyfit(df[plat], df["Sales"], 1)
    xline = np.linspace(df[plat].min(), df[plat].max(), 100)
    ax.plot(xline, m * xline + b, "r--", linewidth=2, label=f"slope={m:.3f}")
    ax.set_title(f"{plat} Spend vs Sales", fontweight="bold")
    ax.set_xlabel(f"{plat} Spend ($k)")
    ax.set_ylabel("Sales (units)")
    ax.legend(fontsize=8)


ax2 = fig.add_subplot(gs[1, 0])
best = results[best_name]
ax2.scatter(y_test, best["y_pred"], alpha=0.7, color="#2196F3", edgecolors="white", s=60)
mn = min(y_test.min(), best["y_pred"].min())
mx = max(y_test.max(), best["y_pred"].max())
ax2.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Fit")
ax2.set_title(f"Actual vs Predicted\n({best_name})", fontweight="bold")
ax2.set_xlabel("Actual Sales")
ax2.set_ylabel("Predicted Sales")
ax2.text(0.05, 0.92, f"R² = {best['R2']:.3f}", transform=ax2.transAxes,
         fontsize=9, color="darkred", fontweight="bold")
ax2.legend(fontsize=8)


ax3 = fig.add_subplot(gs[1, 1])
names  = list(results.keys())
r2s    = [results[n]["R2"] for n in names]
short_names = ["LinReg", "Ridge", "Lasso", "RF", "GBM"]
bar3 = ax3.bar(short_names, r2s, color=palette, alpha=0.85, edgecolor="white")
ax3.set_title("R² Score by Model", fontweight="bold")
ax3.set_ylabel("R² Score")
ax3.set_ylim(0, 1.05)
for bar, val in zip(bar3, r2s):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")


ax4 = fig.add_subplot(gs[1, 2])
bars = ax4.barh(importance_df["Feature"], importance_df["Importance"],
                color="#9C27B0", alpha=0.85)
ax4.set_title("Feature Importances\n(Random Forest)", fontweight="bold")
ax4.set_xlabel("Importance Score")
ax4.invert_yaxis()
for bar in bars:
    ax4.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():.3f}", va="center", fontsize=8)


ax5 = fig.add_subplot(gs[2, 0])
residuals = y_test.values - best["y_pred"]
ax5.scatter(best["y_pred"], residuals, alpha=0.6, color="#FF5722", edgecolors="white", s=50)
ax5.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax5.set_title(f"Residuals Plot\n({best_name})", fontweight="bold")
ax5.set_xlabel("Predicted Sales")
ax5.set_ylabel("Residuals")


ax6 = fig.add_subplot(gs[2, 1])
bar_colors = ["#1976D2" if c > 0 else "#D32F2F" for c in coef_df["Coefficient"]]
ax6.bar(coef_df["Platform"], coef_df["Coefficient"], color=bar_colors, alpha=0.85, edgecolor="white")
ax6.set_title("Ad Platform Impact on Sales\n(Linear Regression Coefficients)", fontweight="bold")
ax6.set_ylabel("Sales Units per $1 Spend")
ax6.axhline(0, color="black", linewidth=0.8)
for i, (_, row) in enumerate(coef_df.iterrows()):
    ax6.text(i, row["Coefficient"]+0.002, f"{row['Coefficient']:.4f}",
             ha="center", fontsize=9, fontweight="bold")


ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(y, bins=25, color="#00BCD4", alpha=0.85, edgecolor="white")
ax7.axvline(y.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean={y.mean():.1f}")
ax7.set_title("Sales Distribution", fontweight="bold")
ax7.set_xlabel("Sales (units)")
ax7.set_ylabel("Frequency")
ax7.legend(fontsize=8)

plt.savefig("task4_sales_prediction_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔ Plot saved as 'task4_sales_prediction_results.png'")


print("\n[STEP 7] Actionable Business Insights")
print("─" * 50)

tv_coef   = coef_df[coef_df["Platform"] == "TV"]["Coefficient"].values[0]
rad_coef  = coef_df[coef_df["Platform"] == "Radio"]["Coefficient"].values[0]
news_coef = coef_df[coef_df["Platform"] == "Newspaper"]["Coefficient"].values[0]
best_plat = coef_df.iloc[0]["Platform"]

print(f"""
  📺 TV Advertising:
     Every $1k increase → +{tv_coef:.4f} units in Sales
     → {'Strong positive impact. Scale TV budget.' if tv_coef > 0.04 else 'Moderate impact.'}

  📻 Radio Advertising:
     Every $1k increase → +{rad_coef:.4f} units in Sales
     → {'High ROI channel — prioritize for budget allocation.' if rad_coef > 0.04 else 'Moderate contribution.'}

  📰 Newspaper Advertising:
     Every $1k increase → +{news_coef:.4f} units in Sales
     → {'Low ROI — consider reallocating budget to TV/Radio.' if news_coef < 0.01 else 'Moderate impact.'}

  🏆 Highest Impact Platform: {best_plat}
  💡 Recommendation: Reallocate spend from low-impact channels to
     high-performing platforms (TV & Radio) to maximize Sales.
""")

print("[STEP 8] Sample Sales Forecast")
sample_data = {
    "TV": [150.0], "Radio": [30.0], "Newspaper": [20.0]
}
sd = pd.DataFrame(sample_data)
sd["TV_Radio"]      = sd["TV"]  * sd["Radio"]
sd["Total_Spend"]   = sd["TV"]  + sd["Radio"] + sd["Newspaper"]
sd["Digital_Spend"] = sd["TV"]  + sd["Radio"]
sd["TV_sq"]         = sd["TV"]  ** 2

sd_scaled       = scaler.transform(sd[feature_sets["Engineered Features"]])
predicted_sales = results[best_name]["model"].predict(sd_scaled)[0]

print(f"  Input → TV: $150k | Radio: $30k | Newspaper: $20k")
print(f"  Predicted Sales: {predicted_sales:.2f} units")

print("\n" + "=" * 60)
print("TASK 4 COMPLETE ✓")
print("=" * 60)
