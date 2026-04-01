import numpy as np
import pandas as pd
import seaborn as s
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')



# =========================
# LOAD DATA
# =========================

data = pd.read_csv('data.csv')
df = data.copy()

# Encode target
df['diagnosis'] = df['diagnosis'].astype('category').cat.codes

# Features & Target
X = df.drop(['diagnosis','id'], axis=1)
Y = df['diagnosis']

# =========================
# SELECT TOP CORRELATED FEATURES
# =========================
corr = X.corr().abs()
top_features = corr.sum().sort_values(ascending=False).head(10).index

# =========================
# 1. HEATMAP (DISPLAY)
# =========================
plt.figure(figsize=(10,8))
s.heatmap(
    X[top_features].corr(),
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 9}
)
plt.title("Top Feature Correlation", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("heatmap_top.png", dpi=300)
print("Displaying Heatmap... Close window to continue.")
plt.show() # <--- This will pop up the window

# =========================
# 2. BOXPLOT (DISPLAY)
# =========================
features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean']
plt.figure(figsize=(15,8))
for i, feature in enumerate(features):
    plt.subplot(2,3,i+1)
    s.boxplot(x=df['diagnosis'], y=df[feature])
    plt.title(feature)
plt.tight_layout()
plt.savefig("boxplot.png")
print("Displaying Boxplots... Close window to continue.")
plt.show()

# =========================
# 3. DISTRIBUTION (DISPLAY)
# =========================
plt.figure(figsize=(8,5))
s.kdeplot(data=df, x="radius_mean", hue="diagnosis")
plt.title("Distribution of Radius Mean")
plt.savefig("distribution.png")
print("Displaying Distribution... Close window to continue.")
plt.show()

# =========================
# MODEL FUNCTION
# =========================
def FitModel(X, Y, name, model, params):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    pred = grid.predict(x_test)
    print(f"\n🔹 {name} Results")
    print("Best Params:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, pred))
    return grid

# TRAIN MODELS
svm_model = FitModel(X, Y, "SVM", SVC(), {'C':[1,10], 'gamma':[0.001,0.01]})
rf_model = FitModel(X, Y, "Random Forest", RandomForestClassifier(), {'n_estimators':[100,300]})
xgb_model = FitModel(X, Y, "XGBoost", XGBClassifier(), {'n_estimators':[100,300]})

# =========================
# SMOTE & FEATURE IMPORTANCE
# =========================
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X, Y)

rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_res, Y_res)

importances = rf.feature_importances_
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# =========================
# 4. FEATURE IMPORTANCE (DISPLAY)
# =========================
plt.figure(figsize=(10,8))
s.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
plt.title("Top 10 Important Features")
plt.savefig("feature_importance.png")
print("Displaying Feature Importance... Close window to finish.")
plt.show()

print("\n✅ PROCESS COMPLETE!")
