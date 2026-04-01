import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
import shap 
import os # Added to handle directory creation
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# --- DIRECTORY SETUP ---
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1️⃣ Load Data
data = pd.read_csv('data.csv')
df = data.copy() 
data['diagnosis'] = data['diagnosis'].astype('category').cat.codes

# 2️⃣ Prepare Features (x) and Target (y)
x = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# --- ADVANCED: FEATURE ENGINEERING ---
corr_matrix = x.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
x = x.drop(to_drop, axis=1)
print(f"✅ Advanced Step: Dropped {len(to_drop)} redundant features.")

# 3️⃣ SMOTE
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(x, y)

# 4️⃣ Split Data
x_train, x_test, y_train, y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=10)

model_names = []
accuracies = []
trained_models = {} 

# 5️⃣ Training Function
def FitModel(algo_name, algorithm, params):
    grid = GridSearchCV(algorithm, param_grid=params, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(x_train, y_train)
    preds = grid.predict(x_test)
    acc = accuracy_score(y_test, preds)

    pickle.dump(grid, open(algo_name + '.pkl', 'wb'))
    trained_models[algo_name] = grid
    model_names.append(algo_name)
    accuracies.append(acc)

    print(f"\nModel: {algo_name} | Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {algo_name}")
    plt.savefig(f"plots/confusion_{algo_name.lower()}.png")
    plt.show() 

# 6️⃣ Train Models
FitModel('RandomForest', RandomForestClassifier(), {'n_estimators': [500]})
FitModel('SVC', SVC(), {'C': [1], 'gamma': [0.001]})
FitModel('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'n_estimators': [500]})

# 7️⃣ Accuracy Comparison
plt.figure(figsize=(8,6))
plt.bar(model_names, accuracies, color=['skyblue', 'orange', 'green'])
plt.ylim(0.800, 1.000)
plt.title("Model Accuracy Comparison")
plt.savefig("plots/accuracy_comparison.png")
plt.show()

# =========================
# 🚀 LIVE WORKING MODEL DEMO
# =========================
print("\n" + "="*30)
print("🚀 LIVE MODEL PREDICTION DEMO")
print("="*30)

sample_idx = 10 
sample_input = x.iloc[[sample_idx]]
actual_label = "Malignant" if y.iloc[sample_idx] == 1 else "Benign"

prediction = trained_models['XGBoost'].predict(sample_input)
result_label = "Malignant" if prediction[0] == 1 else "Benign"

print(f"Testing Patient ID: {df.iloc[sample_idx]['id']}")
print(f"Model Prediction: {result_label} | Actual: {actual_label}")

# 🧠 EXPLAINABLE AI (SHAP)
# =========================
print("\n--- 🧠 EXPLAINABLE AI (SHAP) ---")
explainer = shap.TreeExplainer(trained_models['XGBoost'].best_estimator_)
shap_values = explainer.shap_values(x_test)

# PLOT 1: Beeswarm (The "Detailed" Plot)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, x_test, max_display=10, show=False)
plt.title("Detailed Feature Impact (Beeswarm)")
plt.savefig("plots/shap_beeswarm.png", bbox_inches='tight', dpi=300)
plt.show()

# PLOT 2: Bar Plot (The "Clean" Plot)
plt.figure(figsize=(12, 8))
shap.plots.bar(explainer(x_test), max_display=10, show=False)
plt.title("Cleaned Feature Importance (Bar)")
plt.savefig("plots/shap_bar_clean.png", bbox_inches='tight', dpi=300)
plt.show()

print("✅ All plots saved in the 'plots/' folder!")
