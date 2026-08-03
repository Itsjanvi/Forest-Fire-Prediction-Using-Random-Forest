import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("dataset/Algerian_forest_fires_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Remove spaces from string values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

print("Dataset Loaded Successfully!")

# ==========================
# TARGET COLUMN
# ==========================

df["Classes"] = df["Classes"].replace({
    "fire": 1,
    "not fire": 0,
    "1": 1,
    "0": 0
})

df["Classes"] = pd.to_numeric(df["Classes"], errors="coerce")

# ==========================
# FEATURES
# ==========================

X = df.drop("Classes", axis=1)

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remove invalid rows
data = pd.concat([X, df["Classes"]], axis=1)
data = data.dropna()

X = data.drop("Classes", axis=1)
y = data["Classes"].astype(int)

print("\nDataset Shape :", data.shape)
print("\nClasses:")
print(y.value_counts())

# ==========================
# EDA
# ==========================

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.title("Fire vs Not Fire")
plt.savefig("fire_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)

# ==========================
# MODEL
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X_test)

# ==========================
# RESULTS
# ==========================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ==========================
# SAVE MODEL
# ==========================

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel Saved Successfully!")
# ==========================
# SAVE OUTPUT CSV
# ==========================

output = X_test.copy()

output["Actual"] = y_test.values
output["Predicted"] = y_pred

output.to_csv("Forest_Fire_Output.csv", index=False)

print("\nPrediction CSV Saved Successfully!")

# ==========================
# SAMPLE PREDICTION
# ==========================

sample = X.iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\n🔥 Forest Fire Predicted")
else:
    print("\n🌳 No Forest Fire Predicted")