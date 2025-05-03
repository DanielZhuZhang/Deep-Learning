from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import shap
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("student-mat_modified.csv")
df['Performance'] = df['Performance'].map({'Low': 0, 'Normal': 1, 'High': 2})

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("Categorical columns:", list(categorical_cols))

df_encoded = pd.get_dummies(df, columns=categorical_cols)
print(df.head())

y_raw = df['Performance']
y = to_categorical(y_raw)

X = df_encoded.drop('Performance', axis=1)
mi = mutual_info_classif(X, y_raw)

y = to_categorical(df['Performance'])

mi_scores = pd.Series(mi, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
mi_scores.plot(kind='bar')
plt.title('Mutual Information Scores')
plt.ylabel('MI Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print features sorted by MI scores (highest to lowest)
print("\nFeatures ranked by Mutual Information Score (highest to lowest):\n")
for feature, score in mi_scores.items():
    print(f"{feature}: {score:.4f}")

top_features_list = list(mi_scores.index)
print(top_features_list)