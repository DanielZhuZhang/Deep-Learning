from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import shap
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("student-mat_modified.csv")
print(df.head())

df['Performance'] = df['Performance'].map({'Low': 0, 'Normal': 1, 'High': 2})

print(df.head())

X = df.drop('Performance', axis=1)

print(X.head())

categorical_cols = X.select_dtypes(include=['object', 'category']).columns

print("Categorical columns:", list(categorical_cols))

y = to_categorical(df['Performance'])

X = pd.get_dummies(X, columns=categorical_cols)

bool_cols = X.select_dtypes(include='bool').columns

X[bool_cols] = X[bool_cols].astype(int)

print(X.head())


n_features = X.shape[1]

model = models.Sequential(name="DeepNN", layers=[
    layers.Dense(name="h1", input_dim=n_features,
                 units=int(round((n_features + 1) / 2)),
                 activation='relu'),
    layers.Dropout(name="drop1", rate=0.2),

    layers.Dense(name="h2", units=int(round((n_features + 1) / 4)),
                 activation='relu'),
    layers.Dropout(name="drop2", rate=0.2),

    layers.Dense(name="h3", units=int(round((n_features + 1) / 8)),
                 activation='relu'),
    layers.Dropout(name="drop3", rate=0.2),

    layers.Dense(name="output", units=3 , activation='softmax')
])
model.summary()

# define metrics
def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy',F1])
print("First 5 rows of X:")
print(X[:5])

print("\nFirst 5 rows of y:")
print(y[:5])

model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

for layer in model.layers:
    weights = layer.get_weights()
    print(f"Layer {layer.name} weights:")
    for w in weights:
        print(w)
        print()
