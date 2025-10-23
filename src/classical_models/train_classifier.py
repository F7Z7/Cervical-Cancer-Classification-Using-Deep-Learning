import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

train_features = np.load(
    'C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\features\\resnet152_train_features.npz')
test_features = np.load(
    'C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\features\\resnet152_test_features.npz')

X_train, y_train = train_features['X'], train_features['y']
X_test, y_test = test_features['X'], test_features['y']

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=3000,
    C=1.0,
    solver='lbfgs',
    class_weight='balanced'
)
logistic_model.fit(X_train_scaled, y_train)

y_pred = logistic_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Abnormal", "Normal"],
            yticklabels=["Abnormal", "Normal"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()

import os

os.makedirs("models", exist_ok=True)
joblib.dump(logistic_model, "models/logistic_model.joblib")
joblib.dump(scalar, "models/scalar.joblib")
