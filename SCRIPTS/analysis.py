import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load dataset
df = pd.read_csv("data/churn.csv")

# show first rows
print(df.head())

# show dataset info
df.info()
#statistical summary
print(df.describe())
# missing values
print(df.isnull().sum())
# churn distribution
print(df["Exited"].value_counts())
sns.countplot(x="Exited", data=df)
plt.title("Customer Churn Distribution")
plt.show()
sns.countplot(x="Gender", hue="Exited", data=df)
plt.title("Churn by Gender")
plt.show()
sns.countplot(x="Geography", hue="Exited", data=df)
plt.title("Churn by Geography")
plt.show()
sns.boxplot(x="Exited", y="Age", data=df)
plt.title("Churn by Age")
plt.show()
plt.savefig("visuals/churn_distribution.png")
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Exited", axis=1)
y = df["Exited"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", figsize=(10,6))
plt.title("Feature Importance")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", figsize=(10,6))
plt.title("Feature Importance")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(model.feature_importances_, index=X.columns)
import joblib
joblib.dump(model, "churn_model.pkl")
importance.sort_values().plot(kind="barh", figsize=(10,6))
plt.title("Feature Importance")
plt.show()
importance.sort_values().plot(kind="barh", figsize=(10,6))
plt.title("Feature Importance")
plt.show()
import joblib
joblib.dump(model, "churn_model.pkl")
# Remove useless columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Convert categorical data
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))