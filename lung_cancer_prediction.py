import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/Users/johnny/Downloads/lung cancer.csv')

# Encode categorical columns
# Convert 'gender' to 0/1
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Convert 'treatment_type' to numbers
df['treatment_type'] = df['treatment_type'].astype('category').cat.codes

# Select features and target variable
# Keep only numeric columns
X = df.select_dtypes(include='number').drop(columns=['survived', 'id'], errors='ignore')
y = df['survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=0)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = feature_importances.nlargest(10)

# Print the top 10 features with their importance scores
print("The 10 most important factors for predicting lung cancer survival are:")
for feature, importance in top10.items():
    print(f"- {feature} (importance score: {importance:.3f})")

# Plot
top10.plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()  # Highest at the top
plt.show()