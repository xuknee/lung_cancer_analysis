import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/Users/johnny/Desktop/Projects/Lung Cancer Project/lung cancer.csv')

# Select features and target variable
features = ['gender', 'cancer_stage', 'family_history', 'smoking_status', 'treatment_type']
# Encode categorical columns to numeric
for col in features:
    df[col] = df[col].astype('category').cat.codes
features.extend(['age', 'bmi', 'cholesterol_level', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer'])
# Keep only numeric columns
X = df[features]
y = df['survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Train a Random Forest model
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = feature_importances.nlargest(10)

# Print the top 10 features with their importance scores
print("The 10 most important factors for predicting lung cancer survival are:")
for feature, importance in top10.items():
    print(f"- {feature} (importance score: {importance:.3f})")

# Plot
plt.gcf().set_size_inches(10, 6)
top10.plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()  # Highest at the top
plt.show()
