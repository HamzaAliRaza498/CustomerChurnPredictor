import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report

# Creating the DataFrame
df = {
    'Age': [30, 25, 35, 20, 40, 55, 32, 28],
    'Monthly_Charge': [50, 60, 80, 40, 100, 120, 70, 55],
    'Churn': [0, 1, 0, 1, 0, 1, 0, 1]
}
Data = pd.DataFrame(df)

# Features and target variable
X = Data[['Age', 'Monthly_Charge']]
y = Data['Churn']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Accuracy is:', accuracy)
print('Classification report is:\n', report)

# Getting user input
user_age = float(input('Please enter customer age: '))
user_MC = float(input('Please enter customer monthly charge: '))

# Making prediction for the user input
user_array = np.array([[user_age, user_MC]])
predict = model.predict(user_array)

if predict[0] == 0:
    print("The Customer is Likely to stay")
else:
    print("The Customer is at risk of churning")
