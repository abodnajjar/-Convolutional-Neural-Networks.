import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score , roc_curve , auc


# Load the dataset
file_path = "C:/Users/abdal/OneDrive/Desktop/Customer Churn (1).csv"
df = pd.read_csv(file_path, names=['ID', 'Call Failure', 'Complains', 'Charge Amount','Freq. of use','Freq. of SMS ' 
 ,'Distinct Called Numbers','Age Group','Plan','Status','Age','Customer Value','Churn'])
df = df.dropna() 
print (df)
#print(tabulate(df, headers='keys', tablefmt='grid'))
df.columns = df.columns.str.strip()

numeric_columns = [
    'ID', 'Call Failure', 'Charge Amount', 'Freq. of use', 
    'Freq. of SMS', 'Distinct Called Numbers', 'Age Group','Age', 'Customer Value'
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric

    #You have to perform the following tasks:

    #1
print("Descriptive Statistics for Numeric Columns:")
print(df[numeric_columns].describe())
categorical_summary = df.describe(include=['object'])
print("\nCategorical Summary:")
print(categorical_summary)

#2
churn_distribution = df['Churn'].value_counts()

plt.figure(figsize=(8, 5), num="Churn Distribution Plot")
churn_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Churn')
plt.xlabel('Churn ')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#3
# Group by 'Age Group' and count occurrences of 'Churn'
churn_counts = df.groupby('Age Group')['Churn'].value_counts().unstack(fill_value=0)
# Create a bar chart with 'Yes' and 'No' as separate bars for each age group

churn_counts.plot(kind='bar',figsize=(10, 6), title='Churn Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Customers')
plt.legend(title='Churn Status')
plt.tight_layout()
plt.show()

#4
# Group by 'Charge Amount' and count occurrences of 'Churn'
churn_counts = df.groupby('Charge Amount')['Churn'].value_counts().unstack(fill_value=0)
# Create a bar chart with 'Yes' and 'No' as separate bars for each charge amount
churn_counts.plot(kind='bar', figsize=(10, 6), title='Churn Distribution by Charge Amount')
plt.xlabel('Charge Amount')
plt.ylabel('Number of Customers')
plt.legend(title='Churn Status')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

#5
charge_details = df[['ID', 'Charge Amount']]
print(charge_details)
charge_summary = df["Charge Amount"].describe()
print("Summary Statistics for Charge Amount:")
print(charge_summary)

#6
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is non-numeric
        df[column], _ = pd.factorize(df[column])
        #converts categorical data into numeric values by assigning a unique integer to each unique value in the column.
#'Plan': 'pre-paid' → 0, 'post-paid' → 1.
#'For boolean value "yes/no "': 'no' → 0, 'yes' → 1
# Status -> "active" → 0 ,"not-active" → 1
# Calculate the correlation matrix
correlation_matrix = df.corr()
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

######################################################################################################################

#Regression tasks:
#1

lr_model1 = LinearRegression()
X = df.drop(columns=['Customer Value'])  # Independent variables (drop target variable)
y = df['Customer Value']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_clean = X_train.dropna()
y_train_clean = y_train[X_train_clean.index]  # Keep y values corresponding to the dropped rows
X_test_clean = X_test.dropna()
y_test_clean = y_test[X_test_clean.index]

# Now fit the model
lr_model1.fit(X_train_clean, y_train_clean)
y_pred1 = lr_model1.predict(X_test_clean)

# Evaluate the model
mse1 = mean_squared_error(y_test_clean, y_pred1)
r2_1 = r2_score(y_test_clean, y_pred1)
print(f"Model LRM1 - Mean Squared Error: {mse1}")
print(f"Model LRM1 - R-squared: {r2_1}")

#2
# i seleted this feature becouse the highest correlation 
X1 = df[['Freq. of SMS']]  # Selecting the two features
y1 = df['Customer Value']  # Target variable is 'Customer Value'
lr_model2 = LinearRegression()
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
X1_train_clean = X1_train.dropna()
y1_train_clean = y1_train[X1_train_clean.index]  # Keep y values corresponding to the dropped rows
X1_test_clean = X1_test.dropna()
y1_test_clean = y1_test[X1_test_clean.index]
# Train the model
lr_model2.fit(X1_train_clean, y1_train_clean)
y_pred2 = lr_model2.predict(X1_test_clean )

# Evaluate the model's performance
mse2 = mean_squared_error(y1_test_clean, y_pred2)
r2_2 = r2_score(y1_test_clean, y_pred2)

# Print the evaluation metrics
print(f"Model LRM2 - Mean Squared Error: {mse2}")
print(f"Model LRM2 - R-squared: {r2_2}")

#3

X3 = df[['Customer Value', 'Freq. of SMS', 'Plan', 'Status']] 
# Define the target variable (dependent variable)
y3 = df['Churn']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=42)
# Step 2: Clean the training and testing data by dropping rows with NaN values
X3_train_clean = X3_train.dropna()
y3_train_clean = y3_train[X3_train_clean.index]  # Keep y values corresponding to the dropped rows
X3_test_clean = X3_test.dropna()
y3_test_clean = y3_test[X3_test_clean.index]  # Keep y values corresponding to the dropped rows
# Step 3: Apply Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X3_train_clean, y3_train_clean)
# Step 4: Make predictions using the cleaned test set
y3_pred = lr_model.predict(X3_test_clean)
# Step 5: Evaluate the model
mse = mean_squared_error(y3_test_clean, y3_pred)
r2 = r2_score(y3_test_clean, y3_pred)
# Print the results
print(f"Model LRM3_Mean Squared Error: {mse}")
print(f"Model LRM3_R-squared: {r2}")

#######################################################################################################################################
#Classification tasks:
#1
df = df.dropna()
X5 = df.drop(columns=['ID', 'Churn'])
y5 = df['Churn']

# Split the data
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X5_train = scaler.fit_transform(X5_train)
X5_test = scaler.transform(X5_test)

# Train the k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X5_train, y5_train)

# Predict and evaluate
y5_pred = knn.predict(X5_test)
print("Run k-Nearest Neighbours classifier to predict churn of customers")
print("Accuracy:", accuracy_score(y5_test, y5_pred))
print("Classification Report:\n", classification_report(y5_test, y5_pred))


#2
df = df.dropna()
# Features (X) and target variable (y)
X4 = df.drop(columns=['ID', 'Churn'])  # Drop columns not needed for prediction
y4 = df['Churn']  # Churn is the target variable
# Split the data into training and testing sets (70% training, 30% testing)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=42)
# Standardize the features using StandardScaler
scaler = StandardScaler()
X4_train = scaler.fit_transform(X4_train)
X4_test = scaler.transform(X4_test)
# Initialize the Naive Bayes classifier (GaussianNB is used for continuous features)
nb = GaussianNB()
# Train the model
nb.fit(X4_train, y4_train)
# Predict using the test set
y4_pred = nb.predict(X4_test)
# Evaluate the model
print("Run Naive Bayes classifier to predict churn of customers")
print("Accuracy:", accuracy_score(y4_test, y4_pred))
print("Classification Report:\n", classification_report(y4_test, y4_pred))


#3 
df = df.dropna()
# Features (X) and target variable (y)
X6 = df.drop(columns=['ID', 'Churn'])  # Drop columns not needed for prediction
y6 = df['Churn']  # Churn is the target variable
# Split the data into training and testing sets (70% training, 30% testing)
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.3, random_state=42)
# Standardize the features using StandardScaler
scaler = StandardScaler()
X6_train = scaler.fit_transform(X6_train)
X6_test = scaler.transform(X6_test)
# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
# Train the model
dt.fit(X6_train, y6_train)
# Predict using the test set
y6_pred = dt.predict(X6_test)
# Evaluate the model
print("Run Decision Tree classifier to predict churn of customers")
print("Accuracy:", accuracy_score(y6_test, y6_pred))
print("Classification Report:\n", classification_report(y6_test, y6_pred))

#4 
df = df.dropna()
# Features (X) and target variable (y)
X7 = df.drop(columns=['ID', 'Churn'])  # Drop columns not needed for prediction
y7 = df['Churn']  # Churn is the target variable
# Split the data into training and testing sets (70% training, 30% testing)
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.3, random_state=42)
# Standardize the features using StandardScaler
scaler = StandardScaler()
X7_train = scaler.fit_transform(X7_train)
X7_test = scaler.transform(X7_test)

log_reg = LogisticRegression()
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)

# Train models
log_reg.fit(X7_train, y7_train)
nb.fit(X7_train, y7_train)
knn.fit(X7_train, y7_train)

# Predict using the test set
y_pred_log_reg = log_reg.predict(X7_test)
y_pred_nb = nb.predict(X7_test)
y_pred_knn = knn.predict(X7_test)

# Calculate accuracy
accuracy_log_reg = accuracy_score(y7_test, y_pred_log_reg)
accuracy_nb = accuracy_score(y7_test, y_pred_nb)
accuracy_knn = accuracy_score(y7_test, y_pred_knn)

# Calculate classification reports
report_log_reg = classification_report(y7_test, y_pred_log_reg, output_dict=True)
report_nb = classification_report(y7_test, y_pred_nb, output_dict=True)
report_knn = classification_report(y7_test, y_pred_knn, output_dict=True)

# Calculate ROC-AUC
roc_auc_log_reg = roc_auc_score(y7_test, log_reg.predict_proba(X7_test)[:, 1])
roc_auc_nb = roc_auc_score(y7_test, nb.predict_proba(X7_test)[:, 1])
roc_auc_knn = roc_auc_score(y7_test, knn.predict_proba(X7_test)[:, 1])

# Confusion matrices
cm_log_reg = confusion_matrix(y7_test, y_pred_log_reg)
cm_nb = confusion_matrix(y7_test, y_pred_nb)
cm_knn = confusion_matrix(y7_test, y_pred_knn)

# Plot confusion matrices
fig, ax = plt.subplots(1, 3, figsize=(15, 5), num="Confusion matrices")

sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax[2])
ax[2].set_title("k-NN Confusion Matrix")

plt.show()
  

df = df.dropna()
X8 = df.drop(columns=['ID', 'Churn'])
y8 = df['Churn']

# Encode categorical features using one-hot encoding
X8 = pd.get_dummies(X8, drop_first=True)

# Split into training and test sets
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=0.2, random_state=42)
y8_test = y8_test.map({1: 0, 2: 1})
# Scale the data
scaler2 = StandardScaler()
X8_train_scaled = scaler2.fit_transform(X8_train)
X8_test_scaled = scaler2.transform(X8_test)

# Train classifiers
logreg2 = LogisticRegression(max_iter=1000)
nb2 = GaussianNB()
knn2 = KNeighborsClassifier()

logreg2.fit(X8_train_scaled, y8_train)
nb2.fit(X8_train_scaled, y8_train)
knn2.fit(X8_train_scaled, y8_train)

# Get predictions for ROC curve
y2_pred_logreg = logreg2.predict_proba(X8_test_scaled)[:, 1]
y2_pred_nb = nb2.predict_proba(X8_test_scaled)[:, 1]
y2_pred_knn = knn2.predict_proba(X8_test_scaled)[:, 1]

# Compute ROC curve and AUC for each classifier
fpr2_logreg, tpr2_logreg, _ = roc_curve(y8_test, y2_pred_logreg)
fpr2_nb, tpr2_nb, _ = roc_curve(y8_test, y2_pred_nb)
fpr2_knn, tpr2_knn, _ = roc_curve(y8_test, y2_pred_knn)

auc_logreg = auc(fpr2_logreg, tpr2_logreg)
auc_nb = auc(fpr2_nb, tpr2_nb)
auc_knn = auc(fpr2_knn, tpr2_knn)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8), num="ROC curve and AUC")
plt.plot(fpr2_logreg, tpr2_logreg, color='blue', lw=2, label='Logistic Regression (AUC = %0.2f)' % auc_logreg)
plt.plot(fpr2_nb, tpr2_nb, color='green', lw=2, label='Naive Bayes (AUC = %0.2f)' % auc_nb)
plt.plot(fpr2_knn, tpr2_knn, color='red', lw=2, label='kNN (AUC = %0.2f)' % auc_knn)

# Add labels and title
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()

# Print AUC scores for comparison
print(f'Logistic Regression AUC: {auc_logreg:.2f}')
print(f'Naive Bayes AUC: {auc_nb:.2f}')
print(f'kNN AUC: {auc_knn:.2f}')