# Import necessary libraries
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Fetching the dataset
mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']

# Display a sample digit
some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # reshape to plot

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis("off")
plt.show()

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Shuffle the training set
shuffle_index = np.random.permutation(len(x_train))
x_train, y_train = x_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

# Converting target values to integer
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# Creating a binary target for detecting '2's
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

# Train a logistic regression classifier
clf = LogisticRegression(tol=0.1, max_iter=1000, solver='lbfgs')
clf.fit(x_train, y_train_2)

# Predict an example digit
example = clf.predict([some_digit])
print("Prediction for the example digit:", example)

# Cross-validation
cv_scores = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print("Cross-validation accuracy:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# Model evaluation on the test set
y_test_pred = clf.predict(x_test)
precision = precision_score(y_test_2, y_test_pred)
recall = recall_score(y_test_2, y_test_pred)
f1 = f1_score(y_test_2, y_test_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test_2, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Visualize cross-validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', label="Cross-validation accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-validation Accuracy per Fold")
plt.legend()
plt.grid()
plt.show()
