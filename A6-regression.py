from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Load entire dataset 
X, y = load_diabetes(return_X_y=True, as_frame=False)

#Make a train-test split with 20% of the data in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


### 1A ###
from sklearn.linear_model import LinearRegression

resultA = LinearRegression().fit(X_train, y_train)
resultA.coef_
resultA.intercept_

### 1B.i ###
predicted_train = resultA.predict(X_train)

sse_train = 0

for i in range(0, len(predicted_train)):
    sse_train = sse_train + (predicted_train[i]-y_train[i])**2

mse_train = sse_train/len(predicted_train)

### 1B.ii ###
# Get the predicted y values
predicted_test = resultA.predict(X_test)

sse_test = 0

# Add squared error for between actual and predicted for each x_test
for i in range(0, len(predicted_test)):
    sse_test = sse_test + (predicted_test[i]-y_test[i])**2

# Take mean of the error
mse_test = sse_test/len(predicted_test)


### 1C ###
from sklearn.preprocessing import PolynomialFeatures

# Get the degree 2 fit of the x data
fitted_X = PolynomialFeatures(degree=2).fit_transform(X_train)

# Use fitted data to build polynomial model
resultC = LinearRegression().fit(fitted_X,y_train)

# Make a list of all weights including w0
weights = [resultC.intercept_]
for w in resultC.coef_:
    weights.append(w)
    
print(weights)

### 1D.i ###
predicted_train = resultC.predict(fitted_X)

sse_train = 0

for i in range(0, len(predicted_train)):
    sse_train = sse_train + (predicted_train[i]-y_train[i])**2

mse_train = sse_train/len(predicted_train)


### 1D.ii ###
# Fit the test data to a polynomial
fitted_X_test = PolynomialFeatures(degree=2).fit_transform(X_test)

# Get predicted y values for test data
predicted_test = resultC.predict(fitted_X_test)

sse_test = 0

# Add squared error between actual and predicted for each x in test
for i in range(0, len(predicted_test)):
    sse_test = sse_test + (predicted_test[i]-y_test[i])**2

# Get the mean error for MSE
mse_test = sse_test/len(predicted_test)


### 1E.i ###
from sklearn.linear_model import Ridge

# Used to loop for all
alphas =  [0, 0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 10]

resultE = Ridge(alpha = 10).fit(fitted_X, y_train)

fitted_X_test = PolynomialFeatures(degree=2).fit_transform(X_test)

predicted_test = resultE.predict(fitted_X_test)

sse_test = 0

for i in range(0, len(predicted_test)):
    sse_test = sse_test + (predicted_test[i]-y_test[i])**2

mse_test = sse_test/len(predicted_test)


### 1E.ii ###
from matplotlib import pyplot as plt

errors = []

for alpha in alphas:
    resultE = Ridge(alpha = alpha).fit(fitted_X, y_train)

    predicted_test = resultE.predict(fitted_X_test)

    sse_test = 0

    for i in range(0, len(predicted_test)):
        sse_test = sse_test + (predicted_test[i]-y_test[i])**2

    mse_test = sse_test/len(predicted_test)
    errors.append(mse_test)

plt.figure(figsize=(12,8))
plt.xlabel("Value for Alpha")
plt.ylabel("Mean Squared Error Without Regularization Penalty")
plt.title("Figure 1A: Average MSE Without Regularization Penalty vs Alpha")
plt.plot(alphas, errors)
plt.show()

plt.figure(figsize=(12,8))
plt.xlabel("Log Value for Alpha")
plt.ylabel("Mean Squared Error Without Regularization Penalty")
plt.title("Figure 1B: Average MSE Without Regularization Penalty vs Log of Alpha")
plt.plot(alphas, errors)
plt.xscale("log")
plt.show()

#Make 5 train-test splits of the data
#Because we chose 5 folds, each test set will consist of 20% of the data 
n_splits = 5
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X)

errors = {}

for alpha in alphas:
    errors[str(alpha)] = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    fold_train_X = X[train_index]
    fold_train_y = y[train_index]
    fold_test_X = X[test_index]
    fold_test_y = y[test_index]
    fitted_X = PolynomialFeatures(degree = 2).fit_transform(fold_train_X)
    fitted_test = PolynomialFeatures(degree = 2).fit_transform(fold_test_X)
    for alpha in alphas:
        result = Ridge(alpha = alpha).fit(fitted_X, fold_train_y)
        predicted = result.predict(fitted_test)
        sse_test = 0
        for i in range(0, len(predicted)):
            sse_test = sse_test + (predicted[i]-fold_test_y[i])**2
        mse_test = sse_test/len(predicted)
        errors[str(alpha)].append(mse_test)
        
means = []

for alpha in alphas:
    error_list = errors[str(alpha)]
    means.append(sum(error_list)/len(error_list))
    
    
    
plt.figure(figsize=(12,8))
plt.xlabel("Value for Alpha")
plt.ylabel("Average Mean Squared Error for 5-Fold Validation")
plt.title("Figure 2A: Average MSE for 5-Fold Validation vs Alpha")
plt.plot(alphas, means)
plt.show()
    
    
    
    
    
    