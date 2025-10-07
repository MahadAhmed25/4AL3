import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Linear_Regression():
 
    def __init__(self,x_:list,y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)

    def preprocess(self,):

        #normalize the values
        hmean = np.mean(self.input)
        hstd = np.std(self.input)
        x_train = (self.input - hmean)/hstd

        #arrange in matrix format
        X = np.column_stack((np.ones(len(x_train)),x_train))

        #normalize the values
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean)/gstd

        #arrange in matrix format
        Y = (np.column_stack(y_train)).T

        return X, Y

    def train(self, X, Y):
        #compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def predict(self, X_test,beta):
        #predict using beta
        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)

#  preprocess data

data = pd.read_csv("datasets/training_data.csv")

# Target = Age (Rings + 1.5)
y = data["Rings"].to_numpy() + 1.5  

# Features
features = ["Length", "Diameter", "Height",
            "Whole_weight", "Shucked_weight",
            "Viscera_weight", "Shell_weight"]

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# plot each feature

for feature in features:
    x = data[feature].to_numpy()
    lr_ols = Linear_Regression(x, y)
    X,Y = lr_ols.preprocess()
    beta = lr_ols.train(X,Y)
    Y_predict = lr_ols.predict(X,beta)
    X_ = X[...,1].ravel()

    error = rmse(Y, Y_predict)

    # plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X_, Y, alpha=0.5, label="Data")
    plt.plot(X_, Y_predict, color="red", label="OLS fit")
    plt.title(f"Abalone Age vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Age (years)")
    plt.legend()
    plt.show()

    # print beta values
    print(f"{feature}: [β0,β1] = {beta.ravel()}, error (RMSE) = {error:.4f}")
