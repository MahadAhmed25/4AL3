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

# import data
data = pd.read_csv("datasets/gdp-vs-happiness.csv")

# Preprocess/clean data 

# year 2018
by_year = (data[data['Year']==2018])
# drop columns that will not be used
dropped = by_year.drop(columns=["World regions according to OWID","Code"])
# remove columns with missing values
df = dropped[
    (dropped['Cantril ladder score'].notna()) & 
    (dropped['GDP per capita, PPP (constant 2021 international $)']).notna()
]

#create np.array for gdp and happiness where happiness score is above 4.5
happiness=[]
gdp=[]
filtered = df[df['Cantril ladder score']>4.5]
happiness = filtered['Cantril ladder score'].to_numpy()
gdp = filtered['GDP per capita, PPP (constant 2021 international $)'].to_numpy()

x_mean = np.mean(gdp)
x_std = np.std(gdp)
y_mean = np.mean(happiness)
y_std = np.std(happiness)

x_norm = (gdp - x_mean) / x_std
y_norm = (happiness - y_mean) / y_std

X = np.column_stack((np.ones(len(x_norm)), x_norm))
Y = y_norm.reshape(-1, 1)

beta_ols = np.linalg.inv(X.T @ X) @ X.T @ Y

def gradient_descent(X, Y, lr, epochs):
    m, n = X.shape
    beta = np.zeros((n,1))
    for epoch in range(epochs):
        Y_pred = X @ beta
        error = Y_pred - Y 
        grad = (2/m) * (X.T @ error)
        beta -= lr * grad
    return beta

LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.5]
EPOCHS = [100, 500, 1000, 2000, 5000]

all_betas = []
for lr in LEARNING_RATES:
    for epoch in EPOCHS:
        beta_gd = gradient_descent(X, Y, lr, epoch)
        all_betas.append((lr, epoch, beta_gd))

# Graph 1: Multiple GD lines
plt.figure(figsize=(12, 7))
plt.scatter(x_norm, y_norm, label="Data")

# ====== Ai generated see readme =====
colors = plt.cm.viridis(np.linspace(0, 1, len(all_betas)))
for (lr, ep, beta), c in zip(all_betas, colors):
    y_hat = X @ beta
    plt.plot(x_norm, y_hat, color=c, alpha=0.6,
             label=f"lr={lr}, ep={ep}")
# ====== Ai generated ===============

plt.title("Gradient Descent Regression Lines")
plt.xlabel("GDP (normalized)")
plt.ylabel("Happiness (normalized)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

# ====== Ai generated see readme =====
# Pick best GD run (closest to OLS)
best_run = min(all_betas,
               key=lambda tup: np.linalg.norm(tup[2] - beta_ols))
best_lr, best_ep, best_beta = best_run
# ====== Ai generated =====

# Graph 2: OLS vs Best GD
plt.figure(figsize=(12, 7))
plt.scatter(x_norm, y_norm, label="Data")

y_hat_best = X @ best_beta

# OLS
lr_ols = Linear_Regression(gdp, happiness) 
X,Y = lr_ols.preprocess()
beta = lr_ols.train(X,Y)
Y_predict = lr_ols.predict(X,beta)
X_ = X[...,1].ravel()

plt.plot(X_,Y_predict,color='r', label="OLS")
plt.plot(x_norm, y_hat_best, color="blue",
         label=f"Best GD (lr={best_lr}, ep={best_ep})", linestyle='dashed')

plt.title("OLS vs Best Gradient Descent")
plt.xlabel("GDP (normalized)")
plt.ylabel("Happiness (normalized)")
plt.legend()
plt.show()

print("OLS β:", beta_ols.ravel())
for lr, ep, beta in all_betas:
    print(f"GD β (lr={lr}, ep={ep}):", beta.ravel())
print(f"\nBest GD β: {best_beta.ravel()} with lr={best_lr}, epochs={best_ep}")