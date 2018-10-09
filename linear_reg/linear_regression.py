import sys

import numpy as np
import matplotlib.pyplot as plt

def load_data(datapath):
    """
    generate X and Y vectors from the datapath
    """
    X, Y = [], []
    with open(datapath, "r") as f:
        line = f.readline()
        #line = line[:-2] #remove \n
        key_names = line.split(',')
        line = f.readline()

        while line:
            line = line[:-2] #remove \n
            x_str, y_str = line.split(',')
            x, y = int(x_str), int(y_str)
            X.append(x)
            Y.append(y)
            line = f.readline()

    return X, Y

def normalize(X):
    std = np.std(X)
    mean = np.mean(X)
    X = [(x-std)/mean for x in X]
    return X

def train(X, Y, lr=0.001, plotting=True):
    m = len(X)
    theta = [0, 0] # Theta init
    for iteration in range(15000):
        grad_t0, grad_t1 = 0, 0
        for i in range(m):
            prediction = predict(X[i], theta)
            grad_t0 += (2.0/m)*(prediction - Y[i])
            grad_t1 += (2.0/m)*(prediction - Y[i])*X[i]
        if iteration%1000 == 0 and plotting:
            plt.scatter(X,Y)
            plt.title("Car prices / km")
            plt.xlabel("km normalized")
            plt.ylabel("price")
            plt.plot([-3, 3], [predict(-3, theta), predict(3, theta)], linestyle='-')
            plt.pause(1)
        theta[0] -= lr*grad_t0
        theta[1] -= lr*grad_t1
    plt.show()

    return theta


def predict(x, theta):
    prediction = theta[0] + x*theta[1]
    return prediction

if __name__ == "__main__":
    datapath = "data/data.csv"
    X, Y = load_data(datapath)
    std = np.std(X)
    mean = np.mean(X)
    X = normalize(X)

    print("TRAINING............")
    theta = train(X, Y, plotting=True)
    print("TRAINING COMPLETED !")
    in_ = True
    while in_ != "q":
        in_ = input("Number of km ? (enter q to exit)  :  ")
        try:
            x = int(in_)
        except ValueError:
            print("km must be an integer")
            continue

        x = (x-std)/mean
        print(theta)
        pred = predict(x, theta)
        print("Predicion: {} euros".format(int(pred)))
