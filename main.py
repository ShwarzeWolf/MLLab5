import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings('ignore')

chipsDataset = pd.read_csv("C://Users//1358365//PycharmProjects//lab3//datasets//chips.csv")
geyserDataset = pd.read_csv("C://Users//1358365//PycharmProjects//lab3//datasets//geyser.csv")

chipsDatasetP = chipsDataset[chipsDataset.className == "P"]
chipsDatasetN = chipsDataset[chipsDataset.className == "N"]

plt.scatter(chipsDatasetN['x'], chipsDatasetN['y'], color='green', marker='+')
plt.scatter(chipsDatasetP['x'], chipsDatasetP['y'], color='red', marker='+')
plt.title("dataset visualisation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

X = chipsDataset.drop(['className'], axis='columns')
Y = chipsDataset.className
Y = Y.replace('P', 0)
Y = Y.replace('N', 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

scores = []
iterations = []
for iteration in range(15):
    abc = AdaBoostClassifier(n_estimators=iteration + 1, learning_rate=1)
    model = abc.fit(X_train, Y_train)

    currentScore = model.score(X_test, Y_test)
    scores.append(currentScore)
    iterations.append(iteration)

    plot_decision_regions(X.to_numpy(), Y.to_numpy(),  clf=model, legend=2)
    plt.show()


plt.plot(iterations, scores)
plt.show()


geyserDatasetP = geyserDataset[geyserDataset.className == "P"]
geyserDatasetN = geyserDataset[geyserDataset.className == "N"]

plt.scatter(geyserDatasetN['x'], geyserDatasetN['y'], color='green', marker='+')
plt.scatter(geyserDatasetP['x'], geyserDatasetP['y'], color='red', marker='+')
plt.title("dataset visualisation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

X = geyserDataset.drop(['className'], axis='columns')
Y = geyserDataset.className
Y = Y.replace('P', 0)
Y = Y.replace('N', 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

scores = []
iterations = []
for iteration in range(15):
    abc = AdaBoostClassifier(n_estimators=iteration + 1, learning_rate=1)
    model = abc.fit(X_train, Y_train)

    currentScore = model.score(X_test, Y_test)
    scores.append(currentScore)
    iterations.append(iteration)

    plot_decision_regions(X.to_numpy(), Y.to_numpy(),  clf=model, legend=2)
    plt.show()


plt.plot(iterations, scores)
plt.show()