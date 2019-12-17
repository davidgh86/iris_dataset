import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

style.use("ggplot")

iris_dataset = datasets.load_iris()

X = iris_dataset.data
y = iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)


