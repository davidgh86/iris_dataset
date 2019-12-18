from matplotlib import style
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

style.use("ggplot")

iris_dataset = datasets.load_iris()

X = iris_dataset.data
y = iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)
print(clf.coef_)
print(clf.score(X, y))
print(clf.score(X_test, y_test))

#for i in range(len(iris_dataset.data)):
#    print(clf.predict([iris_dataset.data[i]]), iris_dataset.target[i])
#print(clf.predict([[5.4, 3.9, 1.7, 0.4]))


