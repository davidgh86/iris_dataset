from sklearn import datasets
import matplotlib.pyplot as plt

iris_dataset = datasets.load_iris()
# print(iris_dataset.target_names)
# print(iris_dataset.target)
# print(iris_dataset.data)
# print(iris_dataset.DESCR)

# indices to plot
x_index = 0
y_index = 1

column_x = iris_dataset.data[:, x_index]
column_x_label = iris_dataset.feature_names[x_index]
column_y = iris_dataset.data[:, y_index]
column_y_label = iris_dataset.feature_names[y_index]

plt.figure(figsize=(5, 4))
plt.scatter(column_x, column_y, c=iris_dataset.target)
plt.xlabel(column_x_label)
plt.ylabel(column_y_label)
plt.show()


