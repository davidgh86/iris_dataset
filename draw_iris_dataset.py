from sklearn import datasets
import matplotlib.pyplot as plt

iris_dataset = datasets.load_iris()
# print(iris_dataset.target_names)
# print(iris_dataset.target)
# print(iris_dataset.data)
# print(iris_dataset.DESCR)

# indices to plot

fig, axs = plt.subplots(4, 4)

for x_index in range(4):
    for y_index in range(4):

        column_x = iris_dataset.data[:, x_index]
        column_x_label = iris_dataset.feature_names[x_index]
        column_y = iris_dataset.data[:, y_index]
        column_y_label = iris_dataset.feature_names[y_index]

        axs[x_index, y_index].scatter(column_x, column_y, c=iris_dataset.target)
plt.show()


