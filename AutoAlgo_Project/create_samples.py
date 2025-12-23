import csv
import os

# 1. Create Iris Data (Classification)
iris_data = [
    ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    [5.1, 3.5, 1.4, 0.2, "setosa"],
    [4.9, 3.0, 1.4, 0.2, "setosa"],
    [4.7, 3.2, 1.3, 0.2, "setosa"],
    [7.0, 3.2, 4.7, 1.4, "versicolor"],
    [6.4, 3.2, 4.5, 1.5, "versicolor"],
    [6.9, 3.1, 4.9, 1.5, "versicolor"],
    [6.3, 3.3, 6.0, 2.5, "virginica"],
    [5.8, 2.7, 5.1, 1.9, "virginica"],
    [7.1, 3.0, 5.9, 2.1, "virginica"]
]

with open('iris_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(iris_data)
print("Created iris_test.csv")

# 2. Create Housing Data (Regression)
housing_data = [
    ["square_feet", "bedrooms", "bathrooms", "age_years", "price"],
    [1500, 3, 2, 10, 250000],
    [2000, 4, 3, 5, 350000],
    [1200, 2, 1, 20, 180000],
    [1800, 3, 2, 8, 300000],
    [2500, 4, 3, 2, 450000],
    [1600, 3, 2, 15, 275000],
    [1400, 2, 1.5, 12, 210000],
    [3000, 5, 4, 1, 600000],
    [2200, 4, 2.5, 6, 380000]
]

with open('housing_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(housing_data)
print("Created housing_test.csv")