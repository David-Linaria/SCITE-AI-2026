import numpy as np
import pandas as pd

# X = np.array(
#     [
#         [5.1, 3.5, 1.4, 0.2],
#         [6.4, 3.2, 4.5, 1.5],
#         [6.3, 3.3, 6.0, 2.5],
#     ]
# )

# print("Shape:", X.shape)
# print("Dimensions:", X.ndim)
# print("Size:", X.size)
# print("Data type:", X.dtype)

# scores = np.array(
#     [
#         [80, 90, 70],
#         [85, 95, 90],
#         [60, 75, 80],
#     ]
# )

# average_students = np.mean(scores, 1)
# average_courses = np.mean(scores, 0)
# max_courses = np.max(scores, 0)
# max_course2 = max_courses[1]  # max_course2 = np.max(scores[:, 1])

# print(f'{average_students}')
# print(f'{average_courses}')
# print(f'{max_course2}')

# X = np.arange(12).reshape(3, 4)
# print(f'{X}')
# print(X[:, 2])

# scores = np.array(
#     [
#         [80, 90, 70],
#         [85, 95, 90],
#         [60, 75, 80],
#     ]
# )

# condition = np.mean(scores, 1) >= 85
# student_condition = scores[condition]

# print(f"{student_condition}")

# print(f'{np.shape([5,10])}')

# X = np.array(
#     [
#         [2, 10],
#         [4, 20],
#         [6, 30],
#     ],
#     dtype=float,
# )

# X_mean = np.mean(X, axis=0)
# X_std = np.std(X, axis=0)

# X_standardized = (X - X_mean) / X_std
# X_standardized_mean = np.mean(X_standardized, axis=0)

# condition = np.allclose(
#     X_standardized_mean, 0
# )  # or `condition = np.isclose(X_standardized_mean, 0)` which outputs a tuple of boolean values.

# print(X_standardized)
# print(condition)


df = pd.DataFrame(
    {
        "Name": ["Alice", "Bob", "Charlie"],
        "Math": [80, 85, 60],
        "English": [90, 95, 75],
        "Python": [70, 90, 80],
    },
    index=["S001", "S002", "S003"],
)

average = df.mean(axis=1, numeric_only=True)
df["Average"] = average

students_selected = df.loc[
    df["Average"] >= 85,
    ["Name", "Average"],
]

print(students_selected)
