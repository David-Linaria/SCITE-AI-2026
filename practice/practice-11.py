import math

import numpy as np


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)

    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]

        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for label in labelCounts:
        prob = labelCounts[label] / numEntries
        shannonEnt -= prob * math.log2(prob)

    return shannonEnt


dataSet = [
    [1, 0, "Yes"],
    [1, 1, "Yes"],
    [0, 0, "No"],
    [0, 1, "No"],
]

result = calcShannonEnt(dataSet)

print(np.isclose(result, 1.0))
