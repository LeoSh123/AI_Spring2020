import pandas as pd
import numpy as np
import matplotlib as plt
import math
import networkx as nx

train_df = pd.read_csv("train.csv")

dataTop = train_df.head()


def getMaxInCol(numOfCol)->(float):
    maxVal = max(val for val in train_df.iloc[:,numOfCol])
    return maxVal

def getMinInCol(numOfCol)->(float):
    minVal = min(val for val in train_df.iloc[:,numOfCol])
    return minVal

def getEntropy(indexArray: np.array, dataFrame:train_df)->(float):
    # average = (getMaxInCol(numOfFeature) + getMinInCol(numOfFeature)) / 2
    # array = np.array(train_df.iloc[:,numOfFeature])
    # lowerIndexArray = np.argwhere[array < average]
    # higherIndexArray = np.argwhere[array >= average]
    # probLower = len(lowerIndexArray) / len(array)
    # probHigher = len(higherIndexArray) / len(array)
    #
    # listOfBelowIndex = list()
    # index = 0
    # for val in train_df.iloc[:,numOfFeature]:
    #     if val < average:
    #         listOfBelowIndex.append(index)
    #     index +=1
    # listOfAboveIndex = train_df.index - listOfBelowIndex
    positiveSum = 0
    for index in indexArray:
        if dataFrame.iloc[index , 0] == 1:
            positiveSum += 1
    negativeSum = len(indexArray) - positiveSum
    probZero = negativeSum / len(indexArray)
    probOne = positiveSum / len(indexArray)
    return -1*(probZero * math.log2(probZero) + probOne * math.log2(probOne))





array = np.array(train_df.index)
print(array)
print("entropy is", getEntropy(array, train_df))


