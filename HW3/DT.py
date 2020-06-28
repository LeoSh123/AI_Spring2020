import pandas as pd
import numpy as np
import matplotlib as plt
import math
import networkx as nx

train_df = pd.read_csv("train.csv")

dataTop = train_df.head()


def getMaxInCol(feature:str ,dataFrame:pd.DataFrame)->(float):
    maxVal = max(val for val in dataFrame[feature])
    return maxVal

def getMinInCol(feature:str ,dataFrame:pd.DataFrame)->(float):
    minVal = min(val for val in dataFrame[feature])
    return minVal

def getEntropy(dataFrame:pd.DataFrame)->(float):
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

    for val in dataFrame['diagnosis']:
        if val == 1:
            positiveSum += 1
    NumOfVals = len(dataFrame['diagnosis'])
    negativeSum = NumOfVals - positiveSum
    probZero = negativeSum / NumOfVals
    probOne = positiveSum / NumOfVals
    if probZero == 0:
        v0 = 0
    else:
        v0 = probZero * math.log2(probZero)
    if probOne == 0:
        v1 = 0
    else:
        v1 = probOne * math.log2(probOne)

    return -1*(v0 + v1)







def MajorityClass(dataFrame: pd.DataFrame)->(bool):
    positive = 0
    negative = 0
    for case in dataFrame.iloc[:,0]:
        if case == 0:
            negative += 1
        else:
            positive += 1
    if positive > negative:
        return True
    else:
        return False


def SelectFeature(dataFrame: pd.DataFrame)->(str):
     CurrentEntropy = getEntropy(dataFrame)
     maxIG = 0
     for col in dataFrame:
         if col == 'diagnosis':
             continue


         average =  (getMinInCol(col, dataFrame) + getMaxInCol(col, dataFrame)) /2
         newDataFrameAbove = dataFrame.copy()
         newDataFrameBelow = dataFrame.copy()

         aboveIndex = newDataFrameAbove[newDataFrameAbove[col] > average].index
         belowIndex = newDataFrameAbove[newDataFrameAbove[col] <= average].index

         newDataFrameAbove.drop(belowIndex, inplace=True)
         newDataFrameBelow.drop(aboveIndex, inplace=True)

         aboveLen = len(aboveIndex)
         belowLen = len(belowIndex)
         totalLen = aboveLen + belowLen

         IG = CurrentEntropy - ((aboveLen/totalLen)*getEntropy(newDataFrameAbove) + (belowLen/totalLen)*getEntropy(newDataFrameBelow))
         maxIG = max(IG, maxIG)

     for col in dataFrame:
         if col == 'diagnosis':
             continue

         average = (getMinInCol(col, dataFrame) + getMaxInCol(col, dataFrame)) / 2
         newDataFrameAbove = dataFrame.copy()
         newDataFrameBelow = dataFrame.copy()

         aboveIndex = newDataFrameAbove[newDataFrameAbove[col] > average].index
         belowIndex = newDataFrameAbove[newDataFrameAbove[col] <= average].index

         newDataFrameAbove.drop(belowIndex, inplace=True)
         newDataFrameBelow.drop(aboveIndex, inplace=True)

         aboveLen = len(aboveIndex)
         belowLen = len(belowIndex)
         totalLen = aboveLen + belowLen

         IG = CurrentEntropy - ((aboveLen / totalLen) * getEntropy(newDataFrameAbove) + (belowLen / totalLen) * getEntropy(
             newDataFrameBelow))
         if IG == maxIG:
             return col






# array = np.array(train_df.index)
# print(array)
# print("entropy is", getEntropy(train_df))
# print(train_df['area_mean'])
# print("Max area mean:" , getMaxInCol('area_mean',train_df))
# print("Min area mean:" , getMinInCol('area_mean',train_df))


print(SelectFeature(train_df))







