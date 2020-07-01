import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as pylt
import math
import networkx as nx

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


class Tree:
    def __init__(self):
        self.Classification = False
        self.Feature = None
        self.Vi = None
        self.BelowTree = None
        self.AboveTree = None


def getMaxInCol(feature: str, dataFrame: pd.DataFrame) -> (float):
    maxVal = max(val for val in dataFrame[feature])
    return maxVal


def getMinInCol(feature: str, dataFrame: pd.DataFrame) -> (float):
    minVal = min(val for val in dataFrame[feature])
    return minVal


def getEntropy(dataFrame: pd.DataFrame) -> (float):
    positiveSum = 0

    for val in dataFrame['diagnosis']:
        if val == 1:
            positiveSum += 1
    NumOfVals = len(dataFrame['diagnosis'])
    negativeSum = NumOfVals - positiveSum
    if NumOfVals == 0:
        return 0
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

    return -1 * (v0 + v1)


def MajorityClass(dataFrame: pd.DataFrame) -> (bool, bool):
    positive = 0
    negative = 0
    for case in dataFrame.iloc[:, 0]:
        if case == 0:
            negative += 1
        else:
            positive += 1
    if positive == 0 or negative == 0:
        isZero = True
    else:
        isZero = False
    if positive >= negative:
        return True, isZero
    else:
        return False, isZero


def Eps_SelectFeature(dataFrame: pd.DataFrame,bigDataFrame: pd.DataFrame) -> (str):
    CurrentEntropy = getEntropy(dataFrame)
    maxIG = 0
    for Feature in dataFrame:
        if Feature == 'diagnosis':
            continue

        Vi = np.std(bigDataFrame[Feature])
        newDataFrameAbove = dataFrame.copy()
        newDataFrameBelow = dataFrame.copy()

        aboveIndex = newDataFrameAbove[newDataFrameAbove[Feature] > Vi].index
        belowIndex = newDataFrameAbove[newDataFrameAbove[Feature] <= Vi].index

        newDataFrameAbove.drop(belowIndex, inplace=True)
        newDataFrameBelow.drop(aboveIndex, inplace=True)

        aboveLen = len(aboveIndex)
        belowLen = len(belowIndex)
        totalLen = aboveLen + belowLen

        IG = CurrentEntropy - (
                    (aboveLen / totalLen) * getEntropy(newDataFrameAbove) + (belowLen / totalLen) * getEntropy(
                newDataFrameBelow))
        maxIG = max(IG, maxIG)

    for Feature in dataFrame:
        if Feature == 'diagnosis':
            continue

        Vi = np.std(bigDataFrame[Feature])
        newDataFrameAbove = dataFrame.copy()
        newDataFrameBelow = dataFrame.copy()

        aboveIndex = newDataFrameAbove[newDataFrameAbove[Feature] > Vi].index
        belowIndex = newDataFrameAbove[newDataFrameAbove[Feature] <= Vi].index

        newDataFrameAbove.drop(belowIndex, inplace=True)
        newDataFrameBelow.drop(aboveIndex, inplace=True)

        aboveLen = len(aboveIndex)
        belowLen = len(belowIndex)
        totalLen = aboveLen + belowLen

        IG = CurrentEntropy - (
                    (aboveLen / totalLen) * getEntropy(newDataFrameAbove) + (belowLen / totalLen) * getEntropy(
                newDataFrameBelow))
        if IG == maxIG:
            return Feature


    stopHere = None



# array = np.array(train_df.index)
# print(array)
# print("entropy is", getEntropy(train_df))
# print(train_df['area_mean'])
# print("Max area mean:" , getMaxInCol('area_mean',train_df))
# print("Min area mean:" , getMinInCol('area_mean',train_df))


def Eps_TDIDT(dataFrame: pd.DataFrame, classification: bool, x: int, bigDataFrame: pd.DataFrame) -> Tree:
    newTree = Tree()
    if len(dataFrame['diagnosis']) <= 0:
        newTree.Classification = classification
        return newTree
    classification, isZero = MajorityClass(dataFrame)
    if len(dataFrame['diagnosis']) <= x:
        newTree.Classification = classification
        return newTree
    featureSet = {feature for feature in dataFrame}
    if isZero is True or len(featureSet) <= 1:
        newTree.Classification = classification
        return newTree
    selectedFeature = Eps_SelectFeature(dataFrame,bigDataFrame)

    newTree.Feature = selectedFeature
    newTree.Vi = np.std(bigDataFrame[selectedFeature])

    newDataFrameAbove = dataFrame.copy()
    newDataFrameBelow = dataFrame.copy()

    aboveIndex = newDataFrameAbove[newDataFrameAbove[selectedFeature] > newTree.Vi].index
    belowIndex = newDataFrameAbove[newDataFrameAbove[selectedFeature] <= newTree.Vi].index

    newDataFrameBelow.drop(columns=selectedFeature, inplace=True)
    newDataFrameAbove.drop(columns=selectedFeature, inplace=True)

    newDataFrameAbove.drop(belowIndex, inplace=True)
    newDataFrameBelow.drop(aboveIndex, inplace=True)

    newTree.AboveTree = Eps_TDIDT(newDataFrameAbove, classification, x, bigDataFrame)
    newTree.BelowTree = Eps_TDIDT(newDataFrameBelow, classification, x, bigDataFrame)

    return newTree


# while not train_df.empty:
#     for col in train_df:
#         print(col)
#     feature = SelectFeature(train_df)
#     print("removed feature is:",feature )
#     if feature == None:
#         break
#     train_df.drop(columns= feature,  inplace=True)


def DT_Epsilon_Classify(dataFrame: pd.DataFrame, tree: Tree, index) -> (int, int): # first is positive, second is negative
    if tree.BelowTree is None and tree.AboveTree is None:
        if tree.Classification is True:
            return 1,0
        else:
            return 0,1
    feature = tree.Feature
    Vi = tree.Vi
    value = dataFrame[feature]
    if abs(value[index] - Vi) <= 0.1*Vi:
        pos1, neg1 = DT_Epsilon_Classify(dataFrame, tree.AboveTree, index)
        pos2, neg2 = DT_Epsilon_Classify(dataFrame, tree.BelowTree, index)
        return pos1+pos2, neg1+neg2
    elif value[index] > Vi:
        return DT_Epsilon_Classify(dataFrame, tree.AboveTree, index)
    else:
        return DT_Epsilon_Classify(dataFrame, tree.BelowTree, index)


indices = list(range(0, len(test_df['diagnosis'])))

realClassList = [val for val in test_df['diagnosis']]

bigDf = train_df.copy()

tree = Eps_TDIDT(train_df, True, 9, bigDf)
accurateCount = 0
inAccurateCount = 0

for i in indices:
    newIndices = indices.copy()
    newIndices.remove(i)

    newDataFrame = test_df.copy()
    newDataFrame.drop(index=newIndices, inplace=True)

    positive, negative = DT_Epsilon_Classify(newDataFrame, tree, i)
    if positive >= negative:
        classification = True
    else:
        classification = False
    if classification is True:
        numericalClass = 1
    else:
        numericalClass = 0
    if numericalClass == realClassList[i]:
        accurateCount += 1
    else:
        inAccurateCount += 1

percentage = (accurateCount / (accurateCount + inAccurateCount)) * 100
# percantageList.append(percentage)

# print("Accurate were:", accurateCount)
# print("InAccurate were:", inAccurateCount)
# print("The percentage is:", percentage,"%")
print(percentage)
