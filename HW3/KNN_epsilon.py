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
        self.Average = None
        self.BelowTree = None
        self.AboveTree = None
        self.Examples = str()


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

        average =  (getMinInCol(Feature, dataFrame) + getMaxInCol(Feature, dataFrame)) /2
        newDataFrameAbove = dataFrame.copy()
        newDataFrameBelow = dataFrame.copy()

        aboveIndex = newDataFrameAbove[newDataFrameAbove[Feature] > average].index
        belowIndex = newDataFrameAbove[newDataFrameAbove[Feature] <= average].index

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

        average = (getMinInCol(Feature, dataFrame) + getMaxInCol(Feature, dataFrame)) / 2
        newDataFrameAbove = dataFrame.copy()
        newDataFrameBelow = dataFrame.copy()

        aboveIndex = newDataFrameAbove[newDataFrameAbove[Feature] > average].index
        belowIndex = newDataFrameAbove[newDataFrameAbove[Feature] <= average].index

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
        newTree.Examples = dataFrame.index
        return newTree
    featureSet = {feature for feature in dataFrame}
    if isZero is True or len(featureSet) <= 1:
        newTree.Classification = classification
        newTree.Examples = dataFrame.index
        return newTree
    selectedFeature = Eps_SelectFeature(dataFrame,bigDataFrame)

    newTree.Feature = selectedFeature
    newTree.Vi = np.std(np.array(bigDataFrame[selectedFeature]))
    newTree.Average = (getMinInCol(selectedFeature, dataFrame) + getMaxInCol(selectedFeature, dataFrame)) / 2


    newDataFrameAbove = dataFrame.copy()
    newDataFrameBelow = dataFrame.copy()

    aboveIndex = newDataFrameAbove[newDataFrameAbove[selectedFeature] > newTree.Average].index
    belowIndex = newDataFrameAbove[newDataFrameAbove[selectedFeature] <= newTree.Average].index

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
        return list(tree.Examples)
    feature = tree.Feature
    Vi = tree.Vi
    Average = tree.Average
    value = dataFrame[feature]
    if abs(value[index] - Average) <= 0.1*Vi:
        list1 = DT_Epsilon_Classify(dataFrame, tree.AboveTree, index)
        list2 = DT_Epsilon_Classify(dataFrame, tree.BelowTree, index)
        list3 = list1 + list2
        return list3
    elif value[index] > Average:
        return DT_Epsilon_Classify(dataFrame, tree.AboveTree, index)
    else:
        return DT_Epsilon_Classify(dataFrame, tree.BelowTree, index)



def getNormalizedValue(value: float, min:float, max:float)->float:
    return (value - min)/(max - min)

def getEuclideanDistance(dataFrame: pd.DataFrame,objectDataFrame: pd.DataFrame ,objectIndex:int, yIndex:int)->(float):
    sumOfDi = 0
    for Feature in dataFrame:
        if Feature == 'diagnosis':
            continue
        valuesArray = dataFrame[Feature]
        yValue = valuesArray[yIndex]
        valueTmpArray = objectDataFrame[Feature]
        xValue = valueTmpArray[objectIndex]

        featureMin = getMinInCol(Feature,dataFrame)
        featureMax = getMaxInCol(Feature,dataFrame)

        objectF = getNormalizedValue(xValue,featureMin,featureMax)
        yF = getNormalizedValue(yValue,featureMin,featureMax)

        sumOfDi += math.pow(abs(objectF - yF),2)

    return math.sqrt(sumOfDi)

def takeSecond(elem):
    return elem[1]

def KNN(k:int,dataFrame: pd.DataFrame, objectDataFrame: pd.DataFrame, objectIndex:int,bigDataFrame: pd.DataFrame)->bool:
    indices = dataFrame.index
    neighborsIndexAndDistance = list()
    for i in indices:
        dist = getEuclideanDistance(dataFrame,objectDataFrame,objectIndex,i)
        neighborsIndexAndDistance.append([i,dist])

    neighborsIndexAndDistance.sort(key=takeSecond)
    kClosestList = neighborsIndexAndDistance[0:k:1]

    negativeSum = 0
    positiveSum = 0
    for index in kClosestList:
       if bigDataFrame.iloc[index[0], 0] == 1:
           positiveSum +=1
       else:
           negativeSum +=1

    if negativeSum > positiveSum:
        return False
    else:
        return True



testIndices = list(range(0, len(test_df['diagnosis'])))
trainIndices = list(range(0, len(train_df['diagnosis'])))

realClassList = [val for val in test_df['diagnosis']]

bigDf = train_df.copy()



tree = Eps_TDIDT(train_df, True, 9, bigDf)
accurateCount = 0
inAccurateCount = 0

for i in testIndices:
    newTestIndices = testIndices.copy()
    newTestIndices.remove(i)

    newObjectDataFrame = test_df.copy()
    newObjectDataFrame.drop(index=newTestIndices, inplace=True)

    leafList = DT_Epsilon_Classify(newObjectDataFrame, tree, i)

    newTrainIndices = trainIndices.copy()
    newTrainIndices = [ele for ele in newTrainIndices if ele not in leafList]


    newTrainDataFrame = train_df.copy()
    newTrainDataFrame.drop(index=newTrainIndices, inplace=True)

    classification = KNN(9, newTrainDataFrame, newObjectDataFrame, i, train_df)
    # print("Object number:",i, " its class is:",classification)
    if classification is True:
        numericalClass = 1
    else:
        numericalClass = 0
    if numericalClass == realClassList[i]:
        accurateCount += 1
        # print(" classification was accurate")
    else:
        inAccurateCount += 1
       # print(" classification was not accurate ##########\n")

percentage = (accurateCount / (accurateCount + inAccurateCount)) * 100
print(percentage)



