import pandas as pd
import numpy as np
import matplotlib as plt
import math
import networkx as nx

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
dataTop = train_df.head()

class Tree:
    def __init__(self):
        self.Classification = False
        self.Feature = None
        self.Average = None
        self.BelowTree = None
        self.AboveTree = None


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

    return -1*(v0 + v1)







def MajorityClass(dataFrame: pd.DataFrame)->(bool, bool):
    positive = 0
    negative = 0
    for case in dataFrame.iloc[:,0]:
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




def TDIDT(dataFrame: pd.DataFrame, classification:bool)-> Tree:
    newTree = Tree()
    if len(dataFrame['diagnosis']) <= 0:
        newTree.Classification = classification
        return newTree
    classification, isZero = MajorityClass(dataFrame)
    featureSet = {feature for feature in dataFrame}
    if isZero is True or len(featureSet) <= 1:
        newTree.Classification = classification
        return newTree
    selectedFeature = SelectFeature(dataFrame)

    newTree.Feature = selectedFeature
    newTree.Average = (getMinInCol(selectedFeature, dataFrame) + getMaxInCol(selectedFeature, dataFrame)) / 2

    newDataFrameAbove = dataFrame.copy()
    newDataFrameBelow = dataFrame.copy()

    aboveIndex = newDataFrameAbove[newDataFrameAbove[selectedFeature] > newTree.Average].index
    belowIndex = newDataFrameAbove[newDataFrameAbove[selectedFeature] <= newTree.Average].index

    newDataFrameBelow.drop(columns=selectedFeature, inplace=True)
    newDataFrameAbove.drop(columns=selectedFeature, inplace=True)

    newDataFrameAbove.drop(belowIndex, inplace=True)
    newDataFrameBelow.drop(aboveIndex, inplace=True)

    newTree.AboveTree = TDIDT(newDataFrameAbove, classification)
    newTree.BelowTree = TDIDT(newDataFrameBelow, classification)

    return newTree


# while not train_df.empty:
#     for col in train_df:
#         print(col)
#     feature = SelectFeature(train_df)
#     print("removed feature is:",feature )
#     if feature == None:
#         break
#     train_df.drop(columns= feature,  inplace=True)



def DTClassify(dataFrame:pd.DataFrame, tree:Tree , index)->bool:
    if tree.BelowTree is None and tree.AboveTree is None:
        return tree.Classification
    feature = tree.Feature
    Average = tree.Average
    value = dataFrame[feature]
    if value[index] > Average:
        return DTClassify(dataFrame, tree.AboveTree, index)
    else:
        return DTClassify(dataFrame, tree.BelowTree, index)



tree = TDIDT(train_df,True)

indices = list(range(0, len(test_df['diagnosis'])))

realClassList = [val for val in test_df['diagnosis']]
accurateCount = 0
inAccurateCount = 0

for i in indices:
    newIndices = indices.copy()
    newIndices.remove(i)

    newDataFrame = test_df.copy()
    newDataFrame.drop(index=newIndices, inplace=True)

    classification = DTClassify(newDataFrame, tree, i)
    if classification is True:
        numericalClass =1
    else:
        numericalClass = 0
    if numericalClass == realClassList[i]:
        accurateCount += 1
    else:
        inAccurateCount += 1

percentage = (accurateCount/(accurateCount + inAccurateCount))*100

print("Accurate were:", accurateCount)
print("InAccurate were:", inAccurateCount)
print("The percentage is:", percentage,"%")