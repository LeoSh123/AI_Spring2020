import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as pylt
import math
import networkx as nx



train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


def getMaxInCol(feature: str, dataFrame: pd.DataFrame) -> (float):
    maxVal = max(val for val in dataFrame[feature])
    return maxVal


def getMinInCol(feature: str, dataFrame: pd.DataFrame) -> (float):
    minVal = min(val for val in dataFrame[feature])
    return minVal

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

def KNN(k:int,dataFrame: pd.DataFrame, objectDataFrame: pd.DataFrame, objectIndex:int)->bool:
    indices = list(range(0, len(train_df['diagnosis'])))
    neighborsIndexAndDistance = list()
    for i in indices:
        dist = getEuclideanDistance(dataFrame,objectDataFrame,objectIndex,i)
        neighborsIndexAndDistance.append([i,dist])

    neighborsIndexAndDistance.sort(key=takeSecond)
    kClosestList = neighborsIndexAndDistance[0:k:1]

    negativeSum = 0
    positiveSum = 0
    for index in kClosestList:
       if dataFrame.iloc[index[0], 0] == 1:
           positiveSum +=1
       else:
           negativeSum +=1

    if negativeSum > positiveSum:
        return False
    else:
        return True


indices = list(range(0, len(test_df['diagnosis'])))

realClassList = [val for val in test_df['diagnosis']]



accurateCount = 0
inAccurateCount = 0
for i in indices:
    newIndices = indices.copy()
    newIndices.remove(i)

    objectDataFrame = test_df.copy()
    objectDataFrame.drop(index=newIndices, inplace=True)

    classification = KNN(9,train_df,objectDataFrame,i)
    # print("Object number:",i," out of:", length ," its class is:",classification)
    if classification is True:
        numericalClass =1
    else:
        numericalClass = 0
    if numericalClass == realClassList[i]:
        accurateCount += 1
        # print("k is:",k ," classification was accurate")
    else:
        inAccurateCount += 1
        # print("k is:",k ," classification was not accurate ##########\n")

percentage = (accurateCount/(accurateCount + inAccurateCount))*100
print(percentage)



