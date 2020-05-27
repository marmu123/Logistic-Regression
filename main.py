from CSVReader import CSVReader
from LogisticRegression import MyLogisticRegression
from LogisticRegression import SKLogisticRegression
from termcolor import colored
from utils import multiClassLoss
import numpy as np
import pandas as pd
reader=CSVReader("data/iris.data")
reader.readData()
reader.splitData()


outputsSet=list(set(reader.trainOutputs))
predictsMyRegr=[[0 for _ in range(len(outputsSet))] for _ in range(len(reader.testInputs))]
predictsSkRegr=[[0 for _ in range(len(outputsSet))] for _ in range(len(reader.testInputs))]
realIndexes=[]
predictedIndexesMyRegr=[]
predictedIndexesSkRegr=[]
myLogRegr=MyLogisticRegression()
skLogRegr=SKLogisticRegression()

for labelIndex, label in enumerate(outputsSet):
    realIndexes=[]
    outs=[]
    for el in reader.trainOutputs:
        aux=[0]*len(outputsSet)
        aux[outputsSet.index(el)]=1
        outs.append(aux[labelIndex])

    myLogRegr.fit(reader.trainInputs, outs)
    skLogRegr.fit(reader.trainInputs, outs)

    outTest=[]
    for el in reader.testOutputs:
        aux=[0]*len(outputsSet)
        aux[outputsSet.index(el)]=1
        outTest.append(aux[labelIndex])
        realIndexes.append(outputsSet.index(el))
    for j in range(len(reader.testInputs)):
        rezMyRegr=myLogRegr.predictOneSample(reader.testInputs[j])
        rezSkRegr=skLogRegr.predictOneSample(np.array([float(el) for el in reader.testInputs[j]]).reshape(1,-1))

        predictsMyRegr[j][labelIndex]=rezMyRegr
        predictsSkRegr[j][labelIndex]=rezSkRegr

    print("//////////// "+label+" ////////////")

print("///// MY REGRESSOR /////")
for i in range(len(predictsMyRegr)):
    predictedValue=max(predictsMyRegr[i])
    if predictedValue == 0:
        print(colored(" UNKNOWN(NO VALUES OVER THRESHOLD)", 'yellow'))
        predictedIndexesMyRegr.append(-1)
        continue
    predictedIndex=predictsMyRegr[i].index(predictedValue)
    predictedIndexesMyRegr.append(predictedIndex)
    if predictedIndex!=realIndexes[i]:
        print(colored((" PREDICTED: " + str(outputsSet[predictedIndex]) + " REAL: " + str(outputsSet[realIndexes[i]])), 'red'))
    else:
        print(colored((" PREDICTED: " + str(outputsSet[predictedIndex]) + " REAL: " + str(outputsSet[realIndexes[i]])), 'green'))


acc,prec,recall=multiClassLoss(realIndexes, predictedIndexesMyRegr)
print("Accuracy: "+ str(acc))
print("Precision: ")
for i,el in enumerate(prec):
    print("     "+ outputsSet[i]+" " + str(prec[i]))
print("Recall: ")
for i,el in enumerate(recall):
    print("     "+ outputsSet[i]+" " + str(recall[i]))


print("///// SK REGRESSOR /////")
for i in range(len(predictsSkRegr)):
    predictedValue=max(predictsSkRegr[i])
    if predictedValue == 0:
        print(colored(" UNKNOWN(NO VALUES OVER THRESHOLD)", 'yellow'))
        predictedIndexesSkRegr.append(-1)
        continue
    predictedIndex=predictsSkRegr[i].index(predictedValue)
    predictedIndexesSkRegr.append(predictedIndex)
    if predictedIndex!=realIndexes[i]:
        print(colored((" PREDICTED: " + str(outputsSet[predictedIndex]) + " REAL: " + str(outputsSet[realIndexes[i]])), 'red'))
    else:
        print(colored((" PREDICTED: " + str(outputsSet[predictedIndex]) + " REAL: " + str(outputsSet[realIndexes[i]])), 'green'))


acc,prec,recall=multiClassLoss(realIndexes, predictedIndexesSkRegr)
print("Accuracy: "+ str(acc))
print("Precision: ")
for i,el in enumerate(prec):
    print("     "+ outputsSet[i]+" " + str(prec[i]))
print("Recall: ")
for i,el in enumerate(recall):
    print("     "+ outputsSet[i]+" " + str(recall[i]))

