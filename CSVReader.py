import csv
import numpy as np
class CSVReader:
    def __init__(self,filename):
        self.filename = filename
        self.trainInputs = []
        self.trainOutputs = []
        self.testInputs = []
        self.testOutputs = []
        self.inputData = []
        self.outputData = []

    def readData(self):
        with open(self.filename) as csvfile:
            csvreader=csv.reader(csvfile,delimiter=',')
            for row in csvreader:
                if row:
                    self.inputData.append(row[:2])
                    self.outputData.append(row[-1])

    def splitData(self):
        indexes = [i for i in range(len(self.inputData))]
        trainIndexes = np.random.choice(indexes, int(0.8 * len(self.inputData)), replace=False)
        testIndexex = [i for i in indexes if i not in trainIndexes]

        self.trainInputs=[self.inputData[i] for i in trainIndexes]
        self.trainOutputs=[self.outputData[i] for i in trainIndexes]

        self.testInputs=[self.inputData[i] for i in testIndexex]
        self.testOutputs=[self.outputData[i] for i in testIndexex]
