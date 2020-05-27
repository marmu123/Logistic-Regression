from utils import sigmoid
from sklearn.linear_model import LogisticRegression
class MyLogisticRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []


    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        self.coef_ = [0.0 for _ in range(1 + len(x[0]))]  # y = w0 + w1 * x1 + w2 * x2 + ...

        for epoch in range(noEpochs):
            for i in range(len(x)):  # for each sample from the training data
                ycomputed = sigmoid(self.eval(x[i], self.coef_))  # estimate the output
                crtError = float(ycomputed) - float(y[i])  # compute the error for the current sample
                for j in range(0, len(x[0])):  # update the coefficients
                    self.coef_[j + 1] = self.coef_[j + 1] - learningRate * crtError * float(x[i][j])
                self.coef_[0] = self.coef_[0] - learningRate * crtError * 1

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def eval(self, xi, coef):
        yi = coef[0]
        for j in range(len(xi)):
            yi += float(coef[j + 1]) * float(xi[j])
        return yi

    def predictOneSample(self, sampleFeatures):
        threshold = 0.5
        coefficients =[self.intercept_]+self.coef_
        computedFloatValue = self.eval(sampleFeatures, coefficients)
        computed01Value = sigmoid(computedFloatValue)
        computedLabel = 0 if computed01Value < threshold else 1
        return computedLabel

class SKLogisticRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []
        self.regr=None

    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        self.regr=LogisticRegression(max_iter=noEpochs,tol=learningRate)
        self.regr.fit(x,y)

    def predictOneSample(self,x):
        threshold = 0.5
        predValue=self.regr.predict(x)
        computedLabel=0 if predValue < threshold else 1
        return computedLabel