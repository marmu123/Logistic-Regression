from math import exp

def sigmoid(x):
    return 1 / (1 + exp(-x))


def multiClassLoss(realOutput, computedOutput):
    '''
   input: realLabels, computedLabels - one-dimensional arrays of the same length containing labels
   (two arrays of noSamples labels from {label_1, label_2, ..., label_C},
    noSamples = no of samples/exampeles)
    output: prediction quality expressed by accuracy, precison and recall.
    '''
    labels=set(realOutput)
    acc=sum([1 if realOutput[i]==computedOutput[i] else 0  for i in range(len(computedOutput))])/len(realOutput)
    prec={}
    recall={}
    for el in labels:
        tp=sum([1 if realOutput[i]==el and computedOutput[i]==el else 0  for i in range(len(computedOutput))])
        fp=sum([1 if realOutput[i]!=el and computedOutput[i]==el else 0  for i in range(len(computedOutput))])
        fn=sum([1 if realOutput[i]==el and computedOutput[i]!=el else 0  for i in range(len(computedOutput))])
        prec[el]=tp/(tp+fp)
        recall[el]=tp/(tp+fn)

    return acc,prec,recall



