import torch
from sklearn.metrics import jaccard_score


def calculateIoU(actual, predicted, threshold=0.75 ):
    try:
        actual = actual.detach().cpu().numpy()
    except:
        pass
    try:
        predicted = predicted.detach().cpu().numpy()
    except:
        pass
    actual = (actual > threshold).astype(int).flatten()
    predicted = (predicted > threshold).astype(int).flatten()
    #print("actual: ", actual)
    #print("predicted: ", predicted)
    iou_score = jaccard_score(actual, predicted)
    #print("IOU value: " + str(iou_score))
    return iou_score