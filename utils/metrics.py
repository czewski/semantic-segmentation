from torchmetrics import JaccardIndex

## iou (intersection over union)
def Iou(gt, pre):
    jaccard = JaccardIndex(task='multiclass', num_classes=8)
    return jaccard(pre, gt)