import  torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    #boxes_preds shape is (N,4) where N is the number of bboxes
    #boxes_labels shape is (N, 4)
    """

    :param boxes_preds: tensor  predictions of bounding box (BATCH_SIZE, 4)
    :param boxes_labels: Correct labels of bounding boxes   (BATCH_SIZE, 4)
    :param box_format: midpoint/corners
    :return:
    """
    if box_format =="midpoint":# 中心点坐标 横 竖
        box1x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3] / 2
        box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4] / 2
        box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box1x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box1_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box1_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box1_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    if box_format == "corners":
        box1_x1 = boxes_preds[...,0:1]#...表示前面所有维度
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersction = (x2-x1).clamp(0)*(y2-y1) #clamp（0）用于 they do not intersect

    box1_area  = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))

    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersction / (box1_area + box2_area - intersction + 1e-6)
