import torch
from iou_test import intersection_over_union
def nms(
    bboxes,
    iou_threshold,
    threshold,
    box_format="corners"
):
    #predicitons = [[ ], [], []] 种类 概率  x1 y1 x2 y2

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x:x[1],reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]#种类不同需要保留  不同比较threshold
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)


    return bboxes_after_nms
