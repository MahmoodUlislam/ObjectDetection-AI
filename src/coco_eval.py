import torch
import copy
import numpy as np
from pycocotools.cocoeval import COCOeval


def prepare(predictions):
    # Prepare for COCO format
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": boxes[k],
                    "score": scores[k],
                }
                for k in range(len(boxes))
            ]
        )
    return coco_results


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = prepare(predictions)
        for iou_type in self.iou_types:
            coco_dt = self.coco_gt.loadRes(results[iou_type])
            self.coco_eval[iou_type].cocoDt = coco_dt
            self.coco_eval[iou_type].params.imgIds = list(img_ids)
            self.eval_imgs[iou_type].append(self.coco_eval[iou_type].evaluate())

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            # Assuming self.eval_imgs[iou_type] is a list of numpy arrays
            # Initialization if needed
            self.eval_imgs[iou_type] = []

            # Concatenate arrays and store in list
            concatenated_array = np.concatenate(self.eval_imgs[iou_type], axis=2)
            self.eval_imgs[iou_type].append(concatenated_array)

            self.coco_eval[iou_type].evalImgs = list(self.eval_imgs[iou_type].flatten())
        self.img_ids = np.unique(self.img_ids).tolist()

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.summarize()


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), 1)
