import torch


def get_coco_api_from_dataset(dataset):
    for i in range(len(dataset)):
        img, target = dataset[i]
        if 'boxes' in target:
            break
    return dataset.coco


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), 1)
