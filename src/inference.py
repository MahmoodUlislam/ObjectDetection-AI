import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.my_model import MyModel

# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=False)
model.load_state_dict(torch.load('models/model.pth'))
model.eval()

# Example inference code
input_data = torch.randn(1, 10)  # Replace with your actual input data
output = model(input_data)
print(output)

# Define the COCO instance category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def run_inference():
    modelTest = MyModel()
    try:
        modelTest.load_state_dict(torch.load('models/model.pth', map_location=torch.device('cpu')))
        modelTest.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

    # Example inference code
    input_data_ex = torch.randn(1, 10)  # Replace with actual input data
    output_test = model(input_data_ex)
    print(output_test)


def get_prediction(img, threshold):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


def run(img_path, threshold=0.5):
    img = Image.open(img_path)
    boxes, pred_cls = get_prediction(img, threshold)
    plot_results(img, boxes, pred_cls)


def plot_results(img, boxes, pred_cls):
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    ax = plt.gca()
    for box, cls in zip(boxes, pred_cls):
        rect = patches.Rectangle(box[0], box[1][0] - box[0][0], box[1][1] - box[0][1], linewidth=2, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0][0], box[0][1], cls, fontsize=15, color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Replace 'path_to_image.jpg' with the path to the image you want to test
    run('assets/apple.jpg')
    run_inference()
