import torch
import torchvision
from torch.utils.data import DataLoader
from src.engine import train_one_epoch, evaluate
import src.utils as utils
from src.dataset import CustomDataset
from src.transforms import get_transform


def run():
    # Paths
    root_dir = "data/"

    # Load datasets
    dataset = CustomDataset(root=root_dir, transforms=get_transform(train=True))
    dataset_test = CustomDataset(root=root_dir, transforms=get_transform(train=False))

    # Split datasets
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # Data loaders
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)
    num_classes = 2  # 1 class (object) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    # Save the model
    torch.save(model.state_dict(), 'models/model.pth')
