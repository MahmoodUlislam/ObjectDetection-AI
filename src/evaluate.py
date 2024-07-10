import torch
import torchvision
from torch.utils.data import DataLoader
from src.engine import evaluate
import src.utils as utils
from src.dataset import CustomDataset
from src.transforms import get_transform


def run():
    # Paths
    root_dir = "data/"

    # Load dataset
    dataset_test = CustomDataset(root=root_dir, transforms=get_transform(train=False))

    # Data loader
    data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # Load the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=False)
    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()

    # Evaluation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    evaluate(model, data_loader_test, device=device)
