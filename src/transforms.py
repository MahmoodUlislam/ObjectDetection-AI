import torchvision.transforms as T


def get_transform(train):
    transforms_list = [T.ToTensor()]
    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_list)
