import torch

# Dimension of the input image
IMAGE_SIZE = 224

# mean and standard deviation of imagenet dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Check for availability of GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# imagenet labels
IN_LABELS = "imagenet_classes.txt"