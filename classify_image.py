import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # opencv used BGR while torchvision uses RGB
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE)) #converting into format that pretrained nets use
    image = image.astype("float32")/255.0 # converting in range [0,1]

    # Normalize images
    image -= config.MEAN
    image /= config.STD
    image = np.transpose(image, (2,0,1)) #Number of channels are kept in 1st dimension
    image = np.expand_dims(image, 0) #Dimension for batch. Pytorch accepts: B x C x H x W

    return image

# Command line input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,  help = "path for input image")
ap.add_argument("-m", "--model", type=str, default= "vgg16", 
                        choices= ["vgg16", "vgg19", "inception","densenet", "resnet"], 
                        help="name of the pre-trained network to use")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "inception": models.inception_v3(pretrained= True),
    "densenet": models.densenet121(pretrained= True),
    "resnet": models.resnet50(pretrained=True)
}

print("Loading model {} for prediction...".format(args['model']))
model = MODELS[args['model']].to(config.DEVICE)
model.eval()

# Loading and preprocessing image
print("loading image ...")
image = cv2.imread(args['image'])
orig = image.copy()
image = preprocess_image(image)

# Coverting to torch datatype
image = torch.from_numpy(image)
image = image.to(config.DEVICE)

# Imagenet labels
imagenetlabels = dict(enumerate(open(config.IN_LABELS)))

# Predicting ...
logits = model(image)
probabilities = torch.nn.Softmax(dim = -1)(logits)
sorted_probabilities = torch.argsort(probabilities, dim = -1, descending=True)

# Top 5 predictions
for (i, idx) in enumerate(sorted_probabilities[0,:5]):
    print(i, imagenetlabels[idx.item()].strip())
    print(probabilities[0, idx.item()] * 100)

