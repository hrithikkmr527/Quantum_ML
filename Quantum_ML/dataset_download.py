import torch
from torchgeo.datasets import LandCoverAI


root  = "./Datasets/LandCover/"

## downloading dataset
dataset = LandCoverAI(root = root, download=True)
print("\n Dataset downloaded successfully!..\n")


## Add image cropping and splitting code here later




