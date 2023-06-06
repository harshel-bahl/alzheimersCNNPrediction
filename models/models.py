from .ConvNet import ConvNet
from .InceptionConvNet import InceptionConvNet
from .PretrainedConvNet import PretrainedConvNet
from .VisionTransformer import ViT

MODELS = {
    "ConvNet": ConvNet,
    "InceptionConvNet": InceptionConvNet,
    "PretrainedConvNet": PretrainedConvNet,
    "ViT": ViT
}