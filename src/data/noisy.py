from data import backbone
import torch

class NoisyData(backbone.RestorationData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

