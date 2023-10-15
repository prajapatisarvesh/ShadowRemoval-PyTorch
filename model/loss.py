from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F

class AlphaLoss(object):
    def __init__(self):
        super().__init__()

    
    def __call__(self, output_mask, target_mask):
        return F.mse_loss(output_mask, target_mask)


class CompositionLoss(object):
    def __init__(self):
        super().__init__()

    
    def __call__(self, output_shadow, target_shadow):
        return 1


class CombinationLoss(object):
    def __init__(self):
        super().__init__()
        self.alpha_loss = AlphaLoss()

    
    def __call__(self, output, target_mask, shadow_free, shadow):
        return self.alpha_loss(output, target_mask)