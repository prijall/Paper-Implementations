import torch, config
import torch.nn as nn
from torch.nn import functional as F
from utils import get_iou, bbox_attr 

class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_coord=5  #weighting factor for loss related to bb
        self.l_noobj=0.5 #weighting factor for penalizing wrong pred

        def forward(self, boxA, boxB):
            iou=get_iou(boxA, boxB)  # batch, S, S, B, B
            max_iou=torch.max(iou, dim=-1)[0] # batch, S, S, B

            #getting masks
            bbox_mask=bbox_attr(boxB, 4) > 0.0
            boxA_template=bbox_attr(boxA, 4) > 0.0
            obj_i=bbox_mask[..., 0:1] # 1 if grid I has any object at all

            responsible=torch.zeros_like(boxA_template).scatter_(
                -1, 
                torch.argmax(max_iou, dim=-1, keepdim=True),
                Value=1 # 1 if bb is "responsible" for predicting obj
            )

            obj_ij=obj_i * responsible
            noobj_ij=~obj_ij




