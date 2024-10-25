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

            # XY position loss:
            X_losses=mse_loss(
                  obj_ij * bbox_attr(boxA, 0),
                  obj_ij * bbox_attr(boxB, 0)
            )

            Y_losses=mse_loss(
                obj_ij * bbox_attr(boxA, 1),
                 obj_ij * bbox_attr(boxB, 1)
            )
                  
            

            pos_losses=X_losses + Y_losses #position losses

            # Bbox dimension losses:
            boxA_width=bbox_attr(boxA, 2)
            boxB_width=bbox_attr(boxB, 2)
            
            width_losses=mse_loss(
                obj_ij * torch.sign(boxA_width) * torch.sqrt(torch.abs(boxA_width) + config.EPSILON),
                obj_ij * torch.sqrt(boxB_width)
            )

            boxA_height=bbox_attr(boxA, 3)
            boxB_height=bbox_attr(boxB, 3)
            
            height_losses=mse_loss(
                obj_ij * torch.sign(boxA_height) * torch.sqrt(torch.abs(boxA_height) + config.EPSILON),
                obj_ij * torch.sqrt(boxB_height)
            )
            dim_losses=width_losses + height_losses

            # Confidence loss:
            obj_confidence_losses=mse_loss(
                obj_ij * bbox_attr(boxA, 4),
                obj_ij * torch.ones_like(max_iou)
            )

            noobj_confidence_losses=mse_loss(
                noobj_ij * bbox_attr(boxA, 4),
                noobj_ij * torch.zeros_like(max_iou)
            )

            # classification loss:
            class_losses=mse_loss(
                obj_i * boxA[..., :config.C],
                obj_i * boxB[..., config.C]
            )

            total=self.l_coord * (pos_losses + dim_losses) \
                  + obj_confidence_losses \
                  + self.l_noobj * noobj_confidence_losses \
                  + class_losses
            
            return total / config.BATCH_SIZE
            

def mse_loss(a, b):
    flattened_a=torch.flatten(a, end_dim=-2)
    flattened_b=torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(
        flattened_a,
        flattened_b,
        reduction='sum'
    )


