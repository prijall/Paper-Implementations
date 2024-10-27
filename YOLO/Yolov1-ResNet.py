import config, torch
from torch.nn import nn
from torchvision.models import resnet50, ResNet50_Weights

class Yolov1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth= 5 * config.B + config.C

        #loading resnet:
        foundation=resnet50(weights=ResNet50_Weights.DEFAULT)
        foundation.requires_grad_(False) # freezing backbone params

        # deleting last 2 layers and attaching detection layers
        foundation.avgpool=nn.Identity()
        foundation.fc=nn.Identity()

        self.model=nn.Sequential(
            foundation, 
            Reshape(2048, 14, 14),
            DetectionNet(2048)
        )




class DetectionNet(nn.Module):
    """layers added on for detection as given in paper"""
    def __init__(self, in_channels):
        super.__init__()

        inner_channel=1024
        self.depth=5* config.B + config.C
        self.model=nn.Sequential(
            nn.Conv2D(in_channels, inner_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2D(inner_channel, inner_channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2D(inner_channel, inner_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2D(inner_channel, inner_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channel, 4096),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, config.S * config.S * self.depth)
        )

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (-1, config.S, config.S, self.depth)
        )
    


#### Helper function:
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape=tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))