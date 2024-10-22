import torch 
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset


class YoloPascalVocDataset(Dataset):
    def __init__(self, set_type, normalize=False, augment=False):
        assert  set_type in {'train', 'test'}
        self.dataset=VOCDetection(
            root=config.DATA_PATH,
            year='2007',
            image_set=('train' if set_type=='train' else 'val'),
            download=True, 
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config.IMAGE_SIZE)
            ])
        )

        self.normalize=normalize
        self.augment=augment
        self.classes=utils.load_class_dict() #for object map into numerical values.

    
    def __getitem__(self, i):
        data, label=self.dataset[i]
        original_data=data #copy of original data
        x_shift=int((0.2* random.random()-0.1)*config.IMAGE_SIZE[0])
        y_shift=int((0.2*random.random()-0.1)*config.IMAGE_SIZE[1])
        scale=1 + 0.2 * random.random()

        # Augmenting Images:
        if self.augment:
            data=TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data=TF.adjust_hue(data, 0.2*random.random()-0.1)
            data=TF.adjust_saturation(data, 0.2*random.random()+0.9)

        if self.normalize:
            data=TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        grid_size_x=data.size(dim=2)/config.S #pytorch has channl, height, width
        grid_size_y=data.size(dim=1)/ config.S 
  
        

    def __len__(self):
        return len(self.dataset)