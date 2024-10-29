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
  
        # Processing bounding boxes into S x S x (5B+C) ground truth tensor

        boxes={} # for tracking how many bounding boxes have been assigned to grid cell
        class_names={} # for tracking what class each grid cell has been assigned to
        depth=5*config.B+ config.C
        ground_truth=torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(utils.get_bounding_box(label)):
            name, coords=bbox_pair
            assert name in self.classes, f'Unrecognized class {name}'
            class_index=self.classes[name]
            x_min, x_max, y_min, y_max=coords

        # augment labels
        if self.augment:
            half_width=config.IMAGE_SIZE[0]/2 #for reference coordinatee(center of image)
            half_height=config.IMAGE_SIZE[1]/2
            x_min=utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
            x_max=utils.scale_bbox_coord(x_max, half_height, scale)+ x_shift
            y_min=utils.scale_bbox_coord(y_min, half_width, scale) + y_shift
            y_max=utils.scale_bbox_coord(y_max, half_height, scale)+ y_shift

        #calculating the position of center for BB:
        mid_x=(x_max+x_min)/2
        mid_y=(y_min+y_max)/2
        col=int(mid_x // grid_size_x)
        row=int(mid_y // grid_size_y)

        if 0<=col<config.S and 0<=row<config.S:
            cell=(row, col)
            if cell not in class_names or name==class_names[cell]:
                #inserting class one-hot(binary vector) into ground truth
                one_hot=torch.zeros(config.C)
                one_hot[class_index]=1.0
                ground_truth[row, col, :config.C]=one_hot
                class_names[cell]=name

        #inserting bounding box into ground truth:
        bbox_index=boxes.get(cell, 0)
        if bbox_index<config.B:
            bbox_truth=(
                (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0], # X coord relative to grid square
                (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1], # Y coord relative to gird square
                (x_max - x_min) / config.IMAGE_SIZE[0],             # width 
                (y_max - y_min ) / config.IMAGE_SIZE[1],            # Height
                1.0                                                 # Confidence
             )
        
        #Filling all bbox slots with current bbox which prevents from having "dead" boxes (zeros) at the end
            bbox_start=5*bbox_index+ config.C
            ground_truth[row, col, bbox_start:]=torch.tensor(bbox_truth).repeat(config.B- config.C)
            boxes[cell]=bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        return len(self.dataset)
    
if __name__ == '__main__':
    obj_classes=utils.load_class_array()
    train_set=YoloPascalVocDataset('train', normalize=True, augment=True)
    
