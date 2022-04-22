import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import os
from PIL import Image

## Image Dataloader
## Image Dataloader
class SurgicalToolDataset(Dataset):
    """
        This class provides the implementation for accessing the surgical tool dataset.
    """
    def __init__(self,
                 dataset_dir,
                 split,
                 transform
                ):
        """
            dataset_dir: the path to the dataset folder
            split: either "train" or "test"
            transform: torchvison transformation
        """
        assert split == "train" or split == "test", "split can only be train or test."
        self.split = split
        self.dataset_dir = dataset_dir
        self.transform = transform

        split_file = os.path.join(self.dataset_dir, 'Surgical-Dataset/Test-Train Groups')
        if split == "train":
            split_file += "/train-obj_detector.txt"
        elif split == "test":
            split_file += "/test-obj_detector.txt"
        
        
        with open(split_file) as file:
            # we get the list of the directory of images for this split
            self.lines = file.readlines()
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        """
            Return:
                img: torch tensor type, with shape of C x H x W
                target: dictionary with keys "boxes" and "labels". Tensor type.
        """
        data_name = self.lines[idx].split("/")[-1]
        data_name = data_name.split(".")[0]
        image_dir = os.path.join(self.dataset_dir, "Surgical-Dataset/Images/All/images/" + data_name + ".jpg")
        target_dir = os.path.join(self.dataset_dir, "Surgical-Dataset/Labels/label object names/" + data_name + ".txt")
        
        img = io.imread(image_dir)
        orig_size = img.shape
        img=Image.open(image_dir)
        
        img = self.transform(img)
        size = img.shape[1:]
        
        target = {"boxes": [],
                  "labels": [],
                  "orig_size": torch.tensor(orig_size), # H x W
                  "size": torch.tensor(size), # H x W
                 }

        # construct the target from target file:
        with open(target_dir) as file:
            objects = file.readlines()
            
        for obj in objects:
            features = obj.split()
            label = int(features[0])
            # cxcyhw:
            box = [float(features[1]), 
                   float(features[2]), 
                   float(features[3]), 
                   float(features[4])]
            target['boxes'].append(box)
            target['labels'].append(label)
        
        target['boxes'] = torch.tensor(target['boxes'])
        target['labels'] = torch.tensor(target['labels'])
  
        
        return img, target