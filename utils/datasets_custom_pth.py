import os
import pickle
import torch
from torch.utils.data import Dataset

class PthDataset(Dataset):
    def __init__(self, folder_type, saved_tensors_dir = "saved_tensors"):
        os.path.exists(os.path.join(saved_tensors_dir, folder_type))
        self.folder_type = folder_type
        self.imgs_dir = os.path.join(saved_tensors_dir, folder_type, "imgs")
        self.paths_dir = os.path.join(saved_tensors_dir, folder_type, "paths")
        self.targets_dir = os.path.join(saved_tensors_dir, folder_type, "targets")
        self.shapes_dir = os.path.join(saved_tensors_dir, folder_type, "shapes")
        
        self.target = torch.load(os.path.join(self.targets_dir, f"0_targets_{self.folder_type}.pth"))
        with open(os.path.join(saved_tensors_dir, folder_type, "last_idx.txt")) as f:
            self.last_idx = int(f.read())
        
    def __len__(self):
        return self.last_idx + 1
    
    def __getitem__(self, idx):
        img_pth = os.path.join(self.imgs_dir, f"{idx}_imgs_{self.folder_type}.pth")
        path_pth = os.path.join(self.paths_dir, f"{idx}_paths_{self.folder_type}.pkl")
        target_pth = os.path.join(self.targets_dir, f"{idx}_targets_{self.folder_type}.pth")
        img = torch.load(img_pth)
        with open(path_pth, "rb") as f:
            path = pickle.load(f)
        target = torch.load(target_pth)
        return img, target, path
        