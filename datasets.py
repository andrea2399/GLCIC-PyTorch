import numpy as np
import os
import imghdr
import torch
import tifffile as tiff
import torch.utils.data as data
from PIL import Image


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, recursive_search=False):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir, walk=recursive_search)

    def __len__(self):
        return len(self.imgpaths)
    
    def __getitem__(self, index, color_format='RGB'):
        # Path for CBCT image
        cbct_img_path = self.imgpaths[index]
        # Path for corresponding CT image
        ct_img_path = cbct_img_path.replace('_cbct.tif', '_ct.tif')
 
        # Open the images using tifffile
        cbct_img = tiff.imread(cbct_img_path)
        ct_img = tiff.imread(ct_img_path)
        cbct_img_1 = Image.fromarray(cbct_img)
        ct_img_1 = Image.fromarray(ct_img)
        if self.transform is not None:
            cbct_img_1 = self.transform(cbct_img_1)
            ct_img_1 = self.transform(ct_img_1)       
        cbct_array = np.array(cbct_img_1)
        ct_array = np.array(ct_img_1)
        stacked_img = np.stack([cbct_array, ct_array], axis=-1)
        
        # Ensure the final output is a torch.Tensor
        if not isinstance(stacked_img, torch.Tensor):
            stacked_img = torch.from_numpy(stacked_img)
            stacked_img = stacked_img.permute(2, 0, 1)             

        x = stacked_img
        return x
        
    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and filepath.endswith('.tif'):
            return True
        return False

    def __load_imgpaths_from_dir(self, dirpath, walk=False):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, _, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if not self.__is_imgfile(path):
                    continue
                imgpaths.append(path)
        return imgpaths
