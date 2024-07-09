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
        '''
        img = Image.open(self.imgpaths[index])
        img = img.convert(color_format)
        if self.transform is not None:
            img = self.transform(img)
        return img
        '''
        # Path for CBCT image
        cbct_img_path = self.imgpaths[index]
        # Path for corresponding CT image
        #ct_img_path = cbct_img_path.replace('_cbct.png', '_ct.png')
        # Path for corresponding CT image
        ct_img_path = cbct_img_path.replace('_cbct.tif', '_ct.tif')
        #print(f"Loading CBCT image from path: {cbct_img_path}")
        #print(f"Loading CT image from path: {ct_img_path}")
        #cbct_img = Image.open(cbct_img_path).convert('L')  # Convert to grayscale
        #ct_img = Image.open(ct_img_path).convert('L')      # Convert to grayscale
        
        # Open the images using tifffile and convert to float32
        cbct_img = tiff.imread(cbct_img_path)#.astype(np.float32)
        ct_img = tiff.imread(ct_img_path)#.astype(np.float32)
        cbct_img_1 = Image.fromarray(cbct_img)
        ct_img_1 = Image.fromarray(ct_img)
        if self.transform is not None:
            cbct_img_1 = self.transform(cbct_img_1)
            ct_img_1 = self.transform(ct_img_1)       
        cbct_array = np.array(cbct_img_1)
        ct_array = np.array(ct_img_1)
        #cbct_array = np.array(cbct_img)
        #ct_array = np.array(ct_img)
        stacked_img = np.stack([cbct_array, ct_array], axis=-1)#.squeeze(0)
        #stacked_img = stacked_img.permute(0, 3, 1, 2)
        #print(stacked_img.shape)
        # Convert the numpy array back to PIL Image
        #stacked_img = Image.fromarray(stacked_img)
        
        # Ensure the final output is a torch.Tensor
        if not isinstance(stacked_img, torch.Tensor):
            stacked_img = torch.from_numpy(stacked_img)
            stacked_img = stacked_img.permute(2, 0, 1)             
        #if self.transform is not None:
            #stacked_img = self.transform(stacked_img)

        cbct_img = stacked_img
        return cbct_img #,stacked_img
        
    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        #if os.path.isfile(filepath) and imghdr.what(filepath):
            #return True
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
