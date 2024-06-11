import numpy as np
import os
import imghdr
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
        ct_img_path = cbct_img_path.replace('.png', '_ct.png')
        
        cbct_img = Image.open(cbct_img_path).convert('L')  # Convert to grayscale
        ct_img = Image.open(ct_img_path).convert('L')      # Convert to grayscale

        cbct_array = np.array(cbct_img)
        ct_array = np.array(ct_img)

        stacked_img = np.stack([cbct_array, ct_array], axis=-1)

        # Convert the numpy array back to PIL Image
        stacked_img = Image.fromarray(stacked_img)

        if self.transform is not None:
            stacked_img = self.transform(stacked_img)
        return stacked_img
        
    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
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
