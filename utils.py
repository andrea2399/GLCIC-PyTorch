import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

'''
def gen_input_mask(
        shape, hole_size, hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 is provided,
                holes of size (W, H) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes+1)))
        for _ in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
    return mask
'''

def gen_input_mask(shape, hole_size, hole_area=None):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 2 is provided,
                holes of size (H, W) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    for i in range(bsize):
        # Imposta il numero di buchi a 1
        n_holes = 1
        for _ in range(n_holes):
            # Altezza del buco uguale all'altezza dell'immagine
            hole_h = mask_h
            
            # Larghezza del buco scelta casualmente
            #if isinstance(hole_size, tuple) and len(hole_size) == 1 and isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
            if isinstance(hole_size, tuple) and len(hole_size) == 2:
                hole_w = random.randint(hole_size[0], hole_size[1])
                #print("Sono dentro")    
            else:
                hole_w = hole_size
                #print("Sono fuori")    

            if hole_area is not None:
                offset_x = hole_area[0][0] + random.randint(0, hole_area[1][0] - hole_w)
                offset_y = hole_area[0][1]    
            else: 
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = 0  # Il buco copre tutta l'altezza

            # Riempie il buco con 1.0
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
    return mask

def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))

'''
def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]
'''
def crop(image, mask, size):
    _, _, h, w = image.shape
    top = random.randint(0, max(0, h - size))
    left = random.randint(0, max(0, w - size))
    cropped_image = image[:, :, top:top + size, left:left + size]
    #cropped_mask = mask[:, :, top:top + size, left:left + size]
    return cropped_image
    #, cropped_mask

def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)



def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 2, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 2, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 2, H, W) inpainted with poisson image editing method.
    """
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    
    num_samples = input.shape[0]
    ret = []

    for i in range(num_samples):
        dstimg = np.array(transforms.functional.to_pil_image(input[i]))
        srcimg = np.array(transforms.functional.to_pil_image(output[i]))
        msk = np.array(transforms.functional.to_pil_image(mask[i].squeeze(0)))  # Convert mask to 2D binary mask
        
        # perform inpainting on each channel separately
        out_channels = []
        for channel in range(2):
            dstimg_channel = dstimg[:, :, channel]
            msk_channel = msk
            
            # compute mask's center
            ys, xs = np.where(msk_channel == 255)
            if len(xs) == 0 or len(ys) == 0:
                center = (0, 0)  # default center if mask is empty
            else:
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
            
            # perform inpainting
            dstimg_channel = cv2.inpaint(dstimg_channel, msk_channel, 1, cv2.INPAINT_TELEA)
            out_channels.append(dstimg_channel)
        
        # concatenate the inpainted channels
        out = np.stack(out_channels, axis=-1)
        out = transforms.functional.to_tensor(out)
        ret.append(out)
    
    ret = torch.stack(ret)
    return ret

'''    
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    #mask = (mask > 0.5).float()  # convert to binary mask
    num_samples = input.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(input[i])
        dstimg = np.array(dstimg)
        
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)
        
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)
        
        # perform inpainting on each channel separately
        out_channels = []
        for channel in range(2):
            dstimg_channel = dstimg[:, :, channel]
            msk_channel = msk
            
            # compute mask's center
            xs, ys = [], []
            for j in range(msk_channel.shape[0]):
                for k in range(msk_channel.shape[1]):
                    if msk_channel[j, k] == 255:
                        ys.append(j)
                        xs.append(k)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
            
            # perform inpainting
            dstimg_channel = cv2.inpaint(dstimg_channel, msk_channel, 1, cv2.INPAINT_TELEA)
            out_channels.append(dstimg_channel)
        
        # concatenate the inpainted channels
        out = np.stack(out_channels, axis=-1)
        #if not np.any(srcimg):  # check if srcimg is empty
            # skip seamlessClone if srcimg is empty
            #out = out
        #else:
            #out = cv2.seamlessClone(srcimg, out, msk, center, cv2.NORMAL_CLONE)
        #out = cv2.seamlessClone(srcimg, out, msk, center, cv2.NORMAL_CLONE)
        #out = out[:, :, [2, 1, 0]]  # optional conversion to RGB
        out = transforms.functional.to_tensor(out)
        #out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret
'''

'''
def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 3, H, W) inpainted with poisson image editing method.
    """
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = input.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(input[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret
'''
