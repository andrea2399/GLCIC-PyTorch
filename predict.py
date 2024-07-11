import tifffile as tiff
import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import CompletionNetwork
from utils import poisson_blend, gen_input_mask


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_folder')
parser.add_argument('output_folder')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--img_size_1', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)
parser.add_argument('--hole_size', type=int, default=50)

def main(args):
    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_folder = os.path.expanduser(args.input_folder)
    args.output_folder = os.path.expanduser(args.output_folder)
    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    gpu = torch.device('cuda:0')

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1).to(gpu)
    mpv = 0.2989 * mpv[:, 0:1, :, :] + 0.5870 * mpv[:, 1:2, :, :] + 0.1140 * mpv[:, 2:3, :, :]
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location=gpu))

    # =============================================
    # Scan the input folder for CBCT-CT pairs
    # =============================================
    cbct_files = sorted([os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if '_cbct' in f])
    ct_files = sorted([os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if '_ct' in f])
    if len(cbct_files) != len(ct_files):
        raise Exception('The number of CBCT and CT files must be the same.')

    # =============================================
    # Generate a single mask
    # =============================================
    # Load the first CBCT image to get the dimensions
    cbct_img_example = tiff.imread(cbct_files[0])
    cbct_img_example = Image.fromarray(cbct_img_example)
    cbct_img_example = transforms.Resize((args.img_size, args.img_size_1))(cbct_img_example)
    cbct_img_example = transforms.CenterCrop((args.img_size, args.img_size_1))(cbct_img_example)
    cbct_img_example = transforms.ToTensor()(cbct_img_example)
    
    # Create a mask
    mask = gen_input_mask(
        shape=(1, 1, args.img_size, args.img_size_1),
        hole_size=(          
            (args.hole_min_w, args.hole_max_w),
            (args.hole_min_h, args.hole_max_h),
            #(args.hole_size, args.hole_size),
            #(args.hole_min_h, args.hole_max_h),
        ),
        max_holes=args.max_holes,
    )

    # =============================================
    # Process each pair
    # =============================================
    for idx, (cbct_file, ct_file) in enumerate(zip(cbct_files, ct_files)):
        # Load and process CBCT image
        cbct_img = tiff.imread(cbct_file)
        cbct_img = Image.fromarray(cbct_img)
        cbct_img = transforms.Resize((args.img_size, args.img_size_1))(cbct_img)
        cbct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(cbct_img)
        cbct_img = transforms.ToTensor()(cbct_img)

        # Load and process CT image
        ct_img = tiff.imread(ct_file)
        ct_img = Image.fromarray(ct_img)
        ct_img = transforms.Resize((args.img_size, args.img_size_1))(ct_img)
        ct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(ct_img)
        ct_img = transforms.ToTensor()(ct_img)

        # Stack images along the channel dimension
        stacked_img = torch.cat([cbct_img, ct_img], dim=0).unsqueeze(0)  # Add batch dimension

        # Inpaint
        model.eval()
        with torch.no_grad():
            x_mask = stacked_img - stacked_img * mask + mpv * mask
            x_mask[:, :2, :, :] = stacked_img[:, :2, :, :] 
            x_mask[:, :1, :, :] = stacked_img [:, :1, :, :]  - stacked_img [:, :1, :, :] * mask + mpv * mask 
            input = torch.cat((x_mask, mask), dim=1)
            output = model(input)
            inpainted = poisson_blend(x_mask[:, :1, :, :], output, mask)

            # Save combined images
            base_name = f'slice_{idx:03d}'
            # Save individual images
            tiff.imwrite(os.path.join(args.output_folder, f'{base_name}_input_cbct.tif'), stacked_img[0, 0].cpu().numpy())
            tiff.imwrite(os.path.join(args.output_folder, f'{base_name}_x_mask_cbct.tif'), x_mask[0, 0].cpu().numpy())
            tiff.imwrite(os.path.join(args.output_folder, f'{base_name}_inpainted_cbct.tif'), inpainted[0].cpu().numpy())
            tiff.imwrite(os.path.join(args.output_folder, f'{base_name}_output_cbct.tif'), output[0].cpu().numpy())

    print('Output images were saved in %s.' % args.output_folder)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
