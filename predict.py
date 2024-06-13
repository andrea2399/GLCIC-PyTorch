'''
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
parser.add_argument('cbct_img')
parser.add_argument('ct_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--img_size_1', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)


def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.cbct_img = os.path.expanduser(args.cbct_img)
    args.ct_img = os.path.expanduser(args.ct_img)
    args.output_img = os.path.expanduser(args.output_img)
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
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    #img = Image.open(args.input_img)
    #img = transforms.Resize((args.img_size, args.img_size_1))(img)
    #img = transforms.CenterCrop((args.img_size, args.img_size_1))(img)
    #x = transforms.ToTensor()(img)
    #x = torch.unsqueeze(x, dim=0)
    # Load and process CBCT image
    cbct_img = Image.open(args.cbct_img).convert('L')
    cbct_img = transforms.Resize((args.img_size, args.img_size_1))(cbct_img)
    cbct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(cbct_img)
    #cbct_img = transforms.ToTensor()(cbct_img)

    # Load and process CT image
    ct_img = Image.open(args.ct_img).convert('L')
    ct_img = transforms.Resize((args.img_size, args.img_size_1))(ct_img)
    ct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(ct_img)
    #ct_img = transforms.ToTensor()(ct_img)

    # Create stacked image
    cbct_array = np.array(cbct_img)
    ct_array = np.array(ct_img)
    stacked_img = np.stack([cbct_array, ct_array], axis=0)  # Change axis to 0 for channel-first
    #stacked_img = torch.tensor(stacked_img).unsqueeze(0).float().to(gpu)  # Add batch dimension and convert to tensor
    stacked_img = transforms.ToTensor() (stacked_img)
    stacked_img = torch.unsqueeze(stacked_img, dim=0).to(gpu)
    #stacked_img = np.stack([cbct_array, ct_array], axis=-1)
    #stacked_img = Image.fromarray(stacked_img)
    #stacked_img = transforms.ToTensor()(stacked_img)
    #stacked_img = stacked_img.to(gpu)
    #stacked_img = torch.stack((cbct_img, ct_img), dim=0)
    #stacked_img = torch.unsqueeze(stacked_img, dim=0)  # Add batch dimension
    # create mask
    mask = gen_input_mask(
        shape=(stacked_img.shape[0], 1, stacked_img.shape[2], stacked_img.shape[3]),
        hole_size=(
            (args.hole_min_w, args.hole_max_w)),
        ).to(gpu)

    # inpaint
    model.eval()
    with torch.no_grad():
        x_mask = stacked_img - stacked_img * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        #imgs = torch.cat((x, x_mask, inpainted), dim=0)
        #save_image(imgs, args.output_img, nrow=3)
        save_image(stacked_img, os.path.join(args.output_img, 'input.png'))
        save_image(x_mask, os.path.join(args.output_img, 'x_mask.png'))
        save_image(inpainted, os.path.join(args.output_img, 'inpainted.png'))
    print('output img was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
'''

'''
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
parser.add_argument('cbct_img')
parser.add_argument('ct_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--img_size_1', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)


def main(args):
    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.cbct_img = os.path.expanduser(args.cbct_img)
    args.ct_img = os.path.expanduser(args.ct_img)
    args.output_img = os.path.expanduser(args.output_img)
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
    model = CompletionNetwork().to(gpu)
    model.load_state_dict(torch.load(args.model, map_location=gpu))

    # =============================================
    # Predict
    # =============================================
    # Load and process CBCT image
    cbct_img = Image.open(args.cbct_img).convert('L')
    cbct_img = transforms.Resize((args.img_size, args.img_size_1))(cbct_img)
    cbct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(cbct_img)
    cbct_img = transforms.ToTensor()(cbct_img).to(gpu)

    # Load and process CT image
    ct_img = Image.open(args.ct_img).convert('L')
    ct_img = transforms.Resize((args.img_size, args.img_size_1))(ct_img)
    ct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(ct_img)
    ct_img = transforms.ToTensor()(ct_img).to(gpu)

    # Stack images along the channel dimension
    stacked_img = torch.cat([cbct_img, ct_img], dim=0).unsqueeze(0)  # Add batch dimension

    # Create mask
    mask = gen_input_mask(
        shape=(stacked_img.shape[0], 1, stacked_img.shape[2], stacked_img.shape[3]),
        hole_size=(
            (args.hole_min_w, args.hole_max_w)),
            #(args.hole_min_h, args.hole_max_h),
    ).to(gpu)

    # Inpaint
    model.eval()
    with torch.no_grad():
        x_mask = stacked_img - stacked_img * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)

        # Save combined images
        save_image(stacked_img[0], os.path.join(args.output_img, 'input.png'))
        save_image(x_mask[0], os.path.join(args.output_img, 'x_mask.png'))
        save_image(inpainted[0], os.path.join(args.output_img, 'inpainted.png'))

        # Save individual channels
        save_image(stacked_img[0, 0], os.path.join(args.output_img, 'input_cbct.png'))
        save_image(stacked_img[0, 1], os.path.join(args.output_img, 'input_ct.png'))
        save_image(x_mask[0, 0], os.path.join(args.output_img, 'x_mask_cbct.png'))
        save_image(x_mask[0, 1], os.path.join(args.output_img, 'x_mask_ct.png'))
        save_image(inpainted[0, 0], os.path.join(args.output_img, 'inpainted_cbct.png'))
        save_image(inpainted[0, 1], os.path.join(args.output_img, 'inpainted_ct.png'))
    print('Output images were saved in %s.' % args.output_img)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
'''

'''
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
parser.add_argument('cbct_img')
parser.add_argument('ct_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--img_size_1', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)

def main(args):
    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.cbct_img = os.path.expanduser(args.cbct_img)
    args.ct_img = os.path.expanduser(args.ct_img)
    args.output_img = os.path.expanduser(args.output_img)
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
    model = CompletionNetwork().to(gpu)
    model.load_state_dict(torch.load(args.model, map_location=gpu))

    # =============================================
    # Predict
    # =============================================
    # Load and process CBCT image
    cbct_img = Image.open(args.cbct_img).convert('L')
    cbct_img = transforms.Resize((args.img_size, args.img_size_1))(cbct_img)
    cbct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(cbct_img)
    cbct_img = transforms.ToTensor()(cbct_img).to(gpu)

    # Load and process CT image
    ct_img = Image.open(args.ct_img).convert('L')
    ct_img = transforms.Resize((args.img_size, args.img_size_1))(ct_img)
    ct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(ct_img)
    ct_img = transforms.ToTensor()(ct_img).to(gpu)

    stacked_img = torch.cat([cbct_img, ct_img], dim=0)#.unsqueeze(0)
    
    # Create mask for CBCT image only
    mask_cbct = gen_input_mask(
        shape=(1, 1, cbct_img.shape[1], cbct_img.shape[2]),
        hole_size=(
            (args.hole_min_w, args.hole_max_w)),
    ).to(gpu)

    # Create mask for CT image that preserves it (all zeros)
    mask_ct = torch.zeros((1, 1, ct_img.shape[1], ct_img.shape[2]), device=gpu)

    # Stack masks along the channel dimension
    mask = torch.cat((mask_cbct, mask_ct), dim=1)

    # Inpaint CBCT image
    model.eval()
    with torch.no_grad():
        x_mask = stacked_img - stacked_img * mask + mpv * mask
        input_cbct = torch.cat((x_mask, mask), dim=1)
        output = model(input_cbct)
        inpainted_cbct = poisson_blend(x_mask, output, mask)

        # Save combined images
        save_image(stacked_img[0], os.path.join(args.output_img, 'input.png'))
        save_image(x_mask[0], os.path.join(args.output_img, 'x_mask.png'))
        save_image(inpainted_cbct[0], os.path.join(args.output_img, 'inpainted.png'))

        # Save individual channels
        save_image(stacked_img[0, 0], os.path.join(args.output_img, 'input_cbct.png'))
        save_image(stacked_img[0, 1], os.path.join(args.output_img, 'input_ct.png'))
        save_image(x_mask[0, 0], os.path.join(args.output_img, 'x_mask_cbct.png'))
        save_image(x_mask[0, 1], os.path.join(args.output_img, 'x_mask_ct.png'))
        save_image(inpainted_cbct[0, 0], os.path.join(args.output_img, 'inpainted_cbct.png'))
        save_image(inpainted_cbct[0, 1], os.path.join(args.output_img, 'inpainted_ct.png'))

    print('Output images were saved in %s.' % args.output_img)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
'''
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
parser.add_argument('cbct_img')
parser.add_argument('ct_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--img_size_1', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)

def main(args):
    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.cbct_img = os.path.expanduser(args.cbct_img)
    args.ct_img = os.path.expanduser(args.ct_img)
    args.output_img = os.path.expanduser(args.output_img)
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
    model = CompletionNetwork().to(gpu)
    model.load_state_dict(torch.load(args.model, map_location=gpu))

    # =============================================
    # Predict
    # =============================================
    # Load and process CBCT image
    cbct_img = Image.open(args.cbct_img).convert('L')
    cbct_img = transforms.Resize((args.img_size, args.img_size_1))(cbct_img)
    cbct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(cbct_img)
    cbct_img = transforms.ToTensor()(cbct_img).to(gpu)

    # Load and process CT image
    ct_img = Image.open(args.ct_img).convert('L')
    ct_img = transforms.Resize((args.img_size, args.img_size_1))(ct_img)
    ct_img = transforms.CenterCrop((args.img_size, args.img_size_1))(ct_img)
    ct_img = transforms.ToTensor()(ct_img).to(gpu)

    stacked_img = torch.cat([cbct_img, ct_img], dim=0).unsqueeze(0)  # Add batch dimension

    # Create mask for CBCT image only
    mask_cbct = gen_input_mask(
        shape=(cbct_img.shape[0], 1, cbct_img.shape[1], cbct_img.shape[2]),
        hole_size=(args.hole_min_w, args.hole_max_w)
    ).to(gpu)

    # Create mask for CT image that preserves it (all zeros)
    mask_ct = torch.zeros((ct_img.shape[0], 1, ct_img.shape[1], ct_img.shape[2]), device=gpu)
    
    # Stack masks along the channel dimension
    mask = torch.cat((mask_cbct, mask_ct), dim=0)

    #cbct_img = cbct_img.unsqueeze(0)
    #ct_img = ct_img.unsqueeze(0)
    # Inpaint CBCT image
    model.eval()
    with torch.no_grad():
        #x_mask_cbct = cbct_img -cbct_img * mask_cbct + mpv * mask_cbct
        #x_mask_ct = ct_img - ct_img * mask_ct + mpv * mask_ct
        x_mask = stacked_img - stacked_img * mask + mpv * mask
        #x_mask = torch.cat((x_mask_cbct, x_mask_ct), dim=0)
        input_cbct = torch.cat((x_mask, mask), dim=1)
        output = model(input_cbct)
        inpainted_cbct = poisson_blend(x_mask, output, mask)

        # Save combined images
        save_image(stacked_img[0], os.path.join(args.output_img, 'input.png'))
        save_image(x_mask[0], os.path.join(args.output_img, 'x_mask.png'))
        save_image(inpainted_cbct[0], os.path.join(args.output_img, 'inpainted.png'))

        # Save individual channels
        save_image(stacked_img[0, 0], os.path.join(args.output_img, 'input_cbct.png'))
        save_image(stacked_img[0, 1], os.path.join(args.output_img, 'input_ct.png'))
        save_image(x_mask[0, 0], os.path.join(args.output_img, 'x_mask_cbct.png'))
        save_image(x_mask[0, 1], os.path.join(args.output_img, 'x_mask_ct.png'))
        save_image(inpainted_cbct[0, 0], os.path.join(args.output_img, 'inpainted_cbct.png'))
        save_image(inpainted_cbct[0, 1], os.path.join(args.output_img, 'inpainted_ct.png'))

    print('Output images were saved in %s.' % args.output_img)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

