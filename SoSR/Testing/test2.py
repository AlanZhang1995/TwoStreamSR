import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch

model_path = sys.argv[1]  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
model_name = model_path.split('/')[-1].split('.')[0]
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = '/media/alan/SteinsGate/TSN_testingset/UCF101'
outputdir = '/media/alan/SteinsGate/SR_rgb_frames'

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

dir_input = glob.glob(test_img_folder+'/*')
for No,v in enumerate(dir_input,1):
    scene_name = v.split('/')[-1]
    if not os.path.exists(outputdir+'/{}/{}/'.format(model_name,scene_name)):
        os.makedirs(outputdir+'/{}/{}/'.format(model_name,scene_name))
    else:
        print("Scene No.{} has been done, pass".format(No))
        continue
    
    dir_frames = glob.glob(v + '/*.png')
    dir_frames.sort()

    for i,f in enumerate(dir_frames, 0):
        # read image
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        result_path=outputdir+'/{}/{}/img_{:05d}.png'.format(model_name,scene_name, i)
        cv2.imwrite(result_path, output)
        if not i%100:
            print('Frame {}/{} done'.format(i+1, len(dir_frames)))
    print("Scene No.{} has {} frame and done".format(No,len(dir_frames)))
