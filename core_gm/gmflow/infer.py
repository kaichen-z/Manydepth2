import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
from data import build_train_dataset
from gmflow.gmflow import GMFlow
from loss import flow_loss_func
from evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                      create_sintel_submission, create_kitti_submission, inference_on_dir)
from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from PIL import Image
import cv2
import flow_vis

def sum_params(model):
    s = []
    for p in model.parameters():
        dims = p.size()
        n = p.cpu().data.numpy()
        s.append(np.sum(n))
    return sum(s)

if __name__ == '__main__':
    image_size = (1248, 384)
    #img1 = '/mnt/nas/kaichen/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000000.png'
    #img2 = '/mnt/nas/kaichen/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000001.png'

    img1 = '/mnt/nas/kaichen/kitti/2011_09_26/2011_09_26_drive_0015_sync/image_03/data/0000000197.png'
    img2 = '/mnt/nas/kaichen/kitti/2011_09_26/2011_09_26_drive_0015_sync/image_03/data/0000000198.png'

    #img1 = '/mnt/nas/kaichen/kitti/2011_09_28/2011_09_28_drive_0001_sync/image_03/data/0000000058.png'
    #img2 = '/mnt/nas/kaichen/kitti/2011_09_28/2011_09_28_drive_0001_sync/image_03/data/0000000059.png'
    #image1 = Image.open(img1)
    #image2 = Image.open(img2)
    #image1 = np.array(image1).astype(np.uint8)[..., :3]
    #image2 = np.array(image2).astype(np.uint8)[..., :3]
    #image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    #image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    #img1 = '/mnt/nas/kaichen/kitti_scene/testing/image_3/000000_10.png'
    #img2 = '/mnt/nas/kaichen/kitti_scene/testing/image_3/000000_11.png'

    #img1 = '/mnt/nas/kaichen/kitti_scene/training/image_2/000000_10.png'
    #img2 = '/mnt/nas/kaichen/kitti_scene/training/image_2/000000_11.png'
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]

    image1 = cv2.resize(image1, image_size, interpolation = cv2.INTER_AREA)
    image2 = cv2.resize(image2, image_size, interpolation = cv2.INTER_AREA)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().cuda()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float().cuda()
    print(image1.size(), '+++++1')

    feature_channels = 128
    num_scales = 1
    upsample_factor = 8
    num_head = 1
    attention_type = 'swin'
    ffn_dim_expansion = 4
    num_transformer_layers = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GMFlow(feature_channels=feature_channels,
                num_scales=num_scales,
                upsample_factor=upsample_factor,
                num_head=num_head,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                num_transformer_layers=num_transformer_layers).cuda()
    address = '/mnt/nas/kaichen/eng/OF/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth'
    checkpoint = torch.load(address)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights)
    attn_splits_list = [2]
    corr_radius_list = [-1]
    prop_radius_list = [-1]
    batch = 8
    for i in range(1024):   
        print('===== +++++', i)
        print(image1.max(), image1.min(), "+++++ ---- +++++", image1.mean(), image1.size())
        print(image1[:, 0, 0])
        print(image2[:, 0, 0])
        """
        tensor([21., 22., 18.], device='cuda:0')
        tensor([27., 28., 38.], device='cuda:0')
        """
        with torch.no_grad():
            results_dict = model(image1[None].repeat(batch, 1, 1, 1), image2[None].repeat(batch, 1, 1, 1),
                                attn_splits_list=attn_splits_list,
                                corr_radius_list=corr_radius_list,
                                prop_radius_list=prop_radius_list,)
            flow_preds = results_dict['flow_preds'][-1]  # C: from the first version to the second one.
            print('sum-----', sum_params(model)) # -4943.652777135372
            print('flow----', flow_preds[0].permute(1,2,0).sum())
            bgr_for_vis0 = flow_vis.flow_to_color((flow_preds[0].permute(1,2,0).cpu().numpy()), convert_to_bgr=True)
            img1m_cv = image1.cpu().permute(1,2,0).numpy()
            img2m_cv = image2.cpu().permute(1,2,0).numpy()
            cv2.imwrite('test/%s_FLOW_1.png'%i, np.concatenate((bgr_for_vis0, img1m_cv, img2m_cv), 1))
        break

