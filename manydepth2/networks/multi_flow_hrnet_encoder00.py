import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from manydepth2.layers import BackprojectDepth, Project3D
import logging
from torch.hub import load_state_dict_from_url
import matplotlib.pyplot as plt
from .hr_layers import Multi_Head_Attention
import pdb
class HRnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.
    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale."""
    def __init__(self, cfg, norm_layer, 
                 num_layers, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear', batch_size=12):
        logging.info('+++++multi_flow_hrnet_encoder00+++++')
        super(HRnetEncoderMatching, self).__init__()
        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True
        self.batch_size = batch_size
        self.num_ch_enc = np.array([64, 18, 36, 72, 144])
        bottom = 256
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4
                
        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None
        # ----------- Structure Definition ----------
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        # stem network
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = self.norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels
        
        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)
                 
        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)
                     
        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # ^^^^^^^^^^^^^^^^^^^^^^ Modifying here ^^^^^^^^^^^^^^^^^^^^^^
        self.backprojector_batch = BackprojectDepth(batch_size=self.batch_size,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector_batch = Project3D(batch_size=self.batch_size,
                                   height=self.matching_height,
                                   width=self.matching_width)
        # --------- Cost Volume Propagation
        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)
        
        self.compute_depth_bins(min_depth_bin, max_depth_bin)
        self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
                                              kernel_size=1, stride=1, padding=0),
                                              nn.ReLU(inplace=True))
        """^^^^^^^^^^^^^^^^^fusion module^^^^^^^^^^^^^^^^^"""
        #self.multi_kernel_attention = Multi_Head_Attention()
        self.reduce_conv = nn.Sequential(nn.Conv2d(bottom + self.num_depth_bins,
                                                   out_channels=bottom,
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True))
        
    def compute_dynamic_flow(self, lookup_pose, _flow, depth, _K, _invK, current_image, lookup_images):
        flow = _flow[:depth.size(0)]
        world_points_depth = self.backprojector_batch(depth, _invK)
        pix_locs_depth, _ = self.projector_batch(world_points_depth, _K, lookup_pose[:,0])
        pix_locs_depth = pix_locs_depth.permute(0, 3, 1, 2)
        # --------------- backprojector_batch 
        pix_coords = self.backprojector_batch.pix_coords.view(self.batch_size, 3, \
            self.matching_height, self.matching_width)[:, :2, :, :]
        pix_coords =  pix_coords[:flow.size(0)]
        normal_static_flow = pix_locs_depth - pix_coords
        dynamic_flow = flow - normal_static_flow
        check_flow = torch.norm(dynamic_flow, dim=1, keepdim=True)
        threshold = 10
        segmentation =  check_flow > threshold
        flow_bwd = self.invert_flow(flow)
        seg_flow_bwd = segmentation*flow_bwd
        static_reference = self.warping(lookup_images, seg_flow_bwd)
        return static_reference

    def invert_flow(self, fwd_flow):
        # Get the backward optical flow. 
        B, _, H, W = fwd_flow.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(fwd_flow.device)  # Shape (2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        coords = grid + fwd_flow  # Shape (B, 2, H, W)
        bwd_flow = torch.zeros_like(fwd_flow)
        coords = torch.round(coords).long()
        coords[:, 0].clamp_(0, W - 1)
        coords[:, 1].clamp_(0, H - 1)
        for b in range(B):
            bwd_flow[b, :, coords[b, 1], coords[b, 0]] = -fwd_flow[b]
        return bwd_flow

    def warping(self, lookup_images, flow_bwd):
        B, _, H, W = flow_bwd.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(flow_bwd.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        new_coords = grid + flow_bwd
        new_coords[:, 0, :, :] = (new_coords[:, 0, :, :] / (W - 1)) * 2 - 1
        new_coords[:, 1, :, :] = (new_coords[:, 1, :, :] / (H - 1)) * 2 - 1
        new_coords = new_coords.permute(0, 2, 3, 1)  # Shape (1, H, W, 2)
        static_reference = F.grid_sample(lookup_images[:,0], new_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        return static_reference

    def compute_depth_bins(self, min_depth_bin, max_depth_bin):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""
        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order
        elif self.depth_binning == 'linear': # Chose Linear As the Depth Binning.
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
        else:
            raise NotImplementedError
        self.depth_bins = torch.from_numpy(self.depth_bins).float()
        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        if self.is_cuda:
            self.warp_depths = self.warp_depths.cuda()
        
    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.
        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).
        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""
        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence
        for batch_idx in range(len(current_feats)):
            volume_shape = (int(self.num_depth_bins), self.matching_height, self.matching_width)
            # 1 + 1/6 + 1/6
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]
            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            # loop through ref images adding to the current cost volume
            world_points = self.backprojector(self.warp_depths, _invK)
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]
                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue
                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                # ---------- 1.Feature Warping Based On the Predefined Depth Bins ----------
                pix_locs, _ = self.projector(world_points, _K, lookup_pose)
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)
                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                # -------------------------------- 1. ------------------------------
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)
                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()
                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask
                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(1) * edge_mask
                cost_volume = cost_volume + diffs
                counts = counts + (diffs > 0).float()
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)
            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)
        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)
        return batch_cost_volume, cost_volume_masks
        
    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""
        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp
        
    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""
        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()
        return confidence_mask
        
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),)
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)
        
    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels
        
    def feature_extraction(self, image, return_all_feats=False):
        x = image
        features = []
        list18 = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        list18.append(x)
        x = self.layer1(x)
        if return_all_feats:
            return x, features, list18
        else:
            return x
        
    def forward(self, current_image, lookup_images, poses, flow, depth, K, invK,
                min_depth_bin=None, max_depth_bin=None, using_flow=False):
        # feature extraction
        self.features, features, list18 = self.feature_extraction(current_image, return_all_feats=True)
        current_feats = self.features
        # B, 64, H/4, W/4
        # feature extraction on lookup images - disable gradients to save memory
        """^^^^^^^^Static Flow Extraction^^^^^^^^"""
        if using_flow:
            with torch.no_grad():
                alpha = 0.2
                static_reference = self.compute_dynamic_flow(poses, flow, depth, K, invK, current_image, lookup_images)
                static_reference = F.interpolate(static_reference, scale_factor=4, mode='bilinear', align_corners=False)
                if len(lookup_images.shape) > 4:
                    static_reference = static_reference[:, None]
                lookup_images = alpha*static_reference + (1 - alpha)*lookup_images

        """^^^^^^^^Static Flow Extraction^^^^^^^^"""
        with torch.no_grad():
            if self.adaptive_bins:
                self.compute_depth_bins(min_depth_bin, max_depth_bin)
            batch_size, num_frames, chns, height, width = lookup_images.shape
            lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
            lookup_feats = self.feature_extraction(lookup_images, return_all_feats=False)
            _, chns, height, width = lookup_feats.shape
            lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)
            # warp features to find cost volume
            cost_volume, missing_mask = \
                self.match_features(current_feats, lookup_feats, poses, K, invK)
            confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                           (1 - missing_mask.detach()))
        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)
        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)
        post_matching_feats = self.reduce_conv(torch.cat([self.features, cost_volume], 1))
        
        x = post_matching_feats
        list36 = []
        list72 = []
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        list18.append(y_list[0])
        list36.append(y_list[1])
        
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']): # 3
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']: # 2
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        list18.append(y_list[0])
        list36.append(y_list[1])
        list72.append(y_list[2])
        
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']): #4
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
                    # here generate new scale features (downsample) 
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        list18.append(x[0])
        list36.append(x[1])
        list72.append(x[2])
        mixed_features = [list18] + [list36] + [list72] + [x[3]]
        # B, 64, H, W
        # 4, B, 18, H/2, W/2
        # 3, B, 32, H/4, W/4
        # 2, B, 72, H/8, W/8
        # B, 144, H/16, W/16
        return features + mixed_features, lowest_cost, confidence_mask
        
    def cuda(self):
        super().cuda()
        self.backprojector_batch.cuda()
        self.backprojector.cuda()
        self.projector_batch.cuda()
        self.projector.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cuda()
        
    def cpu(self):
        super().cpu()
        self.backprojector_batch.cpu()
        self.backprojector.cpu()
        self.projector_batch.cpu()
        self.projector.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cpu()
        
    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError

# ------------ HIGH RESOLUTION NETWORK ------------
logger = logging.getLogger('hrnet_backbone')
__all__ = ['hrnet18', 'hrnet32', 'hrnet48','hrnet64']
model_urls = {
    'hrnet18_imagenet': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32_imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48_imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
}
def visual_feature(features):
    for a in range(len(features)):
        feature_map = features[a].squeeze(0).cpu()
        n,h,w = feature_map.size()
        logging.info("{} channel in stage {}".format(n,a))
        list_mean = []
        sum_feature_map = torch.sum(feature_map,0)
        #sum_feature_map,_ = torch.max(feature_map,0)
        for i in range(n):
            list_mean.append(torch.mean(feature_map[i]))
        sum_mean = sum(list_mean)
        feature_map_weighted = torch.ones([n,h,w])
        for i in range(n):
            feature_map_weighted[i,:,:] = (torch.mean(feature_map[i]) / sum_mean) * feature_map[i,:,:]
        sum_feature_map_weighted = torch.sum(feature_map_weighted,0)
        plt.imshow(sum_feature_map,cmap= 'magma')
        plt.savefig('feature_viz/{}_stage.png'.format(a))
        #plt.savefig('feature_viz_ori/{}_stage.png'.format(a))
        plt.imshow(sum_feature_map_weighted,cmap = 'magma')
        plt.savefig('feature_viz/{}_stage_weighted.png'.format(a))
        #plt.savefig('feature_viz_ori/{}_stage_weighted.png'.format(a))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion), )
        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))
        return nn.Sequential(*layers)
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)
    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)
    def get_num_inchannels(self):
        return self.num_inchannels
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True
                        )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck}

def sum_params(model):
    s = []
    for p in model.parameters():
        dims = p.size()
        n = p.cpu().data.numpy()
        s.append(np.sum(n))
    return sum(s)

def _hrnet(arch, num_layers, input_height, input_width, min_depth_bin, max_depth_bin, num_depth_bins, adaptive_bins, depth_binning, \
            pretrained, progress, **kwargs):
    from .hrnet_config import MODEL_CONFIGS
    model = HRnetEncoderMatching(MODEL_CONFIGS[arch], None, num_layers=num_layers, input_height=input_height, input_width=input_width, \
                                min_depth_bin=min_depth_bin, max_depth_bin=max_depth_bin, num_depth_bins=num_depth_bins, \
                                adaptive_bins=adaptive_bins, depth_binning=depth_binning)
    #print(sum_params(model), '===========1')
    if pretrained:
        if arch == 'hrnet64':
            arch = 'hrnet32_imagenet'
            model_url = model_urls[arch]
            loaded_state_dict = load_state_dict_from_url(model_url,
                                                  progress=progress)
            #add weights demention to adopt input change
            exp_layers = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']
            lista = ['transition1.0.0.weight', 'transition1.1.0.0.weight', 'transition2.2.0.0.weight', 'transition3.3.0.0.weight']
            for k,v in loaded_state_dict.items() :
                if k not in exp_layers:
                    if ('layer' not in k) and 'conv' in k or k in lista and len(v.size()) > 1:
                        if k in ['transition1.0.0.weight' , 'transition1.1.0.0.weight']:
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2,0)
                        else:
                            loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2,0)

                    if 'fuse_layer' in k and 'weight' in k and len(v.size()) > 1:
                        loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                        loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2,0)

                    if 'layer' not in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0) 
                        loaded_state_dict[k] = v.squeeze(1) 
                    if 'fuse_layer' in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0) 
                        loaded_state_dict[k] = v.squeeze(1) 
                    if len(loaded_state_dict[k].size()) == 2:
                        loaded_state_dict[k] = loaded_state_dict[k].squeeze(1)
                    # for multi-input 
                    #if k == 'conv1.weight':
                    #  loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
        else:
            #arch = arch + '_imagenet'
            #model_url = model_urls[arch]
            #loaded_state_dict = load_state_dict_from_url(model_url, progress=progress)
            import timm
            arch = arch + '_imagenet'
            loaded_state_dict = timm.create_model('hrnet_w18', pretrained=True).state_dict()
        #if k == 'conv1.weight':
        #    loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
        model.load_state_dict({k: v for k,v in loaded_state_dict.items() if k in model.state_dict()}, strict=False)
    #print(sum_params(model), '===========2')
    return model

def hrnet18(num_layers, input_height, input_width, min_depth_bin, max_depth_bin, num_depth_bins, adaptive_bins, depth_binning,
            pretrained=True, progress=True, **kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', num_layers, input_height, input_width, min_depth_bin, max_depth_bin, num_depth_bins, adaptive_bins, depth_binning,
                  pretrained, progress, **kwargs)

def hrnet32(pretrained=True, progress=True, **kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', pretrained, progress,
                   **kwargs)

def hrnet48(pretrained=True, progress=True, **kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', pretrained, progress,
                   **kwargs)

def hrnet64(pretrained=True, progress=True, **kwargs):
    r"""HRNet-64 model
    """
    return _hrnet('hrnet64', pretrained, progress,
                   **kwargs)