# ------------------------------------------------------------------------------
# Demo code.
# Example command:
# python tools/demo.py --cfg PATH_TO_CONFIG_FILE \
#   --input-files PATH_TO_INPUT_FILES \
#   --output-dir PATH_TO_OUTPUT_DIR
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import cv2
import os
import pprint
import logging
import time
import glob

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import tools._init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import save_debug_images
from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation,creat_panoptic_annotation
import segmentation.data.transforms.transforms as T
from segmentation.utils import AverageMeter



def read_image(file_name, format=None):
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


class CityscapesMeta(object):
    def __init__(self):
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.label_divisor = 1000
        self.ignore_label = 255

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap

class Deeplab(object):
    def __init__(self, **kwargs):
        config.defrost()
        config.merge_from_file('configs/panoptic_deeplab_X101_32x8d_os32_cityscapes.yaml')
        config.freeze()
        self.output_dir = 'output'
        model_state_file =  'configs/panoptic_deeplab_X101_32x8d_os32_cityscapes.pth'
        self.logger = logging.getLogger('demo')
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called
            setup_logger(output=self.output_dir, name='demo')

        # logger.info(pprint.pformat(args))
        self.logger.info(config)

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        gpus = list(config.TEST.GPUS)
        if len(gpus) > 1:
            raise ValueError('Test only supports single core.')
        self.device = torch.device('cuda:{}'.format(gpus[0]))
        # build model
        model = build_segmentation_model_from_cfg(config)
        self.logger.info("Model:\n{}".format(model))
        self.model = model.to(self.device)
        if 'cityscapes' in config.DATASET.DATASET:
            self.meta_dataset = CityscapesMeta()
        else:
            raise ValueError("Unsupported dataset: {}".format(config.DATASET.DATASET))
        # load model
        if os.path.isfile(model_state_file):
            model_weights = torch.load(model_state_file)
            if 'state_dict' in model_weights.keys():
                model_weights = model_weights['state_dict']
                self.logger.info('Evaluating a intermediate checkpoint.')
            model.load_state_dict(model_weights, strict=True)
            self.logger.info('Test model loaded from {}'.format(model_state_file))
        else:
            if not config.DEBUG.DEBUG:
                raise ValueError('Cannot find test model.')
    def main(self,frame,index,total):
        self.model.eval()

        # build image demo transform
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    config.DATASET.MEAN,
                    config.DATASET.STD
                )
            ]
        )

        net_time = AverageMeter()
        post_time = AverageMeter()
        try:
            with torch.no_grad():
                raw_image = frame
                # pad image
                raw_shape = raw_image.shape[:2]
                raw_h = raw_shape[0]
                raw_w = raw_shape[1]
                new_h = (raw_h + 31) // 32 * 32 + 1
                new_w = (raw_w + 31) // 32 * 32 + 1
                input_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                input_image[:, :] = config.DATASET.MEAN
                input_image[:raw_h, :raw_w, :] = raw_image

                image, _ = transforms(input_image, None)
                image = image.unsqueeze(0).to(self.device)

                # network
                start_time = time.time()
                out_dict = self.model(image)
                torch.cuda.synchronize(self.device)
                net_time.update(time.time() - start_time)

                # post-processing
                start_time = time.time()
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])

                panoptic_pred, center_pred = get_panoptic_segmentation(
                    semantic_pred,
                    out_dict['center'],
                    out_dict['offset'],
                    thing_list=self.meta_dataset.thing_list,
                    label_divisor=self.meta_dataset.label_divisor,
                    stuff_area=config.POST_PROCESSING.STUFF_AREA,
                    void_label=(
                            self.meta_dataset.label_divisor *
                            self.meta_dataset.ignore_label),
                    threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                    nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
                    top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                    foreground_mask=None)
                torch.cuda.synchronize(self.device)
                post_time.update(time.time() - start_time)

                self.logger.info('[{}/{}]\t'
                            'Network Time: {net_time.val:.3f}s ({net_time.avg:.3f}s)\t'
                            'Post-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(
                             index, total, net_time=net_time, post_time=post_time))

                # save predictions
                #semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()

                # crop predictions
                #semantic_pred = semantic_pred[:raw_h, :raw_w]
                panoptic_pred = panoptic_pred[:raw_h, :raw_w]

                frame = creat_panoptic_annotation(panoptic_pred,
                                         label_divisor=self.meta_dataset.label_divisor,
                                         colormap=self.meta_dataset.create_label_colormap(),
                                         image=raw_image)
        except Exception:
            self.logger.exception("Exception during demo:")
            raise
        finally:
            self.logger.info("Demo finished.")
            return frame

