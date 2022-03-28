from __future__ import annotations
import os

import torch

from mmcv.utils.config import Config
from mmocr.models import build_detector
from mmocr.apis.inference import model_inference
from mmocr.utils.model import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmocr.utils.fileio import list_from_file
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img


class KeyInformationExtraction:
    def __init__(self, root_path, **kwargs):
        self.config_path = os.path.join(root_path, 'config.py')
        self.model_path = os.path.join(root_path, 'best.pth')
        self.dict_path = os.path.join(root_path, 'dict.txt')
        self.cfg = Config.fromfile(self.config_path)
        self.model = build_detector(self.cfg.model)
        self.model = revert_sync_batchnorm(self.model)
        self.model.cfg = self.cfg
        if kwargs.get('device', None):
            self.device = kwargs['device']
        else:
            self.device = 'cpu'

        load_checkpoint(self.model, self.model_path, map_location=self.device)
        if self.model:
            self.dataset = KIEDataset(dict_file=self.dict_path)


    def generate_labels(self, result, boxes, class_list):
        idx_to_cls = {}
        if class_list is not None:
            for line in list_from_file(class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        max_value, max_idx = torch.max(result['nodes'].detach().cpu(), -1)
        node_pred_label = max_idx.numpy().tolist()
        node_pred_score = max_value.numpy().tolist()
        labels = []
        for i in range(len(boxes)):
            pred_label = str(node_pred_label[i])
            if pred_label in idx_to_cls:
                pred_label = idx_to_cls[pred_label]
            pred_score = node_pred_score[i]
            labels.append((pred_label, pred_score))
        return labels

    def predict(self, image, bb_annotations):
        '''
            bb_annotations: list of dict with bbox 04 point (y, ymax, x, xmax) and the text key and info.
                bb_annotations: [
                    {
                        bbox: [y, ymax, x, xmax],
                        text: ""
                    }
                ]

            bb_list : 
            list with 08 coordinates. The nineth element is score of bb.
            Textract: 
                - y, ymax, x, xmax
                - text
        '''
        annotations = []
        for bb in bb_annotations:
            box_res = {}
            box_res['box'] = [round(x) for x in bb['bbox']]
            box = []
            if len(bb['bbox']) == 4:
                min_x = bb['bbox'][2]
                min_y = bb['bbox'][0]
                max_x = bb['bbox'][3]
                max_y = bb['bbox'][1]

                box = [
                    min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                ]
            box_res['box'] = box
            box_res['text'] = bb['text']
            box_res['box_score'] = 1.0
            box_res['text_score'] = 1.0
            annotations.append(box_res)


        ann_info = self.dataset._parse_anno_info(annotations)
        ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                            ann_info['bboxes'])
        ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                            ann_info['bboxes'])
      
        kie_result, data = model_inference(
                        self.model,
                        image,
                        ann=ann_info,
                        return_data=True)

        gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
        labels = self.generate_labels(kie_result, gt_bboxes,
                                                  self.model.class_list)

        breakpoint()
        return None
