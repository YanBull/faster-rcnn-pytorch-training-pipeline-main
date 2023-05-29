import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch_utils import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
        return coco_eval.stats

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))

def count_classification_results(classes: list, predictions: list, ground_truthes: list, IoU_threshold=0.5, score_threshold=0.5):
    
    if not len(predictions) == len(ground_truthes):
        raise Exception("Number of truthes and predictions is unequal, aborting")
    
    stat_entry = {
        "class_number": 0,
        "class_name": 's',
        "precision": 0,
        "recall": 0,
        "instances_originally": 0,
        "instances_detected" : 0,
        "detected_correctly": 0,
        "misclassified": 0,
        "classified_correctly": 0,
    }
        
    results = []
    
    class_counter = 0
    for i in range(0, len(classes), 1):
        
        if classes[i] != '__background__':
            class_counter+=1
            stat_e = stat_entry.copy()
            
            stat_e["class_name"] = classes[i]
            stat_e["class_number"] = class_counter
            
            results.append(stat_e)
        
    for i, gts in enumerate(ground_truthes):
            
        correct_localizations = get_correct_localizations(predictions[i], gts, IoU_threshold, score_threshold) 
        for class_entry in results:
            
            # TODO: number of detected_correctly is usually bigger than "instances_originally", investigate it
            # possibly it because the model detects multiple instances in place of the same one (all of them are considered correct in this case)
            for l in correct_localizations:
                if l["true_label"] == class_entry["class_number"]:
                    class_entry['detected_correctly']+=1
                    
                    if l['label_predicted'] == l["true_label"]:
                        class_entry["classified_correctly"]+=1
                    else:
                        class_entry["misclassified"]+=1
                
            for index, label in enumerate(gts["labels"]):
                if label == class_entry["class_number"]:
                    class_entry["instances_originally"] += 1
                
            for label in predictions[i]["labels"]:
                if label == class_entry["class_number"]:
                    class_entry["instances_detected"] += 1
            
                    
    for class_entry in results:
        class_entry["recall"] = class_entry["classified_correctly"] / class_entry["instances_originally"]
        class_entry["precision"] = class_entry["classified_correctly"] / class_entry["instances_detected"]
    
    return results
            
def get_correct_localizations(predictions: dict, truthes: dict, IoU_threshold: float, score_threshold: float):
    
    bb = {
        "x1": 1,
        "x2": 2,
        "y1": 1,
        "y2": 2
    }
    
    detection = {
        "bbox" : bb.copy(),
        "label_predicted": 1,
        "true_label": 1
    }
    predict_boxes_t = predictions["boxes"].numpy()
    truth_boxes_t = truthes["boxes"].numpy()
    scores_t = predictions['scores'].numpy()
    results = []
    for index, prediction in enumerate(predict_boxes_t):
        bb_pred = bb.copy()
        bb_pred["x1"], bb_pred["y1"], bb_pred["x2"], bb_pred["y2"] = prediction[0], prediction[1], prediction[2], prediction[3]
        current_score = scores_t[index]
        
        for truth_index, truth in enumerate(truth_boxes_t):
            bb_truth = bb.copy()
            bb_truth["x1"], bb_truth["y1"], bb_truth["x2"], bb_truth["y2"] = truth[0], truth[1], truth[2], truth[3]
            if get_iou(bb_pred, bb_truth) >= IoU_threshold and current_score > score_threshold:
                d = detection.copy()
                d["bbox"] = bb_pred
                d["label_predicted"] = predictions["labels"][index].item()
                d["true_label"] = truthes["labels"][truth_index].item()
                results.append(d)
                
    return results 

            
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
        
    