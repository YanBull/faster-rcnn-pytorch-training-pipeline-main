import logging
import os
import pandas as pd
import wandb
import cv2
import numpy as np

from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

# Initialize Weights and Biases.
def wandb_init(name):
    os.environ['WANDB_SILENT']="true"
    wandb.init(mode="disabled")
    # wandb.init(name=name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def set_log(log_dir):
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # datefmt='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

def log(content, *args):
    for arg in args:
        content += str(arg)
    logger.info(content)

def coco_log(log_dir, stats):
    log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]
    log_dict = {}
    # for i, key in enumerate(log_dict_keys):
    #     log_dict[key] = stats[i]

    with open(f"{log_dir}/train.log", 'a+') as f:
        f.writelines('\n')
        for i, key in enumerate(log_dict_keys):
            out_str = f"{key} = {stats[i]}"
            logger.debug(out_str) # DEBUG model so as not to print on console.
        logger.debug('\n'*2) # DEBUG model so as not to print on console.
    # f.close()

def set_summary_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def tensorboard_loss_log(name, loss_np_arr, writer):
    """
    To plot graphs for TensorBoard log. The save directory for this
    is the same as the training result save directory.
    """
    for i in range(len(loss_np_arr)):
        writer.add_scalar(name, loss_np_arr[i], i)

def tensorboard_map_log(name, val_map_05, val_map, writer):
    for i in range(len(val_map)):
        writer.add_scalars(
            name,
            {
                'mAP@0.5': val_map_05[i], 
                'mAP@0.5_0.95': val_map[i]
            },
            i
        )

def create_log_map_csv(log_dir):
    cols = [
        'epoch', 
        'map', 
        'map_05',
        'train loss',
        'train cls loss',
        'train box reg loss',
        'train obj loss',
        'train rpn loss'
    ]
    results_csv = pd.DataFrame(columns=cols)
    results_csv.to_csv(os.path.join(log_dir, 'results.csv'), index=False)
    
def create_log_classification_csv(dir, IoU:float, score:float):
    
    if not os.path.exists(dir):
        dir = os.makedirs(dir)
    
    cols = [
        "class_number",
        'class',
        'precision',
        'recall',
        'instances_originally',
        'instances_detected',
        "detected_correctly",
        'misclassified',
        'classified_correctly'
    ]
    
    day_str = '{:02d}'.format(datetime.now().timetuple().tm_mday)+'_'+ '{:02d}'.format(datetime.now().timetuple().tm_mon) 
       
    results_csv = pd.DataFrame(columns=cols)
    print("DIR: " + dir)
    file = os.path.join(dir, 'class_res_' + day_str + '_iou-'+ str(IoU) + '_score-' + str(score) + '.csv') 
    results_csv.to_csv(file, index=False)
    
    return file
    
    
def log_classification(data: list, file, class_groups=list()):
        
    group_stats = []
    for index, group in enumerate(class_groups):
        group_stats.append({
            'class_number': 101+index,
            'class': group["GROUP_NAME"],
            'precision': 0,
            'recall': 0,
            'instances_originally': 0,
            'instances_detected': 0,
            "detected_correctly": 0,
            'misclassified': 0,
            'classified_correctly': 0
        })
        
    for entry in data:
        df = pd.DataFrame({
            'class_number': int(entry["class_number"]),
            'class': entry["class_name"],
            'precision': float(round(entry["precision"], 2)),
            'recall': float(round(entry["recall"], 2)),
            'instances_originally': int(entry["instances_originally"]),
            'instances_detected': int(entry["instances_detected"]),
            "detected_correctly": int(entry["detected_correctly"]),
            'misclassified': int(entry["misclassified"]),
            'classified_correctly': int(entry["classified_correctly"])
        }, index=[0])
        
        df.to_csv(
            file, 
            mode='a', 
            index=False, 
            header=False
        )
        
        for group in class_groups:
            if entry["class_name"] in group["CLASSES"]:
                group_stat = next((x for x in group_stats if x['class'] == group["GROUP_NAME"]), None)
                if group_stat is not None:
                    group_stat['instances_originally'] += int(entry["instances_originally"])
                    group_stat['instances_detected'] += int(entry["instances_detected"])
                    group_stat["detected_correctly"] += int(entry["detected_correctly"])
                    group_stat['misclassified'] += int(entry["misclassified"])
                    group_stat['classified_correctly'] += int(entry["classified_correctly"])
                
    for entry in group_stats:
        df = pd.DataFrame({
            'class_number': int(entry["class_number"]),
            'class': entry["class"],
            'precision': float(round(entry["classified_correctly"]/entry["instances_detected"], 2)),
            'recall': float(round(entry["classified_correctly"]/entry["instances_originally"], 2)),
            'instances_originally': int(entry["instances_originally"]),
            'instances_detected': int(entry["instances_detected"]),
            "detected_correctly": int(entry["detected_correctly"]),
            'misclassified': int(entry["misclassified"]),
            'classified_correctly': int(entry["classified_correctly"])
        }, index=[0])
        
        df.to_csv(
            file, 
            mode='a', 
            index=False, 
            header=False
        )

def csv_log(
    log_dir, 
    stats, 
    epoch,
    train_loss_list,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list
):
    if epoch+1 == 1:
        create_log_map_csv(log_dir) 
    
    df = pd.DataFrame(
        {
            'epoch': int(epoch+1),
            'map_05': [float(stats[0])],
            'map': [float(stats[1])],
            'train loss': train_loss_list[-1],
            'train cls loss': loss_cls_list[-1],
            'train box reg loss': loss_box_reg_list[-1],
            'train obj loss': loss_objectness_list[-1],
            'train rpn loss': loss_rpn_list[-1]
        }
    )
    df.to_csv(
        os.path.join(log_dir, 'results.csv'), 
        mode='a', 
        index=False, 
        header=False
    )

def overlay_on_canvas(bg, image):
    bg_copy = bg.copy()
    h, w = bg.shape[:2]
    h1, w1 = image.shape[:2]
    # Center of canvas (background).
    cx, cy = (h - h1) // 2, (w - w1) // 2
    bg_copy[cy:cy + h1, cx:cx + w1] = image
    return bg_copy * 255.

def wandb_log(
    epoch_loss, 
    loss_list_batch,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list,
    val_map_05, 
    val_map,
    val_pred_image,
    image_size
):
    """
    :param epoch_loss: Single loss value for the current epoch.
    :param batch_loss_list: List containing loss values for the current 
        epoch's loss value for each batch.
    :param val_map_05: Current epochs validation mAP@0.5 IoU.
    :param val_map: Current epochs validation mAP@0.5:0.95 IoU. 
    """
    # WandB logging.
    for i in range(len(loss_list_batch)):
        wandb.log(
            {'train_loss_iter': loss_list_batch[i],},
        )
    # for i in range(len(loss_cls_list)):
    wandb.log(
        {
            'train_loss_cls': loss_cls_list[-1],
            'train_loss_box_reg': loss_box_reg_list[-1],
            'train_loss_obj': loss_objectness_list[-1],
            'train_loss_rpn': loss_rpn_list[-1]
        }
    )
    wandb.log(
        {
            'train_loss_epoch': epoch_loss
        },
    )
    wandb.log(
        {'val_map_05_95': val_map}
    )
    wandb.log(
        {'val_map_05': val_map_05}
    )

    bg = np.full((image_size * 2, image_size * 2, 3), 114, dtype=np.float32)

    if len(val_pred_image) == 1:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) == 2:
        log_image = cv2.hconcat(
            [
                overlay_on_canvas(bg, val_pred_image[0]), 
                overlay_on_canvas(bg, val_pred_image[1])
            ]
        )
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) > 2 and len(val_pred_image) <= 8:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        for i in range(len(val_pred_image)-1):
            log_image = cv2.hconcat([
                log_image, 
                overlay_on_canvas(bg, val_pred_image[i+1])
            ])
        wandb.log({'predictions': [wandb.Image(log_image)]})
    
    if len(val_pred_image) > 8:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        for i in range(len(val_pred_image)-1):
            if i == 7:
                break
            log_image = cv2.hconcat([
                log_image, 
                overlay_on_canvas(bg, val_pred_image[i-1])
            ])
        wandb.log({'predictions': [wandb.Image(log_image)]})

def wandb_save_model(model_dir):
    """
    Uploads the models to Weights&Biases.

    :param model_dir: Local disk path where models are saved.
    """
    wandb.save(os.path.join(model_dir, 'best_model.pth'))