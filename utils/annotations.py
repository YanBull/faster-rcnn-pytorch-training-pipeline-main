import numpy as np
import cv2
from datasets import extract_info_from_xml
from torch_utils.coco_eval import get_iou
import os
WITH_MARKING = ['Schraubenkopf_Seite_mit_Markierung',
            'Schraubenkopf_mit_Markierung',
            "Schraubenmutter_mit_Markierung",
            "Schraubenmutter_Seite_mit_Markierung"]

NO_MARKING = ["Schraubenkopf",
            "Schraubenkopf_Seite",
            "Schraubenmutter",
            "Schraubenmutter_Seite",
            "Schraubenkopf_Luftfeder"]

def inference_failed(
    annotations,
    outputs,
    orig_image,
    image,
    detection_threshold,
    image_filename,
    class_groups,
    failed_detections_info,
    args
):
    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    height, width, _ = orig_image.shape
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    draw_boxes = boxes.copy()

    no_marking_group = next((x for x in class_groups if x["GROUP_NAME"] == "NO_MARKING"), None)
    print(len(no_marking_group['CLASSES']))
    for annotation_box in annotations['bboxes']:
        found = False
        bb1 = {"x1" : int(annotation_box["xmin"]/image.shape[1]*width), 
               "x2" : int(annotation_box["xmax"]/image.shape[1]*width), 
               "y1": int(annotation_box["ymin"]/image.shape[0]*height), 
               "y2" : int(annotation_box["ymax"]/image.shape[0]*height)}

        for detection_box in draw_boxes:
            bb2 = {"x1" : int(detection_box[0]/image.shape[1]*width), 
                   "y1" : int(detection_box[1]/image.shape[0]*height), 
                   "x2" : int(detection_box[2]/image.shape[1]*width), 
                   "y2" : int(detection_box[3]/image.shape[0]*height)}
            
            if (get_iou(bb1, bb2) > 0 ):
                found = True

        if not found:
            extract_visual_conditions(failed_detections_info, annotation_box, image, orig_image, image_filename)
            print("Missed " + annotation_box['class'] + " in image: " + image_filename)
            cv2.rectangle(
                orig_image,
                (bb1["x1"], bb1["y1"]), (bb1["x2"], bb1["y2"]),
                color=(0,0,255), 
                thickness=2,
                lineType=cv2.LINE_AA
            )
            w, h = cv2.getTextSize(
                "Missed " + annotation_box['class'], 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw / 3, 
                thickness=tf
            )[0]  # text width, height
            w = int(w - (0.20 * w))
            outside = bb1["y1"] - h >= 3
            # no_marking_group = class_groups.find(a["GROUP_NAME"] == "NO_MARKING")
            color = (255,255,255,0)
            
            if no_marking_group is not None and annotation_box['class'] in no_marking_group['CLASSES']:
                color = (0,0,255,0)
            
            cv2.putText(orig_image,
                "Missed " + annotation_box['class'],
                (bb1['x1'], bb1['y1'] - 5 if outside else bb1['y1'] + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale =lw / 3.8,
                thickness = tf-2,
                color=color)  
            
            
    
    return orig_image

def inference_annotations(
    outputs, 
    detection_threshold, 
    classes,
    colors, 
    orig_image, 
    image, 
    args
):
    global WITH_MARKING, NO_MARKING
    height, width, _ = orig_image.shape
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    draw_boxes = boxes.copy()
    # Get all the predicited class names.
    pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    
    print("Detections")
    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1 = (int(box[0]/image.shape[1]*width), int(box[1]/image.shape[0]*height))
        p2 = (int(box[2]/image.shape[1]*width), int(box[3]/image.shape[0]*height))
        class_name = pred_classes[j]
        
        color = colors[0]
        
        if WITH_MARKING is not None and len(WITH_MARKING) > 0:
            color = (0,255,0) if class_name in WITH_MARKING else (0,0,255)
        else:
            color = colors[classes.index(class_name)]
        
        print(class_name, p1[0], p1[1], p2[0], p2[1])
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color, 
            thickness=lw - 1,
            lineType=cv2.LINE_AA
        )
        if not args['no_labels']:
            # For filled rectangle.
            final_label = class_name + ' ' + str(round(scores[j], 2))
            w, h = cv2.getTextSize(
                final_label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw / 3, 
                thickness=tf
            )[0]  # text width, height
            w = int(w - (0.20 * w))
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            
            cv2.putText(orig_image,
                final_label,
                (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale =lw / 3.8,
                thickness = tf-2,
                color=(255,255,255,0))            

    return orig_image

def save_patches_with_classes(annotations,
    outputs,
    orig_image,
    image,
    detection_threshold,
    image_filename,
    patch_size,
    args):
    
    height, width, _ = orig_image.shape
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    index = 0
    for annotation_box in annotations['bboxes']:
        index+=1
        found = False
        bb1 = {"x1" : int(annotation_box["xmin"]/image.shape[1]*width), 
               "x2" : int(annotation_box["xmax"]/image.shape[1]*width), 
               "y1": int(annotation_box["ymin"]/image.shape[0]*height), 
               "y2" : int(annotation_box["ymax"]/image.shape[0]*height)}

        for detection_box in boxes:
            bb2 = {"x1" : int(detection_box[0]/image.shape[1]*width), 
                   "y1" : int(detection_box[1]/image.shape[0]*height), 
                   "x2" : int(detection_box[2]/image.shape[1]*width), 
                   "y2" : int(detection_box[3]/image.shape[0]*height)}
            
            if (get_iou(bb1, bb2) > 0 ):
                found = True
        # resize the patch to square format using param "patch size"
        # save the patch somewhere with prefix "detected-" or "failed-"
        work_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        patch = work_image[annotation_box["ymin"]:annotation_box["ymax"], annotation_box["xmin"]:annotation_box["xmax"]]
        patch = cv2.resize(patch, patch_size)
        prefix = 'negative' if found == False else 'positive'
        
        if not os.path.exists(os.path.join('data','patches_vae')):
            os.makedirs((os.path.join('data','patches_vae')), exist_ok=True)
        
        cv2.imwrite(os.path.join("data","patches_vae", prefix+str(index)+image_filename+'.png'), patch)

def save_patches(
    outputs,
    orig_image,
    image,
    detection_threshold,
    image_filename,
    patch_size,
    args):
    
    height, width, _ = orig_image.shape
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    index = 0
    
    for detection_box in boxes:
        index+=1
        
        
        # patch = orig_image[bb2["y1"]:bb2["y2"], bb2["x1"]:bb2["x2"]]
        patch = orig_image[detection_box[1] : detection_box[3], detection_box[0] : detection_box[2]]
        patch = cv2.resize(patch, patch_size)
        if args["patches_path"] is not None:
            
            if not os.path.exists(args["patches_path"]):
                os.makedirs(args["patches_path"], exist_ok=True)
            image_name = image_filename + "_" + str(index) + ".jpg" 
            cv2.imwrite(os.path.join(args["patches_path"], image_name), patch)
            
        

def extract_visual_conditions(failed_detections_info: list(), annotation_box, image: cv2.Mat, orig_image: cv2.Mat, image_filename):
    height, width, _ = orig_image.shape
    entry = {
        
    }
    """
    TODO: Get the information about the missed patch like:
            # - median exposure
            # - distance to nearest edge of the image
            # - size of the patch
            # save that information to csv
    """
    gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    patch = gray_image[annotation_box['ymin']:annotation_box['ymax'], annotation_box['xmin']:annotation_box['xmax']]
    median = np.median(patch)
    
    distances_from_edge = [abs(height - annotation_box['ymax']), 
                           abs(0 - annotation_box['ymin']),
                           abs(width - annotation_box['xmax']),
                           abs(0 - annotation_box['xmin'])]
    
    patch_height, patch_width = patch.shape
    entry = {
        "image" :           image_filename,
        "class" :           annotation_box['class'],
        "position":         f"({annotation_box['xmin']},{annotation_box['ymin']}) ({annotation_box['xmax']},{annotation_box['ymax']})",
        "median_exposure":  median,
        "distance_to_edge": min(distances_from_edge),
        "patch_size":       patch_height * patch_width
    }
    
    failed_detections_info.append(entry)
    
def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):
        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )
        return img

def annotate_fps(orig_image, fps_text):
    draw_text(
        orig_image,
        f"FPS: {fps_text:0.1f}",
        pos=(20, 20),
        font_scale=1.0,
        text_color=(204, 85, 17),
        text_color_bg=(255, 255, 255),
        font_thickness=2,
    )
    return orig_image