import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations, inference_failed, save_patches_with_classes, save_patches
from utils.general import set_infer_dir
from utils.transforms import infer_transforms, resize, infer_custom_transforms
from datasets import extract_info_from_xml

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '-c', '--config', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default=None,
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.3, type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show-image', dest='show_image', action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', dest='mpl_show', action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', '--img-size', 
        default=None,
        dest='img_size',
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    ),
    parser.add_argument(
        '-mf','--mark-failed',
        dest='mark_failed',
        action='store_true',
        help='if there is an annotation file for the respective image, marks missed objects with red'
    )
    parser.add_argument(
        '-spv', '--save-patches-vae',
        dest='save_patches_vae',
        action='store_true',
        help='save detected and not detected patches separately for future VAE training'
    )
    parser.add_argument(
        '-sp', '--save-patches',
        dest='save_patches',
        action='store_true',
        help='save all detections as patches'
    )
    parser.add_argument(
        '-pp','--patches-path',
        dest='patches_path',
        help='folder where the patches should be stored (if --save-patches is enabled)'
    )
    parser.add_argument(
        '-ct', '--custom-transforms',
        dest = 'custom_transforms',
        action='store_true',
        help = 'use custom transforms on the images before inference from the transforms.infer_custom_transforms'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['config'] is not None:
        with open(args['config']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']
    OUT_DIR = set_infer_dir()

    # Load the pretrained model
    if args['weights'] is None:
        # If the config file is still None, 
        # then load the default one for COCO.
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['config']['NC']
            CLASSES = checkpoint['config']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    if args['input'] == None:
        DIR_TEST = data_configs['image_path']
        test_images = collect_all_images(DIR_TEST)
    else:
        DIR_TEST = args['input']
        test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    
    # collect information about annotations that were not detected (as patch size, it's median, distance to the edge of image)
    failed_detections_info = list()

    if args['custom_transforms'] and not os.path.exists(os.path.join(OUT_DIR, 'transforms-preview')):
        os.makedirs(os.path.join(OUT_DIR, 'transforms-preview'))
    
    for i in range(len(test_images)):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        print("Image: ", image_name)
        orig_image = cv2.imread(test_images[i])
        frame_height, frame_width, _ = orig_image.shape
        if args['img_size'] != None:
            RESIZE_TO = args['img_size']
        else:
            RESIZE_TO = frame_width
        # orig_image = image.copy()
        image_resized = resize(orig_image, RESIZE_TO)
        image = image_resized.copy()
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if args['custom_transforms']:
            preview_image_path = os.path.join(OUT_DIR, 'transforms-preview',  image_name + "_" + str(i) + ".jpg")
            image = infer_custom_transforms(image, preview_image_path)
        else:
            image = infer_transforms(image)
            
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            # save each detected and not detected patch in square for future training of 
            if args['save_patches_vae']:
                xml_file = test_images[i].replace('images', 'annotations').replace('jpg', 'xml')
                annotations = extract_info_from_xml(xml_file, CLASSES)
                save_patches_with_classes(annotations, outputs, orig_image, image, detection_threshold, image_name, (64,64), args)
            
            if args["save_patches"]:
                save_patches(outputs, orig_image, image, detection_threshold, image_name, (64,64), args)
                
            orig_image = inference_annotations(
                outputs, 
                detection_threshold, 
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )
            
            if args['mark_failed']:
                xml_file = test_images[i].replace('images', 'annotations').replace('jpg', 'xml')
                
                if(os.path.exists(xml_file)):
                    annotations = extract_info_from_xml(xml_file, CLASSES)
                    
                    orig_image = inference_failed(
                        annotations, outputs, orig_image, image_resized, 
                        detection_threshold, image_name, data_configs["CLASS_GROUPS"], failed_detections_info, args
                    )
                

            if args['show_image']:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if args['mpl_show']:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()
        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)
    
    if args['mark_failed']:
        print_failure_thresholds(failed_detections_info, out_dir=OUT_DIR)
    
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.

    
def print_failure_thresholds(failed_detections_info, out_dir):
    max_exposure = 0
    min_exposure = 255
    min_distance = 1000
    min_size = 2000
    
    exposure_list = list()
    size_list = list()
    distance_from_edge_list = list()
    for entry in failed_detections_info:
        
        # exposure_list.append(entry['median_exposure'])
        size_list.append(entry["patch_size"])
        distance_from_edge_list.append(entry['distance_to_edge'])
        
        if entry['median_exposure'] > max_exposure:
            max_exposure = entry['median_exposure']

        if entry['median_exposure'] < min_exposure:
            min_exposure = entry['median_exposure']

        if entry['distance_to_edge'] < min_distance:
            min_distance = entry['distance_to_edge']
        
        if entry['patch_size'] < min_size:
            min_size = entry['patch_size']
    
    median_size_of_failed_patches = np.median(np.sort(size_list))
    median_distance_of_failed_patches = np.median(np.sort(distance_from_edge_list))
    distance_from_edge_list.clear()
    
    df = pd.DataFrame(failed_detections_info, 
                      columns=["image", "class", "position", "median_exposure", "distance_to_edge", "patch_size"])
    
    df.to_csv(os.path.join(out_dir, "failed_detections.csv"), index=False)
    
    # Add only big enough and distanced-from-edge failures to the list of median exposures of failed detections  
    for entry in failed_detections_info:
        if entry['patch_size']/median_size_of_failed_patches > 1.5:
            distance_from_edge_list.append(entry['distance_to_edge'])
            
        if entry['distance_to_edge']/median_distance_of_failed_patches > 1.5 and entry['patch_size']/median_size_of_failed_patches > 1.5:
            exposure_list.append(entry['median_exposure'])
        
            
    plot_and_save("Median exposure of not detected screws, in grayscale value (0-255)", exposure_list)
    plot_and_save("Size of not detected screws, in pixels",size_list)
    plot_and_save("Distance to the closest edge of the image of not detected screws",distance_from_edge_list)
    
    print(f'median_size_of_failed_patches: {median_size_of_failed_patches}\n median_distance_from_edge of failed patches: {median_distance_of_failed_patches}')
    print(f'max exposure: {max_exposure} \n min exposure: {min_exposure} \n min_distance: {min_distance} \n min patch size: {min_size}')
         
def plot_and_save(name, array):
    plt.clf()
    plt.cla()
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.hist(array)
    plt.title(name)
    plt.savefig(name+'.png')
    plt.clf()
    plt.cla()
    
if __name__ == '__main__':
    args = parse_opt()
    main(args)