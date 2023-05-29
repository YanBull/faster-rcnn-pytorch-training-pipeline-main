import os 
import xml.etree.ElementTree as ET


labels_folder = 'home/yan.bulatov/faster-rcnn-pytorch-training-pipeline/data/annotations/' # Absolute path to the folder with labels and images
labels_set = set()

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    

    # Parse the XML Tree
    for elem in root:
        
        # Get details of the bounding box 
        if elem.tag == "object":
            for subelem in elem:
                
                if subelem.tag == "name" and subelem.text not in labels_set:
                    labels_set.add(subelem.text)


def list_annotations(dir):
    annotations = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:
                if file[-3:] == "xml":
                    annotations.append(os.path.join(subdir,file))                                                                         
    return annotations   

# annotations = list_annotations(labels_folder)
annotations = [os.path.join('annotations/train', x) for x in os.listdir('annotations/train') if x[-3:] == "xml"]

print("Found ", len(annotations), " annotations")

for ann in annotations:
    extract_info_from_xml(xml_file=ann)
    
for label in labels_set:
    print(str(label) + ",")
