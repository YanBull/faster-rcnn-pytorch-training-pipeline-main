import os
from sklearn.model_selection import train_test_split
import shutil

# Read images and annotations
images = [os.path.join('images', x) for x in os.listdir('images') if x[-3:] == "jpg"]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]

images.sort()
annotations.sort()

print("Images: " + str(len(images)))
print("Annotations: " + str(len(annotations)))

missing = 0

for i in images:
    if ('annotations/' + i[7:-3] + "xml") not in annotations:
        print("Deleting: ", i)
        os.remove(i)

print("missing ", missing)
# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train/')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
