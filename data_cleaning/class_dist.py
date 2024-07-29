##import subprocess
##
### Package name to install
##package_name = "matplotlib"
##
### Path to the Python 3.8 interpreter
##python_path = r"C:\Users\Admin\AppData\Local\Programs\Python\Python38\python.exe"
##
### Use subprocess to run pip install command with specific Python interpreter
##subprocess.check_call([python_path, "-m", "pip", "install", package_name])
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil

# Set the path to your dataset directory
dataset_dir = r"data"
destination_dir = r"data_0k"

# Ensure destination directories exist
#os.makedirs(destination_dir, exist_ok=True)
# Create a defaultdict to store class frequencies
class_counts = defaultdict(int)
blank_file_count = 0
temp = 0

# Iterate over the files in the dataset directory
for file_name in os.listdir(dataset_dir):
    if file_name.endswith(".txt"):
        # Read the text file containing bounding box annotations
        with open(os.path.join(dataset_dir, file_name), 'r') as f:
            lines = f.readlines()

        # Check if the file is blank (contains no lines)
        if len(lines) == 0:
            # Delete the blank file
            os.remove(os.path.join(dataset_dir, file_name))
            blank_file_count += 1
            continue

        # Extract the class labels from the annotations
        classes = [line.split()[0] for line in lines]

        # Check if there is only one unique class label and it is '3'
        if len(set(classes)) == 1 and '0' in classes:
            #os.remove(os.path.join(dataset_dir, file_name))
            shutil.move(os.path.join(dataset_dir, file_name), os.path.join(destination_dir, file_name))
            print(f"File {file_name} moved as it contains only class_label 0")
        else:
            # Count the occurrences of each class label
            for class_label in classes:
                if class_label == '142':
                    print(file_name)
                class_counts[class_label] += 1

        temp += 1
#[1169, 5740, 544, 1293, 311]
# Extract the class labels and their corresponding frequencies
class_labels = list(class_counts.keys())
class_frequencies = list(class_counts.values())
print("class_labels : ", class_labels)
print("class_frequencies : ", class_frequencies)
# print(['Crack', 'Distortion', 'HDA', 'Rubber_Defect', 'Voids'])
# Plot the class distribution
plt.bar(class_labels, class_frequencies)
plt.xlabel("Class Labels")
plt.ylabel("Frequency")
plt.title("Class Distribution")
plt.xticks(rotation=90)
plt.show()

# Print the count of blank files
print("Number of blank files deleted:", blank_file_count)

##"""
###[53, 140, 66, 75]
###dataset_dir = "C:/Users/Administrator/Desktop/yolov4/darknet/data/obj"# "F:/iocl_22-9-22/captureImage/captureImage/backed/app_code/objdata_updated/data/obj"#
##
##import os
##import matplotlib.pyplot as plt
##from collections import defaultdict
##
### Set the path to your dataset directory
##dataset_dir = r"F:\1470-XRAY10\barrels_defect_segmentation\barrel_segmentation\train\images"#r"C:\Users\Administrator\Desktop\yolov4\darknet\data\obj"#"C:\\Users\\Administrator\\Desktop\\New_obj1"#
##
### Create a defaultdict to store class frequencies
##class_counts = defaultdict(int)
##blank_file_count = 0
##temp=0
### Iterate over the files in the dataset directory
##for file_name in os.listdir(dataset_dir):
##    if file_name.endswith(".txt"):
##        # Read the text file containing bounding box annotations
##        with open(os.path.join(dataset_dir, file_name), 'r') as f:
##            lines = f.readlines()
##
##        # Check if the file is blank (contains no lines)
##        if len(lines) == 0:
##            # Delete the blank file
##            os.remove(os.path.join(dataset_dir, file_name))
##            blank_file_count += 1
##            continue
##        
##        print(temp)
##        # Extract the class labels from the annotations
##        classes = [line.split()[0] for line in lines]
##
##        # Count the occurrences of each class label
##        for class_label in classes:
##            if class_label=='2':
##                print(file_name)
##            class_counts[class_label] += 1
##        temp+=1
##
### Extract the class labels and their corresponding frequencies
##class_labels = list(class_counts.keys())
##class_frequencies = list(class_counts.values())
##print("class_labels : ",class_labels)
##print("class_frequencies : ",class_frequencies)
### Plot the class distribution
##plt.bar(class_labels, class_frequencies)
##plt.xlabel("Class Labels")
##plt.ylabel("Frequency")
##plt.title("Class Distribution")
##plt.xticks(rotation=90)
##plt.show()
##
### Print the count of blank files
##print("Number of blank files deleted:", blank_file_count)"""
