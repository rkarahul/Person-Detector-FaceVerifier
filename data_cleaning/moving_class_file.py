import os
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil

# Set the path to your dataset directory and destination directory
dataset_dir = r"data"
destination_dir = r"data_ok"

# Ensure destination directories exist
os.makedirs(destination_dir, exist_ok=True)

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

        # Check if class label '1' is in the classes
        if '1' in classes:
            shutil.move(os.path.join(dataset_dir, file_name), os.path.join(destination_dir, file_name))
            print(f"File {file_name} moved as it contains class label 1")
        else:
            # Count the occurrences of each class label
            for class_label in classes:
                class_counts[class_label] += 1

        temp += 1

# Extract the class labels and their corresponding frequencies
class_labels = list(class_counts.keys())
class_frequencies = list(class_counts.values())
print("class_labels:", class_labels)
print("class_frequencies:", class_frequencies)

# Plot the class distribution
plt.bar(class_labels, class_frequencies)
plt.xlabel("Class Labels")
plt.ylabel("Frequency")
plt.title("Class Distribution")
plt.xticks(rotation=90)
plt.show()

# Print the count of blank files
print("Number of blank files deleted:", blank_file_count)
