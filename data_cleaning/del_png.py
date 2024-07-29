# import os


# folder_path =r"G:\1470-XRAY10\all_annotated\train\images"

# # Get a list of all files in the folder
# files = os.listdir(folder_path)

# # Iterate over each file in the folder
# for file in files:
#    file_name, file_ext = os.path.splitext(file)
   
#    if file_ext.lower() == ".png":
#        # Check if the corresponding TXT file exists
#        txt_file = file_name + ".txt"
#        if txt_file not in files:
#            # Delete the PNG file
#            os.remove(os.path.join(folder_path, file))

# print("done")

##
import os
import shutil

folder_path = r"G:\1470-XRAY10\all_annotated\train\images"
destination_folder = r"G:\1470-XRAY10\all_annotated\train\all_voids"

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Iterate over each file in the folder
for file in files:
    file_name, file_ext = os.path.splitext(file)
    
    if file_ext.lower() == ".txt":
        # Check if the corresponding PNG file exists
        png_file = file_name + ".png"
        if png_file not in files:
            # Move the TXT file to the destination folder
            print("moving :- ", file)
            shutil.move(os.path.join(folder_path, file), os.path.join(destination_folder, file))

print("done")

#deleting txt file
#import os
#
#folder_path = r"J:\train"
#
## Get a list of all files in the folder
#files = os.listdir(folder_path)
#
## Iterate over each file in the folder
#for file in files:
#    file_name, file_ext = os.path.splitext(file)
#    
#    if file_ext.lower() == ".txt":
#        # Check if the corresponding BMP file exists
#        bmp_file = file_name + ".bmp"
#        if bmp_file not in files:
#            # Delete the TXT file
#            print("deleting :- ",bmp_file)
#            os.remove(os.path.join(folder_path, file))
#
#print("done")

#renaming file of the folder
##import os
##
##def add_yellow_to_filenames(folder_path):
##    try:
##        # List all files in the folder
##        files = os.listdir(folder_path)
##
##        # Iterate through each file
##        for file in files:
##            # Check if the file is an image (you can customize the list of image extensions)
##            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.txt', '.bmp')):
##                # Construct the old and new file paths
##                old_file_path = os.path.join(folder_path, file)
##                new_file_path = os.path.join(folder_path, "n2_" + file)
##
##                # Rename the file
##                os.rename(old_file_path, new_file_path)
##                print(f"File '{file}' successfully renamed to 'new_{file}'.")
##
##        print("All image files renamed.")
##    except FileNotFoundError:
##        print(f"Error: Folder '{folder_path}' not found.")
##
### Example usage
##folder_path = r"G:\YKK\ykk data\data\crops"
##add_yellow_to_filenames(folder_path)

##
##import os
##import shutil
##
### Define the source and destination folder paths
##source_folder = r"G:\button_detecton\datasets\images\train"
##destination_folder = r"G:\button_detecton\datasets\images\temp"
##
### Get a list of all files in the source folder
##files = os.listdir(source_folder)
##
### Iterate over each file in the source folder
##for file in files:
##    file_name, file_ext = os.path.splitext(file)
##    
##    if file_ext.lower() == ".bmp":
##        # Check if the corresponding TXT file exists
##        txt_file = file_name + ".txt"
##        if txt_file not in files:
##            # Move the BMP file to the destination folder
##            shutil.move(os.path.join(source_folder, file), os.path.join(destination_folder, file))
##
##print("done")


##import os
##import shutil
##
### Define the source and destination folder paths
##source_folder = r"C:\Users\Admin\Downloads\Documents\Annotation_ADR\train\images"
##destination_folder = r"C:\Users\Admin\Downloads\Documents\Annotation_ADR\train\void"
##
### Create the destination folder if it doesn't exist
##if not os.path.exists(destination_folder):
##    os.makedirs(destination_folder)
##
### Get a list of all files in the source folder
##files = os.listdir(source_folder)
##
### Iterate over each file in the source folder
##for file in files:
##    file_name, file_ext = os.path.splitext(file)
##    
##    if file_ext.lower() == ".jpg":
##        # Check if the corresponding TXT file exists
##        txt_file = file_name + ".txt"
##        if txt_file not in files:
##            # Move the PNG file to the destination folder
##            shutil.move(os.path.join(source_folder, file), os.path.join(destination_folder, file))
##
##print("done")

