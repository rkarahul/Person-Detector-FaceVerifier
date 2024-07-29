import os
import re

def clean_filenames(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            new_filename = re.sub(r'_png.*\.txt$', '.txt', filename)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')

# Replace with the actual path to your directory
directory = 'labels'  # Windows
# directory = '/Users/YourUsername/Desktop/files'  # macOS/Linux
clean_filenames(directory)
