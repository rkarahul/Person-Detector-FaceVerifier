import os

# Define the folder containing the text files
folder_path = 'data_ok'

# Define the dictionary of replacements
replacements = {
    '0': '5',
}

def replace_in_file(file_path, replacements):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        file_contents = file.read()
    
    # Replace the specified words/phrases
    for old_word, new_word in replacements.items():
        file_contents = file_contents.replace(old_word, new_word)
    
    # Write the modified contents back to the file
    with open(file_path, 'w') as file:
        file.write(file_contents)

def process_folder(folder_path, replacements):
    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            replace_in_file(file_path, replacements)
            print(f"Processed {filename}")

# Run the script
process_folder(folder_path, replacements)
