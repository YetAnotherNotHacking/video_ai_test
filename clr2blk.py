from PIL import Image
import os
import concurrent.futures

def process_image(input_folder, output_folder, file_name):
    # Open the image
    image_path = os.path.join(input_folder, file_name)
    image = Image.open(image_path)
    
    # Convert the image to black and white
    bw_image = image.convert('L')
    
    # Save the black and white image to the output folder
    output_path = os.path.join(output_folder, file_name)
    bw_image.save(output_path)
    
    print(f"Converted {file_name} to black and white.")

def convert_to_bw_parallel(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get a list of all files in the input folder
    files = sorted(os.listdir(input_folder))
    
    # Initialize thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Submit tasks for each file
        futures = []
        for file_name in files:
            if file_name.lower().endswith('.png'):
                futures.append(executor.submit(process_image, input_folder, output_folder, file_name))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()

# Specify input and output folders
input_folder = 'output_frames'
output_folder = 'blk_white'

# Call the function to convert images in parallel
convert_to_bw_parallel(input_folder, output_folder)
