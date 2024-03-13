import os
import concurrent.futures
from PIL import Image

def resize_image(input_file, output_folder, new_width=100, new_height=200):
    try:
        # Open the image
        with Image.open(input_file) as img:
            # Resize the image without cropping
            resized_img = img.resize((new_width, new_height))
            # Extract the filename without extension
            filename = os.path.splitext(os.path.basename(input_file))[0]
            # Save the resized image
            resized_img.save(os.path.join(output_folder, f"{filename}_resized.png"))
            print(f"Resized {input_file} successfully")
    except Exception as e:
        print(f"Error resizing {input_file}: {e}")

def main():
    input_folder = "blk_white"
    output_folder = "final_output"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of input image files
    input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.png')]

    # Concurrently resize the images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit resize tasks
        futures = [executor.submit(resize_image, file, output_folder) for file in input_files]

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
