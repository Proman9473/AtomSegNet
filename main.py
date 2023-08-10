import os
from utils.utils import load_model
from PIL import Image
from skimage.morphology import opening, erosion, disk


def denoise(image, radius=5, method='Opening'):
    kernel = disk(radius)
    if method == 'Opening':
        denoised_image = opening(image, kernel)
    else:
        denoised_image = erosion(image, kernel)
    return denoised_image


def process_images(directory):
    circular_mask_path = 'model_weights/circularMask.pth'
    for subdir, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.png'):
                image_path = os.path.join(subdir, file_name)
                image = Image.open(image_path).convert('L')

                model_output = load_model(circular_mask_path, image, cuda=False, iter=1)

                denoised_image = denoise(model_output)

                denoised_image_pil = Image.fromarray((denoised_image * 255).astype('uint8'), mode='L')

                # Construct the output file name
                output_name = os.path.join(subdir, os.path.splitext(file_name)[0] + '_denoised.png')
                denoised_image_pil.save(output_name)
                print(f"Processed and saved: {output_name}")


# Define the main directory containing .tif files and subdirectories
main_directory = r'C:\Users\MahdiKhalili\Desktop\DM4samps'

# Process all .tif files in the main directory and its subdirectories
process_images(main_directory)
