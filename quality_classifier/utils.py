from zipfile import ZipFile
from typing import List
import PIL.Image
from PIL import Image

def get_images_from_zip(zip_file_path: str) -> List[PIL.Image.Image]:
    """
    Extracts and loads PNG images from a zip file into a list of PIL Image objects.

    Args:
        zip_file_path (str): Path to the zip file containing PNG images

    Returns:
        List[PIL.Image.Image]: List of PIL Image objects loaded from the zip file, 
                              sorted by filename and converted to RGB format
    """
    with ZipFile(zip_file_path) as zf:
        zip_items = [item for item in sorted(zf.namelist()) if item.endswith(".png")]
        pil_imgs = [PIL.Image.open(zf.open(zip_item)).convert("RGB") for zip_item in zip_items]
    return pil_imgs

def concatenate_images(image_paths, image_output_size=None):
    """
    Concatenates multiple images into a 2x2 grid and optionally resizes the result.

    Args:
        image_paths (List[str]): List of paths to the images to concatenate. 
                                Expected to contain 4 images for a 2x2 grid.
        image_output_size (Tuple[int, int], optional): Target size (width, height) to resize 
                                                      the final concatenated image. 
                                                      If None, keeps original size.

    Returns:
        PIL.Image.Image: A single concatenated image containing the input images arranged 
                        in a 2x2 grid, optionally resized to image_output_size.
    """
    images = [Image.open(path).convert("RGB") for path in image_paths]
    image_size = images[0].size
    # concatenate the images to form a single image - since its 4 images, we will have a 2x2 grid
    concatenated_image = Image.new(
        'RGB', (image_size[0] * 2, image_size[1] * 2))
    for i, img in enumerate(images):
        x = i % 2
        y = i // 2
        concatenated_image.paste(img, (x * image_size[0], y * image_size[1]))
    # resize the concatenated image to the output size
    if image_output_size is not None:
        concatenated_image = concatenated_image.resize(image_output_size)
    return concatenated_image