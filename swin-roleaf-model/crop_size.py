import argparse
from PIL import Image
import numpy as np
import os
import glob

def split_image(image_path, tile_size):
    """Split the image into square sub-images of the specified size, and fill the missing parts with black"""
    image = Image.open(image_path)
    width, height = image.size

    image = image.convert('RGBA')

    rows = np.ceil(height / tile_size).astype(int)
    cols = np.ceil(width / tile_size).astype(int)

    new_width = cols * tile_size
    new_height = rows * tile_size
    new_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 255))
    new_image.paste(image, (0, 0))

    for row in range(rows):
        for col in range(cols):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size

            tile = new_image.crop((left, upper, right, lower))
            tile_path = f'{image_path}_tile_{row}_{col}.png'
            tile.save(tile_path)
            print(f'Saved {tile_path}')

def process_folder(folder_path, tile_size):
    for image_path in glob.glob(f"{folder_path}/*"):
        if image_path.lower().endswith(('.jpg', '.png', '.tiff')):
            print(f"Processing {image_path}...")
            split_image(image_path, tile_size)

def main():
    parser = argparse.ArgumentParser(description="Split images in a folder into tiles of specified size")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the images")
    parser.add_argument("tile_size", type=int, help="Size of the square tiles (must be a multiple of 32)")
    args = parser.parse_args()

    if args.tile_size % 32 != 0:
        print("Error: crop size must be a multiple of 32")
        return

    process_folder(args.folder_path, args.tile_size)

if __name__ == "__main__":
    main()