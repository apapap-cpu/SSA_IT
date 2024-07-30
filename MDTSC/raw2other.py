import rawpy
import imageio
import os

def convert_raw_to_images(input_path, output_dir):
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return
    try:
        with rawpy.imread(input_path) as raw:
            rgb_image = raw.postprocess()
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' does not exist.")
        return
    except rawpy.LibRawFileUnsupportedError as e:
        print(f"Error: Unsupported RAW file format or not a RAW file. {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    png_output_path = f"{output_dir}/new.png"
    jpg_output_path = f"{output_dir}/new.jpg"
    
    imageio.imwrite(png_output_path, rgb_image)
    print(f"Saved PNG image at {png_output_path}")
    
    imageio.imwrite(jpg_output_path, rgb_image, quality=95)  
    print(f"Saved JPG image at {jpg_output_path}")

input_raw_path = './Raw/1.raw'
output_directory = './source'

convert_raw_to_images(input_raw_path, output_directory)
