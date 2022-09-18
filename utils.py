import pathlib
import numpy as np
from skimage.io import imread, imsave


def transform_and_store(
    main_path,
    data_folder_name, 
    new_folder_name, 
    transformation_function
    ):

    
    transformation_path_n = pathlib.Path(main_path + new_folder_name + '/n/')
    transformation_path_p = pathlib.Path(main_path + new_folder_name + '/p/')

    transformation_path_n.mkdir(parents=True, exist_ok=True)
    transformation_path_p.mkdir(parents=True, exist_ok=True)

    sub_folders = ['/n/', '/p/']
    for sub_folder in sub_folders:
        data_path = pathlib.Path(main_path + data_folder_name + sub_folder)
        for path in data_path.glob('*'):
            input_path = str(path)
            output_path = input_path.replace(data_folder_name, new_folder_name)
            input_image = imread(input_path)
            transformed_image = transformation_function(input_image)
            transformed_image*=255
            transformed_image = np.uint8(transformed_image)
            transformed_image
            imsave(output_path, transformed_image)