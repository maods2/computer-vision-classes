import pathlib
import numpy as np
from skimage.io import imread, imsave
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def transform_and_store(
    main_path: str,
    data_folder_name: str, 
    new_folder_name: str, 
    transformation_function
    ) -> None:
    """Function responsible for mapping
    
    """

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
            imsave(output_path, transformed_image)


def plot_transformation_comparison(
        main_path: str,
        data_folder_name: str, 
        new_folder_name: str, 
    ) -> None:

    path_original = list(pathlib.Path(main_path + data_folder_name + '/n/' ).glob('*'))
    path_transformed = list(pathlib.Path(main_path + new_folder_name + '/n/' ).glob('*'))
    n_row, n_col = 2, 4
    f, axis = plt.subplots(n_row, n_col, figsize=(12, 7))

    for i in range(4):
        axis[0,i].imshow(mpimg.imread(path_original[i]))
        axis[0,i].set_title(f"{ data_folder_name.split('-')[1].capitalize() } {i+1}")
        axis[0,i].set_xticks([])
        axis[0,i].set_yticks([])

        axis[1,i].imshow(mpimg.imread(path_transformed[i]))
        axis[1,i].set_title(f"{ new_folder_name.split('-')[1].capitalize() } {i+1}")
        axis[1,i].set_xticks([])
        axis[1,i].set_yticks([])

    plt.show()