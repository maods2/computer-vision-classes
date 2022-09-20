import pathlib
import numpy as np
from skimage.io import imread, imsave
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Callable


def transform_and_store(
    main_path: str,
    data_folder_name: str, 
    new_folder_name: str, 
    transformation_function: Callable
    ) -> None:
    """Utility functions that will load data from a given directoty,
    process this data based on a given transformation function and store 
    the output into a new folder also passed as argument of this function.

    Parameters
    ----------
    main_path : string
        Main directory where the data is stored.
    data_folder_name : string
        Data folder name in which the data that will be processed is stored.
    new_folder_name : string
        Name of the new folder that will be created in order to store the
        output data.
    transformation_function : Callable-like.
        Transformation functions that shoul be applyed to the input data 


    Examples
    --------

    >>> transform_and_store(
            main_path='./main_path/',
            data_folder_name='00-original', 
            new_folder_name='01-resized', 
            transformation_function=apply_resizing
        )
    """
    SUB_FOLDERS = ['/n/', '/p/']

    transformation_path_n = pathlib.Path(main_path + new_folder_name + '/n/')
    transformation_path_p = pathlib.Path(main_path + new_folder_name + '/p/')

    transformation_path_n.mkdir(parents=True, exist_ok=True)
    transformation_path_p.mkdir(parents=True, exist_ok=True)


    for sub_folder in SUB_FOLDERS:
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
    """Utility function that will plot images present into given directories.

    Parameters
    ----------
    main_path : string
        Main directory where the data is stored.
    data_folder_name : string
        Data folder name in which the data that will be plotted is stored.
    new_folder_name : string
        Data folder name in which the data that will be plotted is stored.
   

    Examples
    --------

    >>> plot_transformation_comparison(
            main_path='/main_path/',
            data_folder_name='00-original', 
            new_folder_name='01-resized',
        )
    """
    path_original = list(pathlib.Path(main_path + data_folder_name + '/n/' ).glob('*'))
    path_transformed = list(pathlib.Path(main_path + new_folder_name + '/n/' ).glob('*'))
    n_row, n_col = 2, 4
    f, axis = plt.subplots(n_row, n_col, figsize=(12, 7))

    for i in range(4):
        axis[0,i].imshow(mpimg.imread(path_original[i]),  cmap='gray')
        axis[0,i].set_title(f"{ data_folder_name.split('-')[1].capitalize() } {i+1}")
        axis[0,i].set_xticks([])
        axis[0,i].set_yticks([])

        axis[1,i].imshow(mpimg.imread(path_transformed[i]), cmap='gray')
        axis[1,i].set_title(f"{ new_folder_name.split('-')[1].capitalize() } {i+1}")
        axis[1,i].set_xticks([])
        axis[1,i].set_yticks([])

    plt.show()