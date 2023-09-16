import pathlib
import numpy as np
from skimage.io import imread, imsave
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Callable
import gc
from tqdm import tqdm


def transform_and_store(
    main_path: str,
    data_folder_name: str, 
    new_folder_name: str, 
    transformation_function: Callable,
    output_type: str = 'image'
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
            if output_type == 'image':
                transformed_image*=255
                transformed_image = np.uint8(transformed_image)
                imsave(output_path, transformed_image)
            elif output_type == 'nparray':
                np.save(output_path.replace(".jpg", ""), transformed_image)



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

def plot_transformation_comparison2(
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
    path_transformed = list(pathlib.Path(main_path + new_folder_name + '/p/' ).glob('*'))
    n_row, n_col = 2, 4
    f, axis = plt.subplots(n_row, n_col, figsize=(10, 5))

    for i in range(4):
        axis[0,i].imshow(mpimg.imread(path_original[i]),  cmap='gray')
        # axis[0,i].set_title(f"{ data_folder_name.split('-')[1].capitalize() } {i+1}")
        axis[0,i].set_xticks([])
        axis[0,i].set_yticks([])

        axis[1,i].imshow(mpimg.imread(path_transformed[i]), cmap='gray')
        # axis[1,i].set_title(f"{ new_folder_name.split('-')[1].capitalize() } {i+1}")
        axis[1,i].set_xticks([])
        axis[1,i].set_yticks([])

    plt.show()


def plot_features(main_path, folders, titles, result_is_image=True):
    n_row, n_col = 3, 7
    f, axis = plt.subplots(n_row, n_col, figsize=(18, 7))

    for i, folder in enumerate(folders):
        path = list(pathlib.Path(main_path + folder + '/n/').glob('*'))

        if folder == '01-resized':
            img1 = imread(str(path[20]))
            img2 = imread(str(path[30]))
            img3 = imread(str(path[2]))
        else:
            img1 = np.load(str(path[20]))
            img2 = np.load(str(path[30]))
            img3 = np.load(str(path[2]))


        if result_is_image or i == 0:
            axis[0,i].imshow(img1,  cmap='gray')
            axis[0,i].set_title(titles[i])
            axis[0,i].set_xticks([])
            axis[0,i].set_yticks([])

            axis[1,i].imshow(img2,  cmap='gray')
            axis[1,i].set_title(titles[i])
            axis[1,i].set_xticks([])
            axis[1,i].set_yticks([])

            axis[2,i].imshow(img3, cmap='gray')
            axis[2,i].set_title(titles[i])
            axis[2,i].set_xticks([])
            axis[2,i].set_yticks([])
        else:            
            axis[0,i].plot(img1)
            axis[0,i].set_title(titles[i])
            axis[0,i].set_xticks([])
            axis[0,i].set_yticks([])

            axis[1,i].plot(img2)
            axis[1,i].set_title(titles[i])
            axis[1,i].set_xticks([])
            axis[1,i].set_yticks([])

            axis[2,i].plot(img3)
            axis[2,i].set_title(titles[i])
            axis[2,i].set_xticks([])
            axis[2,i].set_yticks([])

    plt.show()



def concat_features(subfolder, image_code):
    main_path='../dados/Hypercellularity/'
    data_path = pathlib.Path(main_path)
    features = []
    for steps in tqdm(['10','11','12']):
        for path in tqdm(data_path.glob(f'**/{steps}.*')):
            image_feature = np.load(str(path) + f'/{subfolder}/{image_code}.npy')
            if len(image_feature.shape) == 3:
                image_feature = image_feature[:,:,0].flatten()
                features.append(image_feature)
            elif len(image_feature.shape) == 2:
                image_feature = image_feature.flatten()
                features.append(image_feature)
            else:
                features.append(image_feature.copy())

            del image_feature
            gc.collect()

    return np.concatenate(features)

def normalize(xs):
    v_min = xs.min(axis=(0), keepdims=True)
    v_max = xs.max(axis=(0), keepdims=True)
    xs = (xs - v_min)/(v_max - v_min)
    return xs


def agregate_features():

    original_path = '../dados/Hypercellularity/00-original/'

    features = []
    labels = []

    SUB_FOLDERS = ['n', 'p']
    for sub_folder in tqdm(SUB_FOLDERS):
        data_path = pathlib.Path(original_path + sub_folder)
        for path in tqdm(data_path.glob('*')):
            image_code = str(path)[:-4].replace(str(data_path)+'/', "")
            image_representation = concat_features(sub_folder, image_code)
            image_representation = normalize(image_representation)
            image_class = 1 if sub_folder == 'n' else -1
            features.append(image_representation)
            labels.append(image_class)
            break
    x = np.vstack(features) 
    y = np.vstack(labels) 

    return x, y
      