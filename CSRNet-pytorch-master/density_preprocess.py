import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import kde
from tqdm import tqdm
import json
from pathlib import Path


def get_total_cell_counts(path):
    counts = []
    for img_path in (glob.glob(os.path.join(path, '*dots.png'))):
        count = 0
        img = Image.open(img_path)

        width, height = img.size
        for i in range(width):
            for j in range(height):
                if (img.getpixel((i, j)) != (0, 0, 0)):
                    count += 1

        counts.append(count)

    return counts


def get_cell_coordinates(img_path):
    """
    Function that computes coordinates of points in a ###dots.jpg file and counts them

    input: str, image path

    output: X,Y: lists, coordinates
            count: int, number of points
    """

    X, Y = [], []

    img = Image.open(img_path)

    width, height = img.size
    for i in range(width):
        for j in range(height):
            if (img.getpixel((i, j)) != (0, 0, 0)):
                # get coordinates of cells in the image
                X.append(i), Y.append(j)

    count = len(X)  # number of cells in the image

    return X, Y, count


def make_density_map(X, Y, path):
    """
    Function that creates the density map of coordinates and saves it in a given path

    Input: X,Y: lists of coordinates
            path: str, path where you want to save image

    Output: none
    """

    count = len(X)

    # creating the density map using a gaussian kde
    nbins = 100
    k = kde.gaussian_kde([X, Y])
    xi, yi = np.mgrid[min(X):max(X):nbins*1j, min(Y):max(Y):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # plot aesthetics
    width, height = 256, 256
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    ax.set_ylim(255, 0)
    plt.savefig(path, dpi=height)


def main():
    data_path = '../Dataset/Dots'
    save_path = '../Dataset/Density/'
    # we are preprocessing all the images
    for img_path in tqdm(glob.glob(os.path.join(data_path, '*dots.png'))):

        img_num = img_path[11:14]  # get the image number

<<<<<<< HEAD
        img_name = img_num + 'density.png'
        im_save_path = os.path.join(save_path, img_name) #creating the path where we save our density map image
=======
        img_name = img_num + 'density.jpg'
        # creating the path where we save our density map image
        im_save_path = os.path.join(save_path, img_name)
>>>>>>> be23f96ea8a56090c2f0731a5eb587a25ca3efc3
        # print('image being saved at {}'.format(im_save_path))

        X, Y, count = get_cell_coordinates(img_path)

        make_density_map(X, Y, im_save_path)
<<<<<<< HEAD
        
 
 
=======


>>>>>>> be23f96ea8a56090c2f0731a5eb587a25ca3efc3
def json_writer():
    path = '../Dataset/Density/'
    paths = []
    for img_path in (glob.glob(os.path.join(path, '*density.png'))):
        paths.append(os.path.abspath(img_path))

    dots_path_store = '../'

<<<<<<< HEAD
    with open('../Density.json', 'w') as f:
=======
    with open('./Dots.json', 'w') as f:
>>>>>>> be23f96ea8a56090c2f0731a5eb587a25ca3efc3
        json.dump(paths, f)

    # return paths


if __name__ == '__main__':
    main()
