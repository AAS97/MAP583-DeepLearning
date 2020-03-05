import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import kde



def get_cell_coordinates(img_path):

    """
    Function that computes coordinates of points in a ###dots.jpg file and counts them

    input: str, image path

    output: X,Y: lists, coordinates
            count: int, number of points
    """

    X,Y = [], []
    
    img = Image.open(img_path)

    width, height = img.size
    for i in range (width):
        for j in range (height):
            if (img.getpixel((i,j)) != (0,0, 0)):
                X.append(i), Y.append(j) #get coordinates of cells in the image
                
    count = len(X)#number of cells in the image 

    return X,Y, count  



def make_density_map(X, Y, path):
    """
    Function that creates the density map of coordinates and saves it in a given path

    Input: X,Y: lists of coordinates
            path: str, path where you want to save image
    
    Output: none
    """

    count = len(X)

    # creating the density map using a gaussian kde
    nbins=100
    k = kde.gaussian_kde([X,Y])
    xi, yi = np.mgrid[min(X):max(X):nbins*1j, min(Y):max(Y):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # plot aesthetics

    f, axs = plt.subplots(figsize=(5,5))
    plt.title('Density map for {} cells'.format(count))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    axs.set_ylim(255, 0) #inverting y axis

    fig_name = plt.savefig(path)




def main():
    data_path = '../Dataset/'

    dots_data_paths = []

    for img_path in glob.glob(os.path.join(data_path, '*dots.png')): #we are preprocessing all the images
        
        img_num = img_path[11:14] #get the image number

        img_name = img_num + 'density.jpg'
        save_path = os.path.join(data_path, img_name) #creating the path where we save our density map image


        X,Y, count = get_cell_coordinates(img_path)

        make_density_map(X, Y, save_path)
        

if (__name__ == __main__):
    main()