
import os
from PIL import Image
import csv
import numpy as np
import pandas as pd


cell_list = os.listdir('../Dataset/Cells/')
df = pd.DataFrame()

# counts = get_total_cell_counts('../Dataset/Dots/')
counts = []
cell_paths = []
dot_paths = []
density_paths = []

for cell_img in cell_list:
    if(cell_img.endswith(".png")):
        cell_paths.append(os.path.join('Cells/', cell_img))
        dot_paths.append(os.path.join(
            'Dots/', cell_img.replace('cell', 'dots')))
        density_paths.append(os.path.join(
            'Density/', cell_img.replace('cell', 'density').replace("png", "jpg")))

        count = 0
        img = Image.open(os.path.join('../Dataset/', dot_paths[-1]))
        width, height = img.size
        for i in range(width):
            for j in range(height):
                if (img.getpixel((i, j)) != (0, 0, 0)):
                    count += 1
        counts.append(count)

df["cell_paths"] = cell_paths
df["dot_paths"] = dot_paths
df["density_paths"] = density_paths
df["counts"] = counts

df.to_csv('../Dataset/all_data.csv')
