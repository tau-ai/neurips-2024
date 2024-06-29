import torch
import numpy as np
import os
import gzip
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from rasterio.plot import show
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta

class CHIRP_parser:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = None
        self.start_date = datetime(2009, 1, 1)
        self.end_date = datetime(2021, 12, 31)
        self.files = self.generate_file_paths()
        self.read_data()
        self.save_tensor(self.data, "dataset")

    def generate_file_paths(self):
        file_paths = []
        current_date = self.start_date

        while current_date <= self.end_date:
            if current_date < datetime(2021, 12, 1):
                file_name = f"chirps-v2.0.{current_date.year}.{current_date.month:02d}.{current_date.day:02d}.tif.gz"
            else:
                file_name = f"chirps-v2.0.{current_date.year}.{current_date.month:02d}.{current_date.day:02d}.tif"
            file_path = os.path.abspath(os.path.join(self.root_dir, file_name))
            print("Generated file path:", file_path)
            file_paths.append(file_path)
            current_date += timedelta(days=1)

        return file_paths

    def read_data(self):
        all_file_tensors = []
        for file in self.files:
            try:
                if file.endswith('.gz'):
                    with gzip.open(file, 'rb') as f:
                        with rasterio.open(BytesIO(f.read())) as src:
                            data = src.read(1)  
                            tensor = torch.from_numpy(data).float()
                            print(file, tensor)
                            all_file_tensors.append(tensor)
                else:
                    with rasterio.open(file) as src:
                        data = src.read(1)  
                        tensor = torch.from_numpy(data).float()
                        print(file, tensor)
                        all_file_tensors.append(tensor)
            except Exception as e:
                print(f"Error reading file {file}: {e}")

        if all_file_tensors:
            self.data = torch.stack(all_file_tensors)
        else:
            print("No data read, please check the files or paths.")

    def save_tensor(self, tensor, name):
        if tensor is not None:
            np.save(os.path.join(self.root_dir, str(name) + '.npy'), tensor)
            print("Tensor saved successfully with shape:", tensor.shape)
        else:
            print("No tensor to save.")

def get_dataset(root_dir):
    dataset = CHIRP_parser(root_dir)
    return dataset

abs_root_dir = os.path.abspath('/mnt/disks/data/CHIRP')
dataset = get_dataset(abs_root_dir)

# if dataset.data is not None:
#     for x in dataset.data:
        # print(x.shape)
        # print(x)

# print(dataset.data)
# print(dataset.files)

if __name__ == "__main__":
    chirp = CHIRP_parser(abs_root_dir)
    print(chirp.data.shape)
    print(chirp.data)
    # print(chirp.data)
    # print(chirp.files)
    # if chirp.data is not None:
    #     for x in chirp.data:
            # print(x.shape)
            # print(x)
