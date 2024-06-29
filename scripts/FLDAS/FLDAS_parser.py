import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
from netCDF4 import Dataset
import xarray as xr
from bounding_box import BoundingBox

# np.set_printoptions(threshold=np.inf)

class FLDAS_parser:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.times = []
        self.files = self.generate_file_paths()
        self.read_data()
        self.save_tensor(self.data, "dataset")
        self.save_tensor(self.times, "times")

        #Spatial alignment
        # [[(a, b), (a, b), (a, b)...], [(a, b), (a, b), (a, b)...]] -> [(a, b), (a,, b)]
        #t * 29 * 256 * 412

    def generate_file_paths(self):
        file_paths = []
        for line in open('fldas_paths.txt').readlines():
            url = line.strip()
            filename = url.split('/')[-1]
            path = os.path.abspath(os.path.join(self.root_dir, filename))
            print(path)
            file_paths.append(path)
        return file_paths

    def read_data(self):
        all_file_tensors = []
        for file in self.files:
            if file.endswith('.nc'):
                # Evap_tavg RadT_tavg SoilMoi00_10cm_tavg Rainf_f_tavg time_bounds
                #FEWS IPC BOUNDING BOX: -17.1 1.7 24.0 27.3
                ds = xr.open_dataset(file)

                # for var_name in ds.data_vars:
                #     print(var_name)
                # return

                list = []
                
                for variable in [
                    'Evap_tavg', 'LWdown_f_tavg', 'Lwnet_tavg', 'Psurf_f_tavg', 'Qair_f_tavg',
                    'Qg_tavg', 'Qh_tavg', 'Qle_tavg', 'Qs_tavg', 'Qsb_tavg', 'RadT_tavg',
                    'Rainf_f_tavg', 'SWE_inst', 'SWdown_f_tavg', 'SnowCover_inst',
                    'SnowDepth_inst', 'Snowf_tavg', 'Swnet_tavg', 'Tair_f_tavg', 'Wind_f_tavg',
                    'SoilMoi00_10cm_tavg', 'SoilMoi10_40cm_tavg', 'SoilMoi40_100cm_tavg',
                    'SoilMoi100_200cm_tavg', 'SoilTemp00_10cm_tavg', 'SoilTemp10_40cm_tavg',
                    'SoilTemp40_100cm_tavg', 'SoilTemp100_200cm_tavg'
                ]:
                    list.append(np.flip(ds[variable].values.squeeze(axis=0), axis=0))
            
                time_bounds = ds['time_bnds'].values.squeeze(axis=0)

                evap_travg = np.flip(ds['Evap_tavg'].values.squeeze(axis=0), axis=0)

                central_value = [evap_travg.shape[0] // 2, evap_travg.shape[1] // 2]
                top_left = [central_value[0] - 148, central_value[1] - 163]
                bottom_right = [central_value[0] + 108, central_value[1] + 248]

                print(central_value, top_left, bottom_right)

                i = 0
                for x in list:
                    list[i] = x[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                    i += 1

                filename = os.path.basename(file)
                a = 0
                for x in list:
                    print(x.shape)
                    plt.imshow(x, cmap='gray')
                    plt.show()
                    output_dir = f"/home/duckb/neurips/scripts/figures/{filename}/"
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(f"/home/duckb/neurips/scripts/figures/{filename}/{filename}_figure_{a}.png")
                    plt.close()
                    a += 1

                tensor = np.stack(list, axis=-1)
                all_file_tensors.append(tensor)
                self.times.append(time_bounds)

                print(file, time_bounds, tensor.shape)
        self.data = all_file_tensors
        
    def save_tensor(self, tensor, name):
        if tensor is not None:
            np.save(os.path.join(self.root_dir, str(name) + '.npy'), tensor)
            print("Tensor saved successfully with shape:", tensor)
        else:
            print("No tensor to save.")

abs_root_dir = os.path.abspath('/mnt/disks/data/FLDAS')

if __name__ == "__main__":
    fldas = FLDAS_parser(abs_root_dir)