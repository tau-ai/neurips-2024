import os
import datetime
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt

class FoodSecurityDataParser:
    def __init__(self, shapefile_dir, datasets_dir):
        self.shapefile_dir = shapefile_dir
        self.datasets_dir = datasets_dir
        self.layers = []
        self.times = []
        self.files = []

        self.cache_files()
        self.calculate_common_bounds()
        self.rasterize_shapefiles()
        
        self.save_tensor(self.create_tensor(self.layers), "food_security_index_tensor_validation")
        self.save_tensor(self.create_tensor(self.times), "food_security_index_times_validation")

    def cache_files(self):
        file_list = [file for file in os.listdir(self.shapefile_dir) if file.endswith('.shp')]
        sorted_files = sorted(file_list)
        self.files = sorted_files
        print(self.files)

    def calculate_common_bounds(self):
        all_bounds = []
        for filename in self.files:
            if filename.endswith('.shp'):
                print("Bounding box calculations underway for", filename)
                shapefile_path = os.path.join(self.shapefile_dir, filename)
                gdf = gpd.read_file(shapefile_path)
                all_bounds.append(gdf.total_bounds)

        minx = round(min(bounds[0] for bounds in all_bounds), 1)
        miny = round(min(bounds[1] for bounds in all_bounds), 1)
        maxx = round(max(bounds[2] for bounds in all_bounds), 1)
        maxy = round(max(bounds[3] for bounds in all_bounds), 1)

        print(minx, maxy, maxx, miny)

        resolution = 0.1  # 1 degree ~ 11.1km at equator
        
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        self.transform = from_origin(minx, maxy, resolution, resolution)
        self.width = width
        self.height = height

    def rasterize_shapefiles(self):
        for filename in self.files:
            if filename.endswith('.shp'):
                shapefile_path = os.path.join(self.shapefile_dir, filename)
                self.rasterize_shapefile(shapefile_path, filename)

    def rasterize_shapefile(self, shapefile_path, filename):
        date = self.parse_date_from_filename(filename)

        if date.year < 2019:
            return

        gdf = gpd.read_file(shapefile_path)

        self.validate_gdf(gdf, filename)

        gdf = gdf[gdf['CS'].between(0, 5)]
        gdf['CS'] = gdf['CS'].round().astype(np.int8)

        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['CS']))

        tensor_data = rasterize(
            shapes,
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=0,
            dtype='int8',
            all_touched=True
        )

        print(f"Shapefile {filename} rasterized successfully. Tensor Data: {tensor_data.shape}")
        plt.imshow(tensor_data, cmap='gray')
        plt.show()
        plt.savefig(f"/home/duckb/neurips/scripts/figures/{filename}_figure.png")

        self.layers.append(tensor_data)
        self.times.append(date)

    def parse_date_from_filename(self, filename):
        date = filename.split("_")[1]
        year = int(date[0:4])
        month = int(date[4:6])

        return datetime.datetime(year, month, 1)

    def validate_gdf(self, gdf, filename):
        if 'CS' not in gdf.columns:
            raise ValueError(f"The shapefile {filename} does not contain the 'CS' column.")
        if gdf['CS'].min() < -128 or gdf['CS'].max() > 127:
            raise ValueError(f"The shapefile {filename} contains values outside the int8 range.")

    def create_tensor(self, array):
        stacked_tensor = np.stack(array)
        print(f"Stacked tensor max value: {np.max(stacked_tensor)} and min value: {np.min(stacked_tensor)}")
        return stacked_tensor

    def save_tensor(self, tensor, name):
        np.save(os.path.join(self.datasets_dir, str(str(name) + '.npy')), tensor)
        print("Tensor saved successfully with shape:", tensor.shape)


if __name__ == "__main__":
    shapefile_dir = "/mnt/disks/data/FEWS/ALL_HFIC/West Africa"
    parser = FoodSecurityDataParser(shapefile_dir, "/mnt/disks/data/FEWS/ALL_HFIC")