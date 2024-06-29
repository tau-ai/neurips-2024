import urllib.request
import os
import requests
from datetime import datetime, timedelta
import geopandas as gdp
import fsspec
import os

def download_file(url):
    filename = os.path.basename(url)
    print(url)
    directory = "/mnt/disks/data/CHIRP"
    filepath = os.path.join(directory, filename)
    urllib.request.urlretrieve(url, filepath)

# base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/"

start_date = datetime(2021, 12, 1)
end_date = datetime(2022, 6, 30)

base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/"

# start_date = datetime(2021, 12, 1)
# end_date = datetime(2022, 6, 30)

file_paths = []

current_date = start_date

while current_date <= end_date:

    # file_name = f"chirps-v2.0.{current_date.year}.{current_date.month:02d}.{current_date.day:02d}.tif.gz"
    file_name = f"chirps-v2.0.{current_date.year}.{current_date.month:02d}.{current_date.day:02d}.tif"
    file_path = base_url + str(current_date.year) + "/" + file_name
    file_paths.append(file_path)

    current_date += timedelta(days=1)


for file_path in file_paths:
    download_file(file_path)