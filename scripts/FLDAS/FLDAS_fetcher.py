import requests
from netrc import netrc
from requests.auth import HTTPBasicAuth
import os

netrc_auth = netrc()
credentials = netrc_auth.authenticators('urs.earthdata.nasa.gov')
username, _, password = credentials

for line in open('fldas_paths.txt').readlines():
    url = line.strip()
    filename = url.split('/')[-1]
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        # Save the file
        
        path = os.path.abspath(os.path.join('/mnt/disks/data/FLDAS/', filename))
        with open(path, 'wb') as file:
            file.write(response.content)
        print(f"Download complete: {path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

