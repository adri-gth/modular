
from pathlib import Path
import requests
import zipfile
import os
import argparse

#create a parser

parser = argparse.ArgumentParser(description="Get data from a URL")
parser.add_argument("--name_file", type=str, help=" file's name, with .zip")

parser.add_argument("--urlDownload", type=str, help="URL to download data")


# Get our arguments from the parser
args = parser.parse_args()

nameImageData = args.name_file
downloadUrl = args.urlDownload

#setup path to a data folder
data_path = Path("data/")
image_path = data_path / "image_data"

#if the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
  print(f"{image_path} directory exists.")
else:
  print(f"{image_path} does not exist, creating one...")
  image_path.mkdir(parents=True,exist_ok=True)

#Downloading data
with open(data_path / nameImageData,"wb") as f:
  request = requests.get(downloadUrl)
  print("Downloading data")
  f.write(request.content)

#Unzip dta
with zipfile.ZipFile(data_path / nameImageData,"r") as zip_ref:
  print("Unzipping data.")
  zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / nameImageData)
