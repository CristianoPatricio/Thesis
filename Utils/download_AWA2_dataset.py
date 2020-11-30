import subprocess
from zipfile import ZipFile
import os

# Download dataset.zip from FTP server
print("[INFO]: Downloading dataset...")

url = 'http://www.sal.ipg.pt/user/1012390/datasets/AWA2.zip'
subprocess.run(["wget", url])

# Extract the content of zip to /current directory
filename = 'AWA2.zip'

print("[INFO]: Extracting content of zip...")
with ZipFile(filename, 'r') as zip:
   zip.extractall()

print('[INFO]: Dataset is available at '+str(os.getcwd())+"/AWA2")
