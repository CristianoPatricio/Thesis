import subprocess
from zipfile import ZipFile
import os
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True, help='Dataset {AWA2, AWA2_mat}')

args = vars(ap.parse_args())

dataset = args['dataset']

# Download dataset.zip from FTP server
print("[INFO]: Downloading dataset...")

url = 'http://www.sal.ipg.pt/user/1012390/datasets/'+dataset+'.zip'
subprocess.run(["wget", url])

# Extract the content of zip to /current directory
filename = dataset+'.zip'

print("[INFO]: Extracting content of zip...")
with ZipFile(filename, 'r') as zip:
   zip.extractall()

print('[INFO]: Dataset is available at '+str(os.getcwd())+"/"+str(dataset))
