from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_folder', type=str, default="./data")
args = parser.parse_args()
path = args.path_to_folder

if os.path.isfile(path+'/train.zip'):
    print("File already exists!")
    exit()

gdd.download_file_from_google_drive(file_id='1Rekp4U6IrK81Txg1FmFgxivxWZfxMsGa',
                                    dest_path=path+'/train.zip',
                                    unzip=True)
print ("data path: " + os.path.abspath(path + '/train/'))