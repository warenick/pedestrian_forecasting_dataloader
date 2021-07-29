from tqdm import tqdm
from os import walk
import numpy as np

if __name__=="__main__":
    print("loading data...")
    path = "data/train/SDD/"
    filenames = next(walk(path), (None, None, []))[2]  # [] if no file
    filenames = list(filter(lambda x: "txt" in x,filenames))
    if len(filenames)==0:
        print("can`t found files. let check path")
        exit(1)
    print("loading data - ok")
    print("found "+str(len(filenames))+" files:")
    print(filenames)
    filenames=[path+x for x in filenames]
    iterator = tqdm(filenames)
    for file in iterator:
        # with open(file):
        data = np.loadtxt(file, delimiter=' ', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], dtype = np.int32)
        data_string = np.loadtxt(file, delimiter=' ', usecols=[9], dtype = str)
        out = data[data_string == '"Pedestrian"']
        out = out[out[:,6]==0]
        npyfile = file[:-3]+"npy"
        np.save(npyfile,out)
    print("well done")
