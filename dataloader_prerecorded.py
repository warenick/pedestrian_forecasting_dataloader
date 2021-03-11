import numpy as np
from PIL import ImageDraw
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import sys
inJupyter = sys.argv[-1].endswith('json')
if not inJupyter:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

try:
    from scripts.config import cfg
except ImportError:
    import sys
    sys.path.insert(0,'./')
    from scripts.config import cfg

from scripts.utils import sort_neigh_history, sdd_crop_and_rotate, transform_points, sort_neigh_future
torch.multiprocessing.set_sharing_strategy('file_system')
from scripts.dataloader import UnifiedInterface

class DatasetFromPrePorccessedFolder(torch.utils.data.Dataset):

    def __init__(self, path):
        super(DatasetFromPrePorccessedFolder, self).__init__()
        self.path = path

    def __len__(self):
        import os
        length = len([f for f in os.listdir(self.path)if os.path.isfile(os.path.join(self.path, f))])
        return length

    def __getitem__(self, index: int):
        if index == 165740:
            index = 123456
        file = self.path+str(index)+".npy"
        data = np.load(file, allow_pickle=True)

        return data.item()




class DatasetFromPrePorccessedFolder10k(torch.utils.data.Dataset):

    def __init__(self, path):
        super(DatasetFromPrePorccessedFolder10k, self).__init__()
        self.path = path
        import os
        files = []
        for f in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, f)):
                files.append(self.path + f)
        length = 0
        print("loading dataset")
        for i in tqdm(range(len(files))):
            length += np.load(files[i], allow_pickle=True).shape[0]
        self.length = length
    def __len__(self):

        return self.length

    def __getitem__(self, index: int):
        file = self.path+str(index - (index % 10000))+".npy"
        data = np.load(file, allow_pickle=True)

        return data[index % 10000]




if __name__ == "__main__":
    pass
    # path = "/media/robot/hdd1/hdd_repos/trajnet_data/192_192/"
    # dataset = DatasetFromPrePorccessedFolder(path)
    # data = dataset[0]
    # a = data["img"]

    path = "/media/robot/hdd/data/192_192_f/"
    dataset = DatasetFromPrePorccessedFolder10k(path)
    data = dataset[123000]
    a = data["img"]
    pass

# coupa_:
#    video0:
#       scale: 0.027995674
#       certainty: 1.0
#    video1:
#       scale: 0.023224545
#       certainty: 1.0
#    video2:
#       scale: 0.024
#       certainty: 1.0
#    video3:
#       scale: 0.025524906
#       certainty: 1.0

# deathCircle_:
#    video0:
#       scale: 0.04064
#       certainty: 1.0
#    video1:
#       scale: 0.039076923
#       certainty: 1.0
#    video2:
#       scale: 0.03948382
#       certainty: 1.0
#    video3:
#       scale: 0.028478209
#       certainty: 1.0
#    video4:
#       scale: 0.038980137
#       certainty: 1.0

# gates:
#    video0:
#       scale: 0.03976968
#       certainty: 1.0
#    video1:
#       scale: 0.03770837
#       certainty: 1.0
#    video2:
#       scale: 0.037272793
#       certainty: 1.0
#    video3:
#       scale: 0.034515323
#       certainty: 1.0
#    video4:
#       scale: 0.04412268
#       certainty: 1.0
#    video5:
#       scale: 0.0342392
#       certainty: 1.0
#    video6:
#       scale: 0.0342392
#       certainty: 1.0
#    video7:
#       scale: 0.04540353
#       certainty: 1.0
#    video8:
#       scale: 0.045191525
#       certainty: 1.0


# hyang:
#    video0:
#       scale: 0.034749693
#       certainty: 1.0
#    video1:
#       scale: 0.0453136
#       certainty: 1.0
#    video2:
#       scale: 0.054992233
#       certainty: 1.0
#    video3:
#       scale: 0.056642
#       certainty: 1.0
#    video4:
#       scale: 0.034265612
#       certainty: 1.0
#    video5:
#       scale: 0.029655497
#       certainty: 1.0
#    video6:
#       scale: 0.052936449
#       certainty: 1.0
#    video7:
#       scale: 0.03540125
#       certainty: 1.0
#    video8:
#       scale: 0.034592381
#       certainty: 1.0
#    video9:
#       scale: 0.038031423
#       certainty: 1.0
#    video10:
#       scale: 0.054460944
#       certainty: 1.0
#    video11:
#       scale: 0.054992233
#       certainty: 1.0
#    video12:
#       scale: 0.054104065
#       certainty: 1.0

#    video13:
#       scale: 0.0541
#       certainty: 0.0
#    video14:
#       scale: 0.0541
#       certainty: 0.0

# little:
#    video0:
#       scale: 0.028930169
#       certainty: 1.0
#    video1:
#       scale: 0.028543144
#       certainty: 1.0
#    video2:
#       scale: 0.028543144
#       certainty: 1.0
#    video3:
#       scale: 0.028638926
#       certainty: 1.0

# nexus:
#    video0:
#       scale: 0.043986494
#       certainty: 1.0
#    video1:
#       scale: 0.043316805
#       certainty: 1.0
#    video2:
#       scale: 0.042247434
#       certainty: 1.0
#    video3:
#       scale: 0.045883871
#       certainty: 1.0
#    video4:
#       scale: 0.045883871
#       certainty: 1.0
#    video5:
#       scale: 0.045395745
#       certainty: 1.0
#    video6:
#       scale: 0.037929168
#       certainty: 1.0
#    video7:
#       scale: 0.037106087
#       certainty: 1.0
#    video8:
#       scale: 0.037106087
#       certainty: 1.0
#    video9:
#       scale: 0.044917895
#       certainty: 1.0
#    video10:
#       scale: 0.043991753
#       certainty: 1.0
#    video11:
#       scale: 0.043766154
#       certainty: 1.0

# quad:
#    video0:
#       scale: 0.043606807
#       certainty: 1.0
#    video1:
#       scale: 0.042530206
#       certainty: 1.0
#    video2:
#       scale: 0.043338169
#       certainty: 1.0
#    video3:
#       scale: 0.044396842
#       certainty: 1.0

