from dataloader import DatasetFromTxt
from config import cfg
path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
files = [  # "biwi_eth/biwi_eth.txt",
        "SDD/bookstore_0.txt", "SDD/coupa_1.txt", "SDD/deathCircle_4.txt", "SDD/gates_1.txt"
        # "crowds/students001.txt",        "crowds/students003.txt",
        # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
        # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
        # "stanford/coupa_3.txt",
        # "stanford/deathCircle_0.txt",
    ]
# files = ["biwi_eth/biwi_eth.txt", "UCY/students01/students01.txt", "UCY/students03/students03.txt",
#                            "UCY/zara01/zara01.txt", "UCY/zara02/zara02.txt"]
cfg["raster_params"]["use_segm"] = False
dataset = DatasetFromTxt(path_, files, cfg)
dataset[0]
