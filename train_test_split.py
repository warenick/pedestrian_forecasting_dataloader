import torch
import copy
import torch


# try:
    
# except ImportError:
    

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
try:
    from config import cfg
    from dataloader import DatasetFromTxt
except:
    from pedestrian_forecasting_dataloader.dataloader import DatasetFromTxt
    from pedestrian_forecasting_dataloader.config import cfg
    from pedestrian_forecasting_dataloader.dataloader import collate_wrapper


sdd_files_all = ["SDD/bookstore_0.txt", "SDD/bookstore_1.txt", "SDD/bookstore_2.txt", "SDD/bookstore_3.txt",
                 "SDD/bookstore_4.txt", "SDD/bookstore_5.txt",
                 "SDD/coupa_0.txt", "SDD/coupa_2.txt", "SDD/coupa_3.txt",
                 "SDD/deathCircle_0.txt", "SDD/deathCircle_1.txt", "SDD/deathCircle_2.txt", "SDD/deathCircle_3.txt",

                 "SDD/gates_0.txt", "SDD/gates_1.txt", "SDD/gates_2.txt", "SDD/gates_3.txt", "SDD/gates_4.txt",
                 "SDD/gates_5.txt",  "SDD/gates_7.txt", "SDD/gates_8.txt",
                 "SDD/hyang_0.txt", "SDD/hyang_1.txt", "SDD/hyang_2.txt", "SDD/hyang_3.txt", "SDD/hyang_4.txt",
                 "SDD/hyang_5.txt", "SDD/hyang_6.txt", "SDD/hyang_7.txt", "SDD/hyang_8.txt", "SDD/hyang_9.txt",
                 "SDD/hyang_10.txt", "SDD/hyang_11.txt",
                 "SDD/hyang_12.txt", "SDD/hyang_13.txt",
                 "SDD/little_0.txt", "SDD/little_2.txt", "SDD/little_3.txt",
                 "SDD/nexus_0.txt", "SDD/nexus_2.txt", "SDD/nexus_3.txt", "SDD/nexus_4.txt",
                 "SDD/nexus_5.txt", "SDD/nexus_6.txt", "SDD/nexus_7.txt", "SDD/nexus_8.txt", "SDD/nexus_9.txt",
                 "SDD/nexus_10.txt", "SDD/nexus_11.txt", "SDD/bookstore_6.txt", "SDD/deathCircle_4.txt", "SDD/little_1.txt",
                      "SDD/gates_6.txt", "SDD/nexus_1.txt", "SDD/coupa_1.txt", "SDD/hyang_14.txt"
                 ]



def get_dataloaders(bs=96, num_w=8,
                    path_="/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/",
                    validate_with="SDD"):
    torch.manual_seed(0)
    print("creating dataloader for validating with ", validate_with)
    train_ds, val_ds = get_train_val_dataloaders(path_, cfg, validate_with, False)
    train_dataloader = DataLoader(train_ds, batch_size=bs,
                                  shuffle=True, num_workers=num_w, collate_fn=collate_wrapper)  # , prefetch_factor=3)
    val_dataloader = DataLoader(val_ds, batch_size=bs,
                                shuffle=True, num_workers=num_w, collate_fn=collate_wrapper)

    return train_dataloader, val_dataloader


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=42)
    datasets = {}
    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)
    return train, val


def get_train_val_dataloaders(path_, cfg, validate_with, *args):
    train_files = []
    val_files = []
    if ("SDD" not in validate_with):
        if validate_with == "eth_hotel":
            train_files = ["biwi_eth/biwi_eth.txt", "UCY/students01/students01.txt", "UCY/students03/students03.txt",
                           "UCY/zara01/zara01.txt", "UCY/zara02/zara02.txt"] + sdd_files_all
            val_files = ["eth_hotel/eth_hotel.txt"]
        elif validate_with == "biwi_eth":
            train_files = ["eth_hotel/eth_hotel.txt", "UCY/students01/students01.txt", "UCY/students03/students03.txt",
                           "UCY/zara01/zara01.txt", "UCY/zara02/zara02.txt"]
            val_files = ["biwi_eth/biwi_eth.txt"]
        elif validate_with == "zara01":
            train_files = ["eth_hotel/eth_hotel.txt", "UCY/students01/students01.txt", "UCY/students03/students03.txt",
                           "biwi_eth/biwi_eth.txt", "UCY/zara02/zara02.txt"]
            val_files = ["UCY/zara01/zara01.txt"]
        elif validate_with == "zara02":
            train_files = ["eth_hotel/eth_hotel.txt", "UCY/students01/students01.txt", "UCY/students03/students03.txt",
                           "biwi_eth/biwi_eth.txt", "UCY/zara01/zara01.txt"]
            val_files = ["UCY/zara02/zara02.txt"]
        elif validate_with == "students":
            train_files = ["eth_hotel/eth_hotel.txt", "UCY/zara02/zara02.txt",
                           "biwi_eth/biwi_eth.txt", "UCY/zara01/zara01.txt"]
            val_files = ["UCY/students03/students03.txt", "UCY/students01/students01.txt"]
        else:
            raise NotImplemented
        cfg_c = copy.deepcopy(cfg)
        train_dataset = DatasetFromTxt(path_, train_files, copy.deepcopy(cfg_c), *args)

        cfg_c["one_ped_one_traj"] = False
        val_dataset = DatasetFromTxt(path_, val_files, copy.deepcopy(cfg_c), *args)
        return train_dataset, val_dataset

    elif validate_with == "SDD":
        files_train = ["SDD/bookstore_0.txt", "SDD/bookstore_1.txt", "SDD/bookstore_2.txt", "SDD/bookstore_3.txt",
                 "SDD/bookstore_4.txt", "SDD/bookstore_5.txt",
                 "SDD/coupa_0.txt", "SDD/coupa_2.txt", "SDD/coupa_3.txt",
                 "SDD/deathCircle_0.txt", "SDD/deathCircle_1.txt", "SDD/deathCircle_2.txt", "SDD/deathCircle_3.txt",

                 "SDD/gates_0.txt", "SDD/gates_1.txt", "SDD/gates_2.txt", "SDD/gates_3.txt", "SDD/gates_4.txt",
                 "SDD/gates_5.txt",  "SDD/gates_7.txt", "SDD/gates_8.txt",
                 "SDD/hyang_0.txt", "SDD/hyang_1.txt", "SDD/hyang_2.txt", "SDD/hyang_3.txt", "SDD/hyang_4.txt",
                 "SDD/hyang_5.txt", "SDD/hyang_6.txt", "SDD/hyang_7.txt", "SDD/hyang_8.txt", "SDD/hyang_9.txt",
                 "SDD/hyang_10.txt", "SDD/hyang_11.txt",
                 "SDD/hyang_12.txt", "SDD/hyang_13.txt",
                 "SDD/little_0.txt", "SDD/little_2.txt", "SDD/little_3.txt",
                 "SDD/nexus_0.txt", "SDD/nexus_2.txt", "SDD/nexus_3.txt", "SDD/nexus_4.txt",
                 "SDD/nexus_5.txt", "SDD/nexus_6.txt", "SDD/nexus_7.txt", "SDD/nexus_8.txt", "SDD/nexus_9.txt",
                 "SDD/nexus_10.txt", "SDD/nexus_11.txt",

                 # "SDD/quad_0.txt", "SDD/quad_1.txt", "SDD/quad_2.txt", "SDD/quad_3.txt",
                 ]

        files_test = ["SDD/bookstore_6.txt", "SDD/deathCircle_4.txt", "SDD/little_1.txt",
                      "SDD/gates_6.txt", "SDD/nexus_1.txt", "SDD/coupa_1.txt", "SDD/hyang_14.txt"]

        # val_files = ["UCY/students01/students01.txt", "UCY/students01/students03.txt"]
        train_dataset = DatasetFromTxt(path_, files_train, cfg, *args)
        val_dataset = DatasetFromTxt(path_, files_test, cfg, *args)
        # train_dataset, val_dataset = train_val_dataset(dataset, 0.2)
        # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len],
        #                                                            generator=torch.Generator().manual_seed(42))
        return train_dataset, val_dataset
    else:
        raise NotImplemented


if __name__ == "__main__":
    torch.manual_seed(42)
    from config import cfg
    from dataloader import collate_wrapper
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True
    path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    path_ = "/home/jovyan/ped_dataset/train/"
    train_ds, val_ds = get_train_val_dataloaders(path_, cfg, "SDD", False)
    train_dataloader = DataLoader(train_ds, batch_size=16,
                                  shuffle=True, num_workers=0, collate_fn=collate_wrapper)
    val_dataloader = DataLoader(val_ds, batch_size=4,
                                shuffle=False, num_workers=0, collate_fn=collate_wrapper)

    for counter, data in enumerate(train_dataloader):
        if counter > 64:
            break