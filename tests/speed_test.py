import torch

try:
    from dataloader import DatasetFromTxt, collate_wrapper
    from config import cfg
    from utils import transform_points, preprocess_data
except:
    from .dataloader import DatasetFromTxt, collate_wrapper
    from .config import cfg
    from .utils import transform_points, preprocess_data


import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader



torch.manual_seed(46)
np.random.seed(46)

files_all = ["SDD/bookstore_0.txt", "SDD/bookstore_1.txt", "SDD/bookstore_2.txt", "SDD/bookstore_3.txt",
                 "SDD/bookstore_4.txt", "SDD/bookstore_5.txt", "SDD/bookstore_6.txt",
                 "SDD/coupa_0.txt", "SDD/coupa_2.txt", "SDD/coupa_3.txt",
                 "SDD/deathCircle_0.txt", "SDD/deathCircle_1.txt", "SDD/deathCircle_2.txt", "SDD/deathCircle_3.txt",
                 "SDD/deathCircle_4.txt",
                 "SDD/gates_0.txt", "SDD/gates_1.txt", "SDD/gates_2.txt", "SDD/gates_3.txt", "SDD/gates_4.txt",
                 "SDD/gates_5.txt",  "SDD/gates_7.txt", "SDD/gates_8.txt",
                 "SDD/hyang_0.txt", "SDD/hyang_1.txt", "SDD/hyang_2.txt", "SDD/hyang_3.txt", "SDD/hyang_4.txt",
                 "SDD/hyang_5.txt", "SDD/hyang_6.txt", "SDD/hyang_7.txt", "SDD/hyang_8.txt", "SDD/hyang_9.txt",
                 "SDD/hyang_10.txt", "SDD/hyang_11.txt",
                 "SDD/hyang_12.txt", "SDD/hyang_13.txt", "SDD/hyang_14.txt",
                 "SDD/little_0.txt", "SDD/little_2.txt", "SDD/little_3.txt",
                 "SDD/nexus_0.txt", "SDD/nexus_2.txt", "SDD/nexus_3.txt", "SDD/nexus_4.txt",
                 "SDD/nexus_5.txt", "SDD/nexus_6.txt", "SDD/nexus_7.txt", "SDD/nexus_8.txt", "SDD/nexus_9.txt",
                 "SDD/nexus_10.txt", "SDD/nexus_11.txt",
                 ]
path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"

num_batches_to_load = 50

def test_speed_img_segm_area_local_meters():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True

    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    dataset = DatasetFromTxt(path_, files_all, cfg)
    dataloader = DataLoader(dataset, batch_size=128 + 64,
                            shuffle=True, num_workers=0, collate_fn=collate_wrapper)  # , prefetch_factor=3)

    pbar = tqdm(dataloader)

    for batch_num, data in enumerate(pbar):
        pbar.set_description("dataloader with images and segm")
        if batch_num >= num_batches_to_load:
            break
    print()


def test_speed_img_area_local_meters():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = False

    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    dataset = DatasetFromTxt(path_, files_all, cfg)
    dataloader = DataLoader(dataset, batch_size=128 + 64,
                            shuffle=True, num_workers=8, collate_fn=collate_wrapper)  # , prefetch_factor=3)


    pbar = tqdm(dataloader)

    for batch_num, data in enumerate(pbar):
        pbar.set_description("dataloader with images and no segm")
        if batch_num >= num_batches_to_load:
            break
    print()



def test_speed_img_area_local_meters_preproc():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = False

    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    dataset = DatasetFromTxt(path_, files_all, cfg)
    dataloader = DataLoader(dataset, batch_size=128 + 64,
                            shuffle=True, num_workers=8, collate_fn=collate_wrapper)  # , prefetch_factor=3)


    pbar = tqdm(dataloader)

    for batch_num, data in enumerate(pbar):
        if cfg["raster_params"]["use_map"]:
            imgs, segm = preprocess_data(data, cfg, "cuda")
        pbar.set_description("test_speed_img_area_local_meters_preproc")
        if batch_num >= num_batches_to_load:
            break
    print()


def test_speed_no_imgs():
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["use_segm"] = False

    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    dataset = DatasetFromTxt(path_, files_all, cfg)
    dataloader = DataLoader(dataset, batch_size=128 + 64,
                            shuffle=True, num_workers=8, collate_fn=collate_wrapper)  # , prefetch_factor=3)


    pbar = tqdm(dataloader)

    for batch_num, data in enumerate(pbar):
        pbar.set_description("dataloader with no images and no segm")
        if batch_num >= num_batches_to_load:
            break
    print()


def test_speed_no_imgs_no_norm():
    cfg["raster_params"]["normalize"] = False
    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["use_segm"] = False

    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    dataset = DatasetFromTxt(path_, files_all, cfg)
    dataloader = DataLoader(dataset, batch_size=128 + 64,
                            shuffle=True, num_workers=8, collate_fn=collate_wrapper)  # , prefetch_factor=3)


    pbar = tqdm(dataloader)

    for batch_num, data in enumerate(pbar):
        pbar.set_description("dataloader with no images and no segm, no norm")
        if batch_num >= num_batches_to_load:
            break
    print()





if __name__ == "__main__":
    from train_test_split import get_train_val_dataloaders
    from dataloader import collate_wrapper

    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True

    test_speed_no_imgs()
    test_speed_img_segm_area_local_meters()
    test_speed_img_area_local_meters_preproc()
    test_speed_img_area_local_meters()
    test_speed_no_imgs_no_norm()

    # cfg["raster_params"]["use_map"] = True
    # cfg["raster_params"]["normalize"] = False
    # cfg["raster_params"]["use_segm"] = False
    # visualize_test(cfg, "fmap")
    #
    # cfg["raster_params"]["use_map"] = True
    # cfg["raster_params"]["normalize"] = False
    # cfg["raster_params"]["use_segm"] = True
    # visualize_test(cfg, "fmask")
    # cfg["raster_params"]["use_map"] = True
    # cfg["raster_params"]["normalize"] = True
    # cfg["raster_params"]["use_segm"] = False
    # visualize_test(cfg, "nomask")
    #
    # cfg["raster_params"]["use_map"] = True
    # cfg["raster_params"]["normalize"] = True
    # cfg["raster_params"]["use_segm"] = True
    # visualize_test(cfg, "nm")
