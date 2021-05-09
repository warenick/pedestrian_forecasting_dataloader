try:
    from dataloader import DatasetFromTxt
    from config import cfg
    from utils import transform_points
except:
    from .dataloader import DatasetFromTxt
    from .config import cfg
    from .utils import transform_points
import torch
import numpy as np


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

def test_img_area_local_meters():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True

    for i in range(30):
        file_index = np.random.randint(0, len(files_all))
        files = [files_all[file_index]]
        cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
        dataset = DatasetFromTxt(path_, files, cfg)
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]
        init_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)

        data = dataset[ind]
        second_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        third_path = data[2].copy()


        # print (np.mean(second_path - init_path)**2)
        assert np.allclose(second_path, init_path, rtol=1.e-3, atol=1.e-3)
        # print(np.mean(third_path - init_path)**2)
        assert np.allclose(init_path, third_path, rtol=1.e-3, atol=1.e-3)


def test_img_area_global_pix():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True

    for i in range(30):
        cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
        file_index = np.random.randint(0, len(files_all))
        files = [files_all[file_index]]
        dataset = DatasetFromTxt(path_, files, cfg)
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]

        init_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]

        second_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        third_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        # assert np.allclose(second_path_pix, init_path_pix, third_path_pix)
        assert np.allclose(second_path_pix, init_path_pix)
        assert np.allclose(init_path_pix, third_path_pix)
        assert np.allclose(init_path_pix[data[3] == 1], data[16][data[3] == 1])

def test_pix_to_m():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True

    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    for i in range(30):
        file_index = np.random.randint(0, len(files_all))
        files = [files_all[file_index]]
        dataset = DatasetFromTxt(path_, files, cfg)
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]
        init_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)

        data = dataset[ind]
        second_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        third_path = data[2].copy()
        rot_mat = data[14]
        rot_mat[:2:, 2] *= 0
        gt_meters = transform_points(data[16][:, :2], data[18])
        gt_meters -= gt_meters[0]
        gt_meters = transform_points(gt_meters, rot_mat)
        # print (np.mean(second_path - init_path)**2)
        assert np.allclose(second_path[:,:2][data[3]==1], gt_meters[data[3]==1], rtol=1.e-3, atol=1.e-3)
        assert np.allclose(second_path, init_path, rtol=1.e-5, atol=1.e-5)
        # print(np.mean(third_path - init_path)**2)
        assert np.allclose(init_path, third_path, rtol=1.e-5, atol=1.e-5)


def test_img_area_local_meters_no_mask():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = False


    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    for i in range(30):
        file_index = np.random.randint(0, len(files_all))
        files = [files_all[file_index]]
        dataset = DatasetFromTxt(path_, files, cfg)
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]
        init_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)
        # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
        data = dataset[ind]
        second_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
        data = dataset[ind]
        third_path = data[2].copy()

        # print(np.mean(second_path - init_path) ** 2)
        assert np.allclose(second_path, init_path, rtol=1.e-3, atol=1.e-3)
        # print(np.mean(third_path - init_path) ** 2)
        assert np.allclose(init_path, third_path, rtol=1.e-3, atol=1.e-3)


def test_img_area_global_pix_no_mask():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = False

    for i in range(30):
        cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
        file_index = np.random.randint(0, len(files_all))
        files = [files_all[file_index]]
        dataset = DatasetFromTxt(path_, files, cfg)
        # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]

        init_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)
        # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
        data = dataset[ind]

        second_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
        data = dataset[ind]
        third_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        # assert np.allclose(second_path_pix, init_path_pix, third_path_pix)
        assert np.allclose(second_path_pix, init_path_pix)
        assert np.allclose(init_path_pix, third_path_pix)


def test_img_area_local_meters_no_norm():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = False
    cfg["raster_params"]["use_segm"] = False

    cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
    file_index = np.random.randint(0, len(files_all))
    files = [files_all[file_index]]
    dataset = DatasetFromTxt(path_, files, cfg)
    for i in range(30):
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]
        init_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        second_path = data[2].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        third_path = data[2].copy()

        assert np.allclose(second_path, init_path, rtol=1.e-3, atol=1.e-3)
        assert np.allclose(init_path, third_path, rtol=1.e-3, atol=1.e-3)


def test_img_area_global_pix_no_norm():
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = False
    cfg["raster_params"]["use_segm"] = False

    for i in range(30):
        cfg["cropping_cfg"]["image_area_meters"] = [20, 20]
        file_index = np.random.randint(0, len(files_all))
        files = [files_all[file_index]]
        dataset = DatasetFromTxt(path_, files, cfg)
        # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]

        init_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]

        second_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        third_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])

        # assert np.allclose(second_path_pix, init_path_pix, third_path_pix)
        assert np.allclose(second_path_pix, init_path_pix)
        assert np.allclose(init_path_pix, third_path_pix)


if __name__ == "__main__":
    from train_test_split import get_train_val_dataloaders
    from dataloader import collate_wrapper

    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True

    test_pix_to_m()

    test_img_area_local_meters()

    test_img_area_global_pix()

    test_img_area_local_meters_no_mask()

    test_img_area_local_meters_no_norm()
    #
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
