import sys

sys.path.insert(0, "../.")
sys.path.insert(0, "../../")

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

files_all = ["biwi_eth/biwi_eth.txt",
             "eth_hotel/eth_hotel.txt",
             "UCY/zara02/zara02.txt",
             "UCY/zara01/zara01.txt",
             "UCY/students01/students01.txt",
             "UCY/students03/students03.txt"
            ]

path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"

cfg["one_ped_one_traj"] = False

def test_ucy_poses_with_and_without_map():
    #  TEST that (future and observed) poses of ETH\UCY is the same with and without map  """

    path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    for i in range(30):
        cfg["raster_params"]["use_map"] = True
        cfg["raster_params"]["normalize"] = True
        cfg["raster_params"]["use_segm"] = True
        file_index = np.random.randint(0, len(files_all))
        files = [files_all[file_index]]

        dataset = DatasetFromTxt(path_, files, cfg)
        ind = int(np.random.rand() * len(dataset))
        init_data = dataset[ind]
        init_path = init_data[2].copy() * init_data.agent_pose_av.reshape(8, 1)
        init_future = init_data[4].copy() * init_data[5].reshape(12, 1)

        cfg["raster_params"]["use_map"] = False
        cfg["raster_params"]["normalize"] = True
        cfg["raster_params"]["use_segm"] = False

        dataset = DatasetFromTxt(path_, files, cfg)
        # ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]
        second_path = data[2].copy() * data[3].reshape(8, 1)
        second_future = data[4].copy() * data[5].reshape(12, 1)
        assert np.allclose(second_future,
                           init_future, rtol=1.e-3, atol=1.e-3)
        assert np.allclose(second_path,
                           init_path, rtol=1.e-3, atol=1.e-3)


def test_img_area_local_meters():
    """
    TEST that cfg["cropping_cfg"]["image_area_meters"] doesn't affect observed and future poses
    """

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
        future = data[4].copy()

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)

        data = dataset[ind]
        second_path = data[2].copy()
        future_2 = data[4].copy()
        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        third_path = data[2].copy()

        assert np.allclose(second_path, init_path, rtol=1.e-3, atol=1.e-3)

        assert np.allclose(init_path, third_path, rtol=1.e-3, atol=1.e-3)

        assert np.allclose(future_2, future)


def test_img_area_global_pix():
    """
        TEST that orig_pixels_hist from dataset same with 'transformed' data and
        doenst depend at image_area_meters parameter
    """
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

        init_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])[:, :2]

        cfg["cropping_cfg"]["image_area_meters"] = [40, 40]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]

        second_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])[:, :2]

        cfg["cropping_cfg"]["image_area_meters"] = [10, 10]
        dataset = DatasetFromTxt(path_, files, cfg)
        data = dataset[ind]
        third_path_pix = transform_points(data[2][:, :2], data[12] @ data[8])[:, :2]

        # assert np.allclose(second_path_pix, init_path_pix, third_path_pix)
        assert np.allclose(second_path_pix, init_path_pix)
        assert np.allclose(init_path_pix, third_path_pix)
        assert np.allclose(init_path_pix[data[3] == 1], data[16][data[3] == 1])


def test_pix_to_m():
    """
    test that agent position (in meters) doesnt depend at cfg["cropping_cfg"]["image_area_meters"] and the same as
    pixel_poses_from_dataset @  data[18]
    :return:
    """
    # SDD ONLY ? ###
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
        gt_meters = transform_points(gt_meters[:, :2], rot_mat)[:, :2]
        # print (np.mean(second_path - init_path)**2)
        assert np.allclose(second_path[:, :2][data[3] == 1], q[data[3] == 1], rtol=1.e-3, atol=1.e-3)
        assert np.allclose(second_path, init_path, rtol=1.e-5, atol=1.e-5)
        # print(np.mean(third_path - init_path)**2)
        assert np.allclose(init_path, third_path, rtol=1.e-5, atol=1.e-5)


def test_img_area_local_meters_no_mask():
    """
    test that image_area_meters doest affect observed path (in meters)
    """
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
    """
    test that observed path in pixels (global image from DS) is the same for different image_area_meters with Map&Norm
    """
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = False

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

        assert np.allclose(second_path_pix, init_path_pix)
        assert np.allclose(init_path_pix, third_path_pix)


def test_img_area_local_meters_no_norm():
    """
    Map without norm: test observed path in meters same for different image_area_meters
    :return:
    """
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
    cfg["one_ped_one_traj"] = False
    test_pix_to_m()
    # test_img_area_local_meters_no_norm()
    # test_img_area_local_meters()

    # test_img_area_local_meters()
    #
    # test_pix_to_m()
    #
    # test_img_area_global_pix()
    #
    # test_img_area_local_meters_no_mask()
    #
    # test_img_area_local_meters_no_norm()
