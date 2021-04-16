import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from config import cfg
from dataloader import DatasetFromTxt
from utils import transform_points
from tqdm import tqdm
import torch
import cv2

draw_n = 1
draw_a_h = 1
draw_speeds = 0
draw_targets = 1
once = 1

def visualize_test():
    Rad = 2
    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["normalize"] = True
    files = [
        "SDD/bookstore_0.txt"
    ]
    path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    
    dataset = DatasetFromTxt(path_, files, cfg)
    # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
    torch.manual_seed(42)
    np.random.seed(42)
    for i in tqdm(range(0, len(dataset))):
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]
        if cfg["raster_params"]["use_map"]:
            if cfg["raster_params"]["normalize"]:
                img = cv2.warpAffine(data[0], data[14][:2, :], (data[0].shape[1], data[0].shape[0]))
                img = img[int(data[15][1]):int(data[15][3]), int(data[15][0]):int(data[15][2])]
                img = cv2.resize(img, (cfg["cropping_cfg"]["image_shape"][0], cfg["cropping_cfg"]["image_shape"][1]))
            else:
                img = data[0]
            img = Image.fromarray(np.asarray(img, dtype="uint8"))
            draw = ImageDraw.Draw(img)
            if data[3][1] == 0:
                continue
            if draw_a_h:
                for num, pose in enumerate(data[2]):
                    if data[3][num]:
                        # pose = data["glob_to_local"] @ np.array([pose[0], pose[1], 1.])
                        pose_raster = data[8] @ np.array([pose[0], pose[1], 1.])
                        draw.ellipse(
                            (pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad),
                            fill='red', outline='red')
                        # if num == 0:
                        if draw_speeds:

                            na = np.asarray(img)
                            if (np.sum(data["agent_speed"][num] ** 2) > 1e-6) and (
                                    data["agent_hist_avail"][num] * data["agent_hist_avail"][num + 1]):
                                speed = data["agent_speed"][num]
                                na = cv2.arrowedLine(na, (int(pose_raster[0]), int(pose_raster[1])), (
                                int(pose_raster[0] + (speed[0] * 300)), int(pose_raster[1] + (speed[1] * 300))), (0, 0, 0),
                                                     8)
                                img = Image.fromarray(np.asarray(na, dtype="uint8"))
                                draw = ImageDraw.Draw(img)

            if draw_n:
                for ped in range(0, data[6].shape[0]):
                    for num, pose in enumerate(data[6][ped, :, :]):
                        if data[7][ped, num]:
                            pose_raster = data[8] @ np.array([pose[0], pose[1], 1.])
                            rgb = (int(255 * ((num + 1) / 8)), int(255 * ((num + 1) / 8)), int(255 * ((num + 1) / 8)))
                            # draw.ellipse((pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad), fill='#ffcc99', outline='#ffcc99')
                            draw.ellipse(
                                (pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad),
                                fill=rgb, outline='#ffcc99')

            if draw_targets:
                for num, pose in enumerate(data[4]):
                    if data[5][num]:
                        R = 4
                        # pose = data["glob_to_local"] @ np.array([pose[0], pose[1], 1.])
                        pose_raster = data[8] @ np.array([pose[0], pose[1], 1.])
                        draw.ellipse(
                            (pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad),
                            fill='#33cc33', outline='#33cc33')
            img.save("test/"+str(i)+".jpg")
        # pix_path = torch.einsum("bki, bji-> bjk", torch.tensor(data.loc_im_to_glob @ data.raster_from_agent).float(),
        #                         path_.float())
        # print()
if __name__ == "__main__":
    from train_test_split import get_train_val_dataloaders
    from dataloader import  collate_wrapper
    import numpy as np

    visualize_test()
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = False

    # path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    files = [
        "SDD"
    ]

    for val_file in files:
        train_ds, val_ds = get_train_val_dataloaders(path_, cfg, val_file, False)
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_ds, batch_size=256,
                                      shuffle=True, num_workers=0, collate_fn=collate_wrapper, pin_memory=True)
        val_dataloader = DataLoader(val_ds, batch_size=256,
                                    shuffle=False, num_workers=0, collate_fn=collate_wrapper)

        train_poses = np.zeros((0, 2))
        val_poses = np.zeros((0, 2))
        from utils import preprocess_data
        import time

        # st = time.time()
        for num, data in enumerate(tqdm(train_dataloader)):
            imgs, masks = preprocess_data(data, cfg, "cuda")
            # print(time.time() - st)

            if num>=50:
                exit()
        for num,data in enumerate(tqdm(val_dataloader)):
            val_poses = np.concatenate((val_poses, data.history_positions[:,:,:2][data.history_av != 0].reshape(-1,2)))
            if num>10:
                break
        exit()




        n_bins = 60

        plt.hist(val_poses[:, 0], n_bins, alpha=0.5, label='val_poses')
        plt.hist(train_poses[:,0], n_bins, alpha=0.5, label='train_poses')
        plt.legend(loc='upper right')
        plt.savefig("tests/"+val_file+'_x.png')
        plt.show()
        plt.close()
        plt.hist(val_poses[:, 1], n_bins, alpha=0.5, label='val_poses')
        plt.hist(train_poses[:, 1], n_bins, alpha=0.5, label='train_poses')
        plt.legend(loc='upper right')
        plt.savefig("tests/"+val_file + '_y.png')
        plt.show()

