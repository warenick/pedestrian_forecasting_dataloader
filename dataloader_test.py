import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from config import cfg
from dataloader import DatasetFromTxt
from utils import transform_points
from tqdm import tqdm
import torch

draw_n = 1
draw_a_h = 1
draw_speeds = 0
draw_targets = 1
once = 1

def visualize_test():
    Rad = 2
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    files = [
        "eth_hotel/eth_hotel.txt"
    ]
    path_ = "./data/train/"
    dataset = DatasetFromTxt(path_, files, cfg)
    # path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
    for i in tqdm(range(0, len(dataset))):
        ind = int(np.random.rand() * len(dataset))
        data = dataset[ind]

        img = Image.fromarray(np.asarray(data["img"], dtype="uint8"))
        draw = ImageDraw.Draw(img)

        if draw_a_h:
            for num, pose in enumerate(data["agent_hist"]):
                if data["agent_hist_avail"][num]:
                    # pose = data["glob_to_local"] @ np.array([pose[0], pose[1], 1.])
                    pose_raster = data["raster_from_agent"] @ np.array([pose[0], pose[1], 1.])
                    draw.ellipse(
                        (pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad),
                        fill='red', outline='red')
                    # if num == 0:
                    if draw_speeds:
                        import cv2
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
            for ped in range(0, data["neighb"].shape[0]):
                for num, pose in enumerate(data["neighb"][ped, :, :]):
                    if data["neighb_avail"][ped, num]:
                        pose_raster = data["raster_from_agent"] @ np.array([pose[0], pose[1], 1.])
                        rgb = (int(255 * ((num + 1) / 8)), int(255 * ((num + 1) / 8)), int(255 * ((num + 1) / 8)))
                        # draw.ellipse((pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad), fill='#ffcc99', outline='#ffcc99')
                        draw.ellipse(
                            (pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad),
                            fill=rgb, outline='#ffcc99')

        if draw_targets:
            for num, pose in enumerate(data["target"]):
                if data["target_avil"][num]:
                    R = 4
                    # pose = data["glob_to_local"] @ np.array([pose[0], pose[1], 1.])
                    pose_raster = data["raster_from_agent"] @ np.array([pose[0], pose[1], 1.])
                    draw.ellipse(
                        (pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad),
                        fill='#33cc33', outline='#33cc33')
        img.show()
        print()
if __name__ == "__main__":
    from train_test_split import get_train_val_dataloaders
    from dataloader import  collate_wrapper
    import numpy as np
    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["normalize"] = True
    path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    val_files = ["eth_hotel", "biwi_eth", "zara01", "zara02", "students"]
    for val_file in val_files:
        train_ds, val_ds = get_train_val_dataloaders(path_, cfg, val_file, False)
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_ds, batch_size=512,
                                      shuffle=True, num_workers=0, collate_fn=collate_wrapper)
        val_dataloader = DataLoader(val_ds, batch_size=512,
                                    shuffle=False, num_workers=0, collate_fn=collate_wrapper)

        train_poses = np.zeros((0,2))
        val_poses = np.zeros((0, 2))
        for num,data in enumerate(tqdm(train_dataloader)):
            train_poses = np.concatenate((train_poses, data.history_positions[:,:,:2][data.history_av != 0].reshape(-1,2)))
            if num>10:
                break
        for num,data in enumerate(tqdm(val_dataloader)):
            val_poses = np.concatenate((val_poses, data.history_positions[:,:,:2][data.history_av != 0].reshape(-1,2)))
            if num>10:
                break

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

