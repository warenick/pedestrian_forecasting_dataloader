import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scripts.config import cfg
from scripts.dataloader import DatasetFromTxt
from tqdm import tqdm
import torch

draw_n = 1
draw_a_h = 0
draw_speeds = 0
draw_targets = 0
once = 1
if __name__ == "__main__":

    files = ["stanford/bookstore_0.txt", "stanford/bookstore_1.txt", "stanford/bookstore_2.txt",
             "stanford/bookstore_3.txt", "stanford/coupa_3.txt", "stanford/deathCircle_0.txt",
             "stanford/deathCircle_1.txt", "stanford/deathCircle_2.txt",
             "stanford/deathCircle_3.txt", "stanford/deathCircle_4.txt",
             "stanford/gates_0.txt", "stanford/gates_1.txt", "stanford/gates_3.txt", "stanford/gates_4.txt",
             "stanford/gates_5.txt", "stanford/gates_6.txt", "stanford/gates_7.txt", "stanford/gates_8.txt",
             "stanford/hyang_5.txt", "stanford/hyang_6.txt", "stanford/hyang_7.txt", "stanford/hyang_9.txt",
             "stanford/nexus_1.txt", "stanford/nexus_3.txt", "stanford/nexus_4.txt", "stanford/nexus_7.txt",
             "stanford/nexus_8.txt", "stanford/nexus_9.txt"]

    path_ = "../data/train/"
    dataset = DatasetFromTxt(path_, files, cfg, use_forces=True, forces_file="forces_19.02.txt")


    #### VISUALIZE TRAJECTORIES

    # radius of point to be plotted
    Rad = 2
    path_to_save = "/home/robot/repos/SDD_forces/192_192_f_n/"
    for i in tqdm(range(0, len(dataset))):
        data = dataset[i]
        # np.save(path_to_save+str(i), data)
        # continue
        #
        # if data["img"] is np.array([None]):
        #     continue
        img = Image.fromarray(np.asarray(data["img"], dtype="uint8"))
        draw = ImageDraw.Draw(img)
        # if np.sum(data["target_avil"]) >= 3:
        #     continue

        if draw_a_h:
            for num, pose in enumerate(data["agent_hist"]):
                if data["agent_hist_avail"][num]:
                    # pose = data["glob_to_local"] @ np.array([pose[0], pose[1], 1.])
                    pose_raster = data["raster_from_agent"] @ np.array([pose[0], pose[1], 1.])
                    draw.ellipse((pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad),
                                 fill='red', outline='red')
                    # if num == 0:
                    if draw_speeds:
                        import cv2
                        na = np.asarray(img)
                        if (np.sum(data["agent_speed"][num]**2) > 1e-6) and (data["agent_hist_avail"][num] * data["agent_hist_avail"][num+1]):
                            speed = data["agent_speed"][num]
                            na = cv2.arrowedLine(na, (int(pose_raster[0]), int(pose_raster[1])), (int(pose_raster[0]+(speed[0]*300)), int(pose_raster[1]+(speed[1]*300))), (0, 0, 0), 8)
                            img = Image.fromarray(np.asarray(na, dtype="uint8"))
                            draw = ImageDraw.Draw(img)

        if draw_n:
            for ped in range(0, data["neighb"].shape[0]):
                for num, pose in enumerate(data["neighb"][ped, :, :]):
                    if data["neighb_avail"][ped, num]:

                        pose_raster = data["raster_from_agent"] @ np.array([pose[0], pose[1], 1.])
                        rgb = (int(255*((num+1)/8)), int(255*((num+1)/8)), int(255*((num+1)/8)))
                        # draw.ellipse((pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad), fill='#ffcc99', outline='#ffcc99')
                        draw.ellipse((pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad), fill=rgb, outline='#ffcc99')

        if draw_targets:
            for num, pose in enumerate(data["target"]):
                if data["target_avil"][num]:
                    R = 4
                    # pose = data["glob_to_local"] @ np.array([pose[0], pose[1], 1.])
                    pose_raster = data["raster_from_agent"] @ np.array([pose[0], pose[1], 1.])
                    draw.ellipse((pose_raster[0] - Rad, pose_raster[1] - Rad, pose_raster[0] + Rad, pose_raster[1] + Rad), fill='#33cc33', outline='#33cc33')
        data["img"] = np.asarray(img)
        np.save(path_to_save + str(i), data)
        if once:
            once = False
            plt.imshow(img)
            plt.savefig(path_to_save+str(i)+".jpg")
        # plt.close()
        # exit()
        # plt.show()
