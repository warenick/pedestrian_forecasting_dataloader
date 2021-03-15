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

if __name__ == "__main__":
    # cfg["raster_params"]["use_map"] = True
    # cfg["raster_params"]["normalize"] = False
    #
    # files = ["UCY/students03/students03.txt"]
    #
    #
    # # files = ["eth_hotel/eth_hotel.txt"]
    files = ["SDD/bookstore_0.txt", "SDD/bookstore_1.txt", "SDD/bookstore_2.txt", "SDD/bookstore_3.txt",
             "SDD/bookstore_4.txt", "SDD/bookstore_5.txt", "SDD/bookstore_6.txt", "SDD/coupa_0.txt", "SDD/coupa_1.txt",
             "SDD/coupa_2.txt", "SDD/coupa_3.txt", "SDD/deathCircle_0.txt", "SDD/deathCircle_1.txt", "SDD/deathCircle_2.txt",
             "SDD/deathCircle_3.txt", "SDD/deathCircle_4.txt"]
    path_ = "./data/train/"
    # dataset = DatasetFromTxt(path_, files, cfg)
    # pix_to_image = dataset.cfg["zara3_pix_to_image_cfg"]
    # pix_to_m = dataset.cfg['zara_h']
    # pix_to_image = dataset.cfg["students_pix_to_image_cfg"]
    # pix_to_m = dataset.cfg['student_h']
    # pix_to_m = dataset.cfg['student_h']


    # pix_to_image = dataset.cfg["eth_hotel_pix_to_image_cfg"]
    # # pix_to_m = dataset.cfg['eth_hotel_h']
    # # pix_to_image = dataset.cfg["eth_univ_pix_to_image_cfg"]
    # # pix_to_m = dataset.cfg['eth_univ_h']
    # img = dataset[0]["img"]
    #
    # cfg["raster_params"]["use_map"] = False
    # cfg["raster_params"]["normalize"] = False
    # dataset = DatasetFromTxt(path_, files, cfg)
    # img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
    #
    # bord = 400
    # img_pil = ImageOps.expand(img_pil, (bord, bord))
    # draw = ImageDraw.Draw(img_pil)
    #
    # R = 2
    #
    # for i in range(0, len(dataset), 80):
    #     data = dataset[i]
    #     agent_history = data["agent_hist"][:, :2]
    #     agent_history = transform_points(agent_history, np.linalg.inv(pix_to_m["scale"]))
    #     for number, pose in enumerate(agent_history):
    #         if data["agent_hist_avail"][number]:
    #             draw.ellipse((pix_to_image["coef_x"] * pose[0] - R + pix_to_image["displ_x"] + bord,
    #                           pix_to_image["coef_y"] * pose[1] - R + pix_to_image["displ_y"] + bord,
    #                           pix_to_image["coef_x"] * pose[0] + R + pix_to_image["displ_x"] + bord,
    #                           pix_to_image["coef_y"] * pose[1] + R + pix_to_image["displ_y"] + bord
    #                           ), fill='blue', outline='blue')
    #
    # img_pil.show()


    #### VISUALIZE TRAJECTORIES

    # radius of point to be plotted
    Rad = 2
    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
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
        img.show()
        print()
        pass
        pass
        # data["img"] = np.asarray(img)
        # np.save(path_to_save + str(i), data)
        # if once:
        #     once = False
        #     plt.imshow(img)
        #     plt.savefig(path_to_save+str(i)+".jpg")
        # # plt.close()
        # # exit()
        # # plt.show()
