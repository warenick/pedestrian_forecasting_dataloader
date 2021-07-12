import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
import copy
try:

    from trajenetloader import TrajnetLoader
    from utils import crop_image_crowds
    from utils import trajectory_orientation, rotate_image, DataStructure
    from utils import sdd_crop_and_rotate, transform_points
    from config import cfg
    from transformations import ChangeOrigin, Rotate
except ImportError:
    # relative import

    from .trajenetloader import TrajnetLoader
    from .utils import crop_image_crowds
    from .utils import trajectory_orientation, rotate_image
    from .utils import sdd_crop_and_rotate, transform_points, DataStructure
    from .config import cfg
    from .transformations import ChangeOrigin, Rotate

import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# import os

np.random.seed(seed=42)


class DatasetFromTxt(torch.utils.data.Dataset):
    def __init__(self, path, files, cfg_=None, use_forces=False, forces_file=None):
        super(DatasetFromTxt, self).__init__()
        self.loader = TrajnetLoader(path, files, cfg_)
        self.index = 0
        self.cfg = cfg_
        self.files = files
        self.use_forces = use_forces
        if use_forces:
            raise NotImplemented

    def __len__(self):

        if self.cfg["one_ped_one_traj"]:
            return self.loader.unique_ped_ids_len

        return self.loader.data_len

    def __getitem__(self, index: int):

        dataset_index = self.loader.get_subdataset_ts_separator(index)

        # name of file, from which data will be taken, example: 'SDD/bookstore_5.txt'
        file = self.files[dataset_index]

        agent_future, agent_hist_avail, agent_history, forces, hist_avail, \
        neighb_time_sorted_future, neighb_time_sorted_hist, ped_id, \
        target_avil, ts = self.get_all_poses_and_masks(
            dataset_index, file, index)

        assert agent_future.shape == (12, 4)  # time_hor, (ts, ped_id, pose_x, pose_y)
        assert target_avil.shape == (12,)  # time_hor
        assert agent_history.shape == (8, 4)
        assert agent_hist_avail.shape == (8,)  # time_hor
        assert neighb_time_sorted_future.ndim == 3
        assert neighb_time_sorted_future.shape[1:] == (12, 4)
        assert neighb_time_sorted_hist.shape[1:] == (8, 4)

        neighb_future_avail = (neighb_time_sorted_future[:, :, 0] != -1).astype(int)
        img, mask, reshape_and_border_transform = self.loader.get_map(dataset_index, ped_id, ts)

        if not self.cfg["raster_params"]["use_map"]:
            res = self.no_map_prepocessing(agent_future, agent_hist_avail, agent_history, file, forces, hist_avail, img,
                                           mask, neighb_time_sorted_hist, target_avil,
                                           neighb_future=neighb_time_sorted_future,
                                           neighb_future_av=neighb_future_avail)
            return res

        if self.cfg["raster_params"]["normalize"]:

            res = self.crop_and_normilize(agent_future.copy(), agent_hist_avail.copy(), agent_history.copy(), file,
                                          hist_avail, img, target_avil, neighb_time_sorted_hist.copy(), forces, mask,
                                          border_width=self.loader.border_datastes["SDD"], reshape_and_border_transform=reshape_and_border_transform.copy(),
                                          neighb_future=neighb_time_sorted_future.copy(), neighb_future_av=neighb_future_avail)
            # print(os.getpid(), 3, time.time())

        else:

            file, res = self.ssd_unnorm_image(agent_future, agent_hist_avail, agent_history, file, forces,
                                              hist_avail, img, mask, neighb_time_sorted_future,
                                              neighb_time_sorted_hist, target_avil, reshape_and_border_transform)

        # res.append(file)
        if "SDD" not in res.file:
            pseudo_res = self.no_map_prepocessing(agent_future, agent_hist_avail, agent_history, file, forces, hist_avail, img,
                                           mask, neighb_time_sorted_hist, target_avil,
                                           neighb_future=neighb_time_sorted_future,
                                           neighb_future_av=neighb_future_avail)
            res.agent_pose = pseudo_res[2]
            res.target = pseudo_res[4]
            res.neighb_poses = pseudo_res[6]
        return res

    def ssd_unnorm_image(self, agent_future, agent_hist_avail, agent_history, file, forces, hist_avail, img, mask,
                         neighb_time_sorted_future, neighb_time_sorted_hist, target_avil, transform):
        # TODO: merge with main proccess
        dataset = file[:file.index("/")]
        file = file[file.index("/") + 1:file.index(".")]

        pix_to_m = (np.eye(3) * self.cfg['SDD_scales'][file]["scale"])
        pix_to_m[2, 2] = 1
        agent_center_local_pix = transform @ (np.append(agent_history[:, self.loader.coors_row][0], 1))
        co_operator = ChangeOrigin(new_origin=agent_center_local_pix[:2], rotation=np.eye(2))
        raster_to_agent = pix_to_m @ co_operator.transformation_matrix
        agent_hist_localM = transform_points(agent_history[:, self.loader.coors_row],
                                             raster_to_agent @ transform)
        target_localM = transform_points(agent_future[:, self.loader.coors_row], raster_to_agent @ transform)
        neigh_localM = transform_points(neighb_time_sorted_hist[:, :, self.loader.coors_row],
                                        raster_to_agent @ transform)
        neigh_futureM = transform_points(neighb_time_sorted_future[:, :, self.loader.coors_row],
                                         raster_to_agent @ transform)
        raster_from_agent = np.linalg.inv(raster_to_agent)
        raster_from_world = transform
        world_from_agent = np.linalg.inv(transform @ pix_to_m @ co_operator.transformation_matrix)
        agent_from_world = np.linalg.inv(world_from_agent)
        speed = -np.gradient(agent_hist_localM, axis=0) / (12 * self.loader.delta_t[dataset])  # TODO: 12?
        speed[:, 0][speed[:, 0] == 0] += 1e-6
        orientation = np.arctan((speed[:, 1]) / (speed[:, 0]))
        neigh_speed = -np.gradient(neigh_localM, axis=1) / (12 * self.loader.delta_t[dataset])  #
        neigh_speed[:, :, 0][neigh_speed[:, :, 0] == 0] += 1e-6
        neigh_orient = np.arctan((neigh_speed[:, :, 1]) / (neigh_speed[:, :, 0]))
        agent_hist_localM, neigh_localM = self.calc_speed_accel(agent_hist_localM, neigh_localM,
                                                                agent_hist_avail, neigh_localM)
        res = [img.astype(np.uint8), mask, agent_hist_localM, agent_hist_avail, target_localM, target_avil,
               neigh_localM,
               hist_avail, np.linalg.inv(raster_to_agent), raster_from_world, world_from_agent,
               agent_from_world,
               np.linalg.inv(raster_to_agent) @ world_from_agent, forces,
               None,
               None]
        output = DataStructure()
        output.update_from_list(res)
        output.file = dataset+"/"+file
        # print(output.file)
        # print(output.file)
        # print(output.file)
        # print(output.file)
        return dataset+"/"+file, output

    def normilized_img_ucy_eth(self, agent_future, agent_hist_avail, agent_history, file, forces, hist_avail, img, mask,
                               mask_pil, neighb_time_sorted_hist, pix_to_image, pix_to_m, target_avil):
        border_width = 0
        img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
        # border_width = int(abs(img_pil.size[0] - img_pil.size[1]) * 1.5)
        # img_pil = ImageOps.expand(img_pil, (border_width, border_width))
        if mask is not None:
            mask_pil = Image.fromarray(np.asarray(mask, dtype="uint8"))
            # mask_pil = ImageOps.expand(mask_pil, (border_width, border_width))
        if self.cfg["raster_params"]["draw_hist"]:

            draw = ImageDraw.Draw(img_pil)
            R = 5

            for number, pose in enumerate(agent_history[:, 2:]):
                if agent_hist_avail[number]:
                    rgb = (0, 0, 255 // (number + 1))
                    draw.ellipse((pix_to_image["coef_x"] * pose[0] - R + pix_to_image["displ_x"] + border_width,
                                  pix_to_image["coef_y"] * pose[1] - R + pix_to_image["displ_y"] + border_width,
                                  pix_to_image["coef_x"] * pose[0] + R + pix_to_image["displ_x"] + border_width,
                                  pix_to_image["coef_y"] * pose[1] + R + pix_to_image["displ_y"] + border_width
                                  ), fill=rgb, outline='blue')
        # img_pil.show()
        # angle_deg
        angle_deg = -trajectory_orientation(agent_history[0][2:], agent_history[1][2:])
        if "eth" in file:
            angle_deg = -angle_deg
        if agent_hist_avail[1] == 0:
            angle_deg = 0.
        angle_rad = angle_deg / 180 * math.pi
        center = np.array([pix_to_image["coef_x"] * agent_history[0, 2] + pix_to_image["displ_x"],
                           pix_to_image["coef_y"] * agent_history[0, 3] + pix_to_image["displ_y"]])
        img_pil, img_to_rotated, mask_pil = rotate_image(img_pil, angle_deg,
                                                         center=center + np.array([border_width, border_width]),
                                                         mask=mask_pil)
        pix_to_image_matrix = np.array([[pix_to_image["coef_x"], 0, pix_to_image["displ_x"]],
                                        [0, pix_to_image["coef_y"], pix_to_image["displ_y"]],
                                        [0, 0, 1]])
        imBorder_to_image = np.eye(3)
        imBorder_to_image[:2, 2] = -border_width
        pix_to_pibxborder = np.eye(3)
        pix_to_pibxborder[:2, 0] = pix_to_image["coef_x"] * (-border_width)
        pix_to_pibxborder[:2, 1] = pix_to_image["coef_y"] * (-border_width)
        to_rot_mat = img_to_rotated.copy()
        to_rot_mat[:2, 2] = 0
        Rimage_to_word = pix_to_m["scale"] @ np.linalg.inv(
            pix_to_image_matrix) @ img_to_rotated @ imBorder_to_image
        image_to_word = pix_to_m["scale"] @ np.linalg.inv(pix_to_image_matrix) @ imBorder_to_image
        center_img = pix_to_image_matrix @ (np.array([agent_history[0][2], agent_history[0][3], 1])) \
                     + np.array([border_width, border_width, 0])
        center_img_wb = pix_to_image_matrix @ (np.array([agent_history[0][2], agent_history[0][3], 1]))
        agent_history_m = pix_to_m["scale"] @ np.array([agent_history[0][2], agent_history[0][3], 1])
        error = Rimage_to_word @ (center_img) - agent_history_m
        crop_img, scale, mask_pil_crop = crop_image_crowds(img_pil, self.cfg["cropping_cfg"],
                                                           agent_center_img=center_img,
                                                           transform=Rimage_to_word, rot_mat=to_rot_mat, file=file,
                                                           mask=mask_pil)
        world_to_agent_matrix = np.eye(3)
        world_to_agent_matrix[:, 2] = -agent_history_m
        pix_to_agent = world_to_agent_matrix @ pix_to_m["scale"]
        agent_hist_localM = transform_points(agent_history[:, self.loader.coors_row], pix_to_agent)
        target_localM = transform_points(agent_future[:, self.loader.coors_row], pix_to_agent)
        neigh_localM = transform_points(neighb_time_sorted_hist[:, :, self.loader.coors_row], pix_to_agent)
        # crop -> image, new_transf
        #
        agent_form_raster = pix_to_agent @ np.linalg.inv(pix_to_image_matrix)
        agent_form_raster_ = np.eye(3)
        agent_form_raster_[:2, 2] = -np.array(self.cfg["cropping_cfg"]["agent_center"]) * np.array(
            self.cfg["cropping_cfg"]["image_shape"])
        agent_form_raster__ = np.eye(3)
        agent_form_raster__[:2, :2] = Rimage_to_word[:2, :2] / np.array(scale)
        agent_form_raster = agent_form_raster__ @ agent_form_raster_
        raster_from_agent = np.linalg.inv(agent_form_raster)
        raster_from_world = agent_form_raster @ world_to_agent_matrix
        loc_im_to_glob = np.linalg.inv(
            (np.append(np.array(scale), 1) * np.eye(3)) @ img_to_rotated @ np.linalg.inv(imBorder_to_image))
        agent_hist_localM, neigh_localM = self.calc_speed_accel(agent_hist_localM, neigh_localM,
                                                                agent_hist_avail, hist_avail)
        res = {"img": np.copy(crop_img),
               "segm": np.copy(mask_pil_crop),
               "agent_hist": agent_hist_localM,
               "agent_hist_avail": agent_hist_avail,
               "target": target_localM,
               "target_avil": target_avil,
               "neighb": neigh_localM,
               "neighb_avail": hist_avail,

               "raster_from_agent": raster_from_agent,
               "raster_from_world": raster_from_world,  # raster_from_world,
               "world_from_agent": np.linalg.inv(world_to_agent_matrix),  # world_from_agent,
               "agent_from_world": world_to_agent_matrix,  # agent_from_world
               "forces": transform_points(forces, to_rot_mat),
               "loc_im_to_glob": np.linalg.inv(loc_im_to_glob)
               }
        return res

    def get_ucy_eth_img_mask_scale(self, agent_future, agent_history, file, img, mask, neighb_time_sorted_hist):
        pix_to_image = {}
        pix_to_m = np.eye(3)
        if "zara" in file:
            img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
            mask_pil = None
            if mask is not None:
                mask_pil = Image.fromarray(np.asarray(mask, dtype="uint8"))
            if "zara02" in file:
                pix_to_image = self.cfg["zara2_pix_to_image_cfg"]
            elif "zara01" in file:

                pix_to_image = self.cfg["zara1_pix_to_image_cfg"]
            else:
                pix_to_image = self.cfg["zara3_pix_to_image_cfg"]
                img_pil = img_pil.resize([int(img_pil.size[0] * 0.8), int(img_pil.size[1] * 0.8)])
                if mask is not None:
                    mask_pil = mask_pil.resize([int(img_pil.size[0] * 0.8), int(img_pil.size[1] * 0.8)], 0)
            pix_to_m = self.cfg["zara_h"]

            if not "zara03" in file:
                if mask is not None:
                    mask_pil = mask_pil.rotate(90, expand=1, center=(img_pil.size[0] / 2, img_pil.size[1] / 2))
                img_pil = img_pil.rotate(90, expand=1, center=(img_pil.size[0] / 2, img_pil.size[1] / 2))
            img = np.asarray(img_pil, dtype="uint8")
            if mask is not None:
                mask_np = np.asarray(mask_pil, dtype="uint8")
            agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                neighb_time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))

        elif "students" in file:
            pix_to_image = self.cfg["students_pix_to_image_cfg"]
            pix_to_m = self.cfg["student_h"]
            agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                neighb_time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))

        elif "biwi_eth" in file:
            pix_to_image = self.cfg["eth_univ_pix_to_image_cfg"]
            pix_to_m = self.cfg['eth_univ_h']
            img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
            img_pil = img_pil.rotate(90, expand=1, center=(img_pil.size[0] / 2, img_pil.size[1] / 2))
            img_pil = img_pil.resize([int(img_pil.size[0] * 1.3), int(img_pil.size[1] * 1)])
            img = np.asarray(img_pil, dtype="uint8")
            mask_pil = None
            if mask is not None:
                mask_pil = Image.fromarray(np.asarray(mask, dtype="uint8")).rotate(90, expand=1,
                                                                                   center=(img_pil.size[0] / 2,
                                                                                           img_pil.size[1] / 2))
                mask_pil = mask_pil.resize([int(img_pil.size[0] * 1.3), int(img_pil.size[1] * 1)])
                mask_np = np.asarray(mask_pil, dtype="uint8")

            agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                neighb_time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
        elif "eth_hotel" in file:
            pix_to_image = self.cfg["eth_hotel_pix_to_image_cfg"]
            pix_to_m = self.cfg['eth_hotel_h']
            img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
            img_pil = img_pil.rotate(90, expand=0, center=(img_pil.size[0] / 2, img_pil.size[1] / 2))
            img_pil = img_pil.resize([int(img_pil.size[0] * 2), int(img_pil.size[1] * 2)])
            img = np.asarray(img_pil, dtype="uint8")
            mask_pil = None
            if mask is not None:
                mask_pil = Image.fromarray(np.asarray(mask, dtype="uint8")).rotate(90, expand=1,
                                                                                   center=(img_pil.size[0] / 2,
                                                                                           img_pil.size[1] / 2))
                mask_pil = mask_pil.resize([int(img_pil.size[0] * 2), int(img_pil.size[1] * 2)])
                mask_np = np.asarray(mask_pil, dtype="uint8")

            agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                neighb_time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
        else:
            raise NotImplemented
        return img, mask_pil, pix_to_image, pix_to_m

    def no_map_prepocessing(self, agent_future, agent_hist_avail, agent_history, file, forces, hist_avail, img, mask,
                            neighb_time_sorted_hist, target_avil, neighb_future, neighb_future_av):

        if "SDD" not in file :
            pix_to_m = {"scale": self.loader.homography[file]}
            agent_future[:, 2:] = world2image(agent_future[:, 2:], np.linalg.inv(self.loader.homography[file]))

            agent_history[:, 2:] = world2image(agent_history[:, 2:], np.linalg.inv(self.loader.homography[file]))
            for i in range(len(neighb_time_sorted_hist)):
                neighb_time_sorted_hist[i][:, 2:] = world2image(neighb_time_sorted_hist[i][:, 2:],
                                                         np.linalg.inv(self.loader.homography[file]))
            for i in range(len(neighb_future)):
                neighb_future[i][:, 2:] = world2image(neighb_future[i][:, 2:],
                                                      np.linalg.inv(self.loader.homography[file]))
            if "eth" in file or "UCY" in file:
                agent_future[:, 2:] = np.flip(agent_future[:, 2:], axis=1)
                agent_history[:, 2:] = np.flip(agent_history[:, 2:], axis=1)
                neighb_time_sorted_hist[:, :, 2:] = np.flip(neighb_time_sorted_hist[:, :, 2:], axis=2)
                neighb_future[:, :, 2:] = np.flip(neighb_future[:, :, 2:], axis=2)

        # # TODO transform neighb_future
        # if "zara" in file:
        #     pix_to_m = self.cfg["zara_h"]
        #     agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
        #         neighb_time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
        # elif "students" in file:
        #     pix_to_m = self.cfg["student_h"]
        #     agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
        #         neighb_time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
        if ("stanford" in file) or ("SDD" in file):
            dataset = file[file.index("/") + 1:file.index(".")]
            pix_to_m = np.eye(3) * self.cfg['SDD_scales'][dataset]["scale"]
            pix_to_m[2, 2] = 1
            pix_to_m = {"scale": pix_to_m}
        # elif "biwi_eth" in file:
        #     pix_to_m = self.cfg['eth_univ_h']
        #     agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
        #         neighb_time_sorted_hist[:, :, self.loader.coors_row],
        #         np.linalg.inv(pix_to_m["scale"]))
        # elif "eth_hotel" in file:
        #     pix_to_m = self.cfg['eth_hotel_h']
        #     agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
        #     neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
        #         neighb_time_sorted_hist[:, :, self.loader.coors_row],
        #         np.linalg.inv(pix_to_m["scale"]))
        #
        # if "eth" in file or "UCY" in file:
        #     agent_future[:, 2:] = np.flip(agent_future[:, 2:], axis=1)
        #     agent_history[:, 2:] = np.flip(agent_history[:, 2:], axis=1)
        #     neighb_time_sorted_hist[:, :, 2:] = np.flip(neighb_time_sorted_hist[:, :, 2:], axis=2)
        #     neighb_future[:, :, 2:] = np.flip(neighb_future[:, :, 2:], axis=2)

        elif "ros" in file:
            pix_to_m = {"scale":self.loader.homography[file]} #self.cfg['ros_h']
            agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
            neighb_time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                neighb_time_sorted_hist[:, :, self.loader.coors_row],
                np.linalg.inv(pix_to_m["scale"]))



        agent_history = transform_points(agent_history[:, 2:], pix_to_m["scale"])
        agent_future = transform_points(agent_future[:, 2:], pix_to_m["scale"])
        neighb_future = transform_points(neighb_future[:, :, self.loader.coors_row],
                                         pix_to_m["scale"])
        neighb_time_sorted_hist = transform_points(neighb_time_sorted_hist[:, :, self.loader.coors_row],
                                                   pix_to_m["scale"])
        to_localM_transform = np.eye(3)
        if self.cfg["raster_params"]["normalize"]:
            angle_deg = trajectory_orientation(agent_history[0][:2], agent_history[1][:2])
            if agent_hist_avail[1] == 0:
                angle_deg = 0
            co_operator = ChangeOrigin(new_origin=agent_history[0][:2], rotation=np.eye(2))
            r_operator = Rotate(angle=angle_deg, rot_center=agent_history[0][:2])
            to_localM_transform = co_operator.transformation_matrix @ r_operator.transformation_matrix
        else:
            co_operator = ChangeOrigin(new_origin=np.array([0., 0]), rotation=np.eye(2))
            to_localM_transform = co_operator.transformation_matrix
        agent_history = transform_points(agent_history, to_localM_transform)
        agent_future = transform_points(agent_future, to_localM_transform)
        neighb_future = transform_points(neighb_future, to_localM_transform)
        neigh_localM = transform_points(neighb_time_sorted_hist, to_localM_transform)

        raster_from_agent = np.linalg.inv(pix_to_m["scale"]) @ np.linalg.inv(co_operator.transformation_matrix)
        agent_history, neigh_localM = self.calc_speed_accel(agent_history, neigh_localM, agent_hist_avail,
                                                            hist_avail)
        res = [img, mask, agent_history, agent_hist_avail, agent_future, target_avil,
               neigh_localM, hist_avail, raster_from_agent, np.eye(3), np.eye(3), np.eye(3),
               np.eye(3), forces, None, None, None, None, pix_to_m["scale"], file, neighb_future, neighb_future_av]
        return res

    def get_all_poses_and_masks(self, dataset_index, file, index):
        # get ped_id and ts from index and dataset
        ped_id, ts = self.loader.get_pedId_and_timestamp_by_index(dataset_index, index)
        # get all ped_ids available at specified timestamp
        indexes = self.loader.get_all_agents_with_timestamp(dataset_index, ts)
        assert sum(indexes < 0) == 0, "negative ped_ids are not allowed"
        try:
            argsort_inexes = self.loader.argsort_inexes[file]
        except:
            argsort_inexes = None
        # all agents with ped_id history
        agents_history = self.loader.get_agent_history(dataset_index, ped_id, ts, indexes, argsort_inexes)
        seen_peds_ds = agents_history[:, 0, 1]
        # all agents with ped_id future
        agents_future = self.loader.get_agent_future(dataset_index, ped_id, ts, indexes, argsort_inexes)
        # current agent (to be predicted) history
        agent_history = agents_history[0]
        # current agent (to be predicted) future
        agent_future = agents_future[0]
        # neighbours history
        neighb_time_sorted_hist = np.array(agents_history[1:])  # sort_neigh_history(others_history)
        # neighbours future
        neighb_time_sorted_future = np.array(agents_future[1:])  # sort_neigh_future(others_future)
        forces = np.zeros(6)
        if self.use_forces:
            forces = self.force_from_txt.get(index)
        if len(neighb_time_sorted_future) == 0:
            neighb_time_sorted_future = np.zeros((0, agent_future.shape[0], agent_future.shape[1]))
        if len(neighb_time_sorted_hist) == 0:
            neighb_time_sorted_hist = np.zeros((0, agent_history.shape[0], agent_history.shape[1]))

        agent_hist_avail = (agent_history[:, 0] != -1).astype(int)
        target_avil = (agent_future[:, 0] != -1).astype(int)
        hist_avail = (neighb_time_sorted_hist[:, :, 0] != -1).astype(int)
        return agent_future, agent_hist_avail, agent_history, forces, hist_avail, neighb_time_sorted_future, \
               neighb_time_sorted_hist, ped_id, target_avil, ts

    def calc_speed_accel(self, agent_history, neigh_localM, agent_history_av, neigh_localM_av):
        real_agent_history = 1 * agent_history
        real_agent_history[np.sum(agent_history_av):] = real_agent_history[np.sum(agent_history_av) - 1]
        speed = -np.gradient(real_agent_history, axis=0) / 0.4
        speed[:, 0][speed[:, 0] == 0] += 1e-6
        speed[agent_history_av == 0] *= 0
        acc = -np.gradient(speed, axis=0)
        acc[:, 0][acc[:, 0] == 0] += 1e-6
        acc[agent_history_av == 0] *= 0
        # TODO: unprecise [1, 0, 0] -> grad: [not0, not0, 0]

        neigh_speed = -np.gradient(neigh_localM, axis=1) / 0.4
        neigh_speed[:, :, 0][neigh_speed[:, :, 0] == 0] += 1e-6
        neigh_speed[neigh_localM_av == 0] *= 0
        neigh_acc = -np.gradient(neigh_speed, axis=1)
        neigh_acc[:, :, 0][neigh_acc[:, :, 0] == 0] += 1e-6
        neigh_acc[neigh_localM_av == 0] *= 0
        agent_history = np.concatenate((agent_history, speed, acc), axis=1)
        neigh_localM = np.concatenate((neigh_localM, neigh_speed, neigh_acc), axis=2)
        return agent_history, neigh_localM

    def crop_and_normilize(self, agent_future, agent_hist_avail, agent_history, folder_file, hist_avail, img,
                           target_avil,
                           time_sorted_hist, forces, mask, border_width, reshape_and_border_transform, neighb_future=None,
                           neighb_future_av=None):
        output = DataStructure()
        output.neighb_target_av = neighb_future_av
        # rotate in a such way that last hisory points are horizontal (elft to right),
        # crop to spec in cfg area and resize
        #  calcultate transformation matrixes for pix to meters
        #  transform poses from pix to meters (local CS)

        if "SDD" not in folder_file:
            agent_future[:, 2:] = world2image(agent_future[:, 2:], np.linalg.inv(self.loader.homography[folder_file]))

            agent_history[:, 2:] = world2image(agent_history[:, 2:], np.linalg.inv(self.loader.homography[folder_file]))
            for i in range(len(time_sorted_hist)):
                time_sorted_hist[i][:, 2:] = world2image(time_sorted_hist[i][:, 2:],
                                                         np.linalg.inv(self.loader.homography[folder_file]))
            for i in range(len(neighb_future)):
                neighb_future[i][:, 2:] = world2image(neighb_future[i][:, 2:],
                                                      np.linalg.inv(self.loader.homography[folder_file]))
            if "eth" in folder_file or "UCY" in folder_file:
                agent_future[:, 2:] = np.flip(agent_future[:, 2:], axis=1)
                agent_history[:, 2:] = np.flip(agent_history[:, 2:], axis=1)
                time_sorted_hist[:, :, 2:] = np.flip(time_sorted_hist[:, :, 2:], axis=2)
                neighb_future[:, :, 2:] = np.flip(neighb_future[:, :, 2:], axis=2)

        file = folder_file[folder_file.index("/") + 1:folder_file.index(".")]

        # save file, and initial (from dataset) positions and availabilities
        output.file = file
        output.orig_pixels_hist = np.copy(agent_history[:, self.loader.coors_row])
        output.orig_pixels_future = np.copy(agent_future[:, self.loader.coors_row])
        output.agent_pose_av = agent_hist_avail
        output.target_av = target_avil
        output.neighb_poses_av = hist_avail

        tl_y, tl_x, br_y, br_x = (0, 0, 0, 0)
        if ("SDD" in folder_file) or ("sdd" in folder_file):
            pix_to_m = np.eye(3) * self.cfg['SDD_scales'][file]["scale"]
            pix_to_m[2, 2] = 1
            scale_factor = self.loader.resize_datastes["SDD"]
        else:
            pix_to_m = self.loader.homography[folder_file]
            scale_factor = self.loader.resize_datastes["ETH/UCY"]

        img, _, final_image_resize_coef, mask, _, (tl_y, tl_x, br_y, br_x), \
        rotation_matrix = sdd_crop_and_rotate(img, agent_history[:, 2:],
                                              draw_traj=self.cfg["raster_params"]["draw_hist"],
                                              pix_to_meters=pix_to_m,
                                              cropping_cfg=self.cfg['cropping_cfg'],
                                              file=file,
                                              mask=mask,
                                              scale_factor=scale_factor,
                                              transform=reshape_and_border_transform,
                                              neighb_hist=time_sorted_hist,
                                              neighb_hist_avail=hist_avail, agent_hist_avail=agent_hist_avail)

        import cv2

        ###
        agent_center_intermidiate = reshape_and_border_transform @ np.append(agent_history[:, 2:][0], 1)
        agent_center_intermidiate = agent_center_intermidiate / agent_center_intermidiate[2]
        # scale_ = self.cfg['SDD_scales'][file]["scale"]
        # pix_to_meters = (np.eye(3) * scale_)
        radius = np.sqrt(2) * (round(
            max(np.linalg.norm(agent_center_intermidiate[:2] - np.array([tl_x, tl_y])),
                np.linalg.norm(agent_center_intermidiate[:2] - np.array([br_x, br_y])))))

        img_ = img[max(int(agent_center_intermidiate[1] - radius),0): int(agent_center_intermidiate[1] + radius),
               max(0, int(agent_center_intermidiate[0] - radius)): int(agent_center_intermidiate[0] + radius)]

        if mask is not None:
            mask = mask[int(agent_center_intermidiate[1] - radius): int(agent_center_intermidiate[1] + radius),
                        int(agent_center_intermidiate[0] - radius): int(agent_center_intermidiate[0] + radius)]

        tl_x_intermidiate = tl_x - (agent_center_intermidiate[0] - radius)
        tl_y_intermidiate = tl_y - (agent_center_intermidiate[1] - radius)
        br_y_intermidiate = br_y - (agent_center_intermidiate[1] - radius)
        br_x_intermidiate = br_x - (agent_center_intermidiate[0] - radius)

        # assert (tl_x_intermidiate - br_x_intermidiate) < 0
        assert (tl_y_intermidiate - br_y_intermidiate) < 0
        assert tl_x_intermidiate > 0
        assert tl_y_intermidiate > 0

        output.cropping_points = np.array([tl_x_intermidiate, tl_y_intermidiate, br_x_intermidiate, br_y_intermidiate])

        # TODO: depend at pic size in meters!
        size_constant = self.cfg["cropping_cfg"]["image_area_meters"][0] * 0.08
        datasaet_name = "SDD" if "SDD" in file else "ETH/UCY"
        img_size = (
            int(size_constant * self.loader.img_size[datasaet_name][0]),
            int(size_constant * self.loader.img_size[datasaet_name][1]))
        new_im = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        new_im[:img_.shape[0], :img_.shape[1]] = np.copy(img_)
        del img_
        img = new_im
        assert img.shape[0] > br_x_intermidiate and img.shape[0] > tl_x_intermidiate
        assert img.shape[1] > br_y_intermidiate and img.shape[1] > tl_y_intermidiate
        output.img = img

        co_operator = ChangeOrigin(
            new_origin=(agent_center_intermidiate[0] - radius, agent_center_intermidiate[1] - radius),
            rotation=np.eye(2))

        rotation_matrix_cropped = co_operator.transformation_matrix @ rotation_matrix @ np.linalg.inv(
            co_operator.transformation_matrix)

        output.rot_mat = rotation_matrix_cropped
        output.forces = transform_points(forces, rotation_matrix_cropped)

        tl_co_operator = ChangeOrigin(
            new_origin=((tl_x, tl_y)),
            rotation=np.eye(2))

        intermidiate_to_locraster = final_image_resize_coef @ tl_co_operator.transformation_matrix @ rotation_matrix

        if mask is not None:
            new_mask = np.zeros((img_size[0], img_size[1], 1), dtype=np.uint8)
            new_mask[:mask.shape[0], :mask.shape[1]] = np.copy(mask)
            mask = new_mask.astype(np.uint8)

        output.mask = mask

        # pix_to_m = np.eye(3) * self.cfg['SDD_scales'][file]["scale"]
        # pix_to_m[2, 2] = 1

        output.pix_to_m = pix_to_m

        initial_resize = reshape_and_border_transform.copy()
        initial_resize[:2, 2:] *= 0
        new_origin_operator = ChangeOrigin(new_origin=(intermidiate_to_locraster @ agent_center_intermidiate)[:2],
                                           rotation=np.eye(2))
        no_tm = new_origin_operator.transformation_matrix

        agent_from_raster = np.linalg.inv(initial_resize) @ np.linalg.inv(final_image_resize_coef) @ pix_to_m @ no_tm

        global_agent_pix_pose_co_operator = ChangeOrigin(
            new_origin=(agent_history[:, self.loader.coors_row][0]),
            rotation=np.eye(2))

        global_center_in_local_cs = pix_to_m[:, 2] / pix_to_m[2, 2]
        global_center_to_local_center = ChangeOrigin(
            new_origin=global_center_in_local_cs,
            rotation=np.eye(2))
        # agent_from_glob_raster = np.linalg.inv(scale) @ pix_to_m @ new_origin_operator.transformation_matrix @ scale @ tl_co_operator.transformation_matrix @ rotation_matrix @ transform #@ np.append(agent_history[:, self.loader.coors_row][0], 1) @ np.append(agent_history[:, self.loader.coors_row][0], 1)
        agent_from_glob_raster = pix_to_m @ global_agent_pix_pose_co_operator.transformation_matrix @ np.linalg.inv(
            reshape_and_border_transform) @ rotation_matrix @ reshape_and_border_transform
        # global_pix_from_raster = np.linalg.inv(transform) @ np.linalg.inv(globPix_to_locraster)
        world_from_raster = pix_to_m @ np.linalg.inv(intermidiate_to_locraster @ reshape_and_border_transform)

        output.loc_im_to_glob = np.linalg.inv(pix_to_m) @ world_from_raster

        raster_from_world = np.linalg.inv(world_from_raster)
        world_from_agent = world_from_raster @ np.linalg.inv(agent_from_raster)
        agent_from_world = np.linalg.inv(world_from_agent)

        output.agent_from_world = agent_from_world
        output.world_from_agent = world_from_agent
        output.raster_from_world = raster_from_world
        output.raster_from_agent = np.linalg.inv(agent_from_raster)

        agent_hist_GlobalM = transform_points(agent_history[:, self.loader.coors_row],
                                              agent_from_glob_raster)
        agent_hist_localM = transform_points(agent_hist_GlobalM[:, :2],
                                             global_center_to_local_center.transformation_matrix)[:, :2]
        neigh_GlobalM = transform_points(time_sorted_hist[:, :, self.loader.coors_row],
                                         agent_from_glob_raster)
        neigh_localM = transform_points(neigh_GlobalM[:, :, :2], global_center_to_local_center.transformation_matrix)

        neighb_future_globalM = transform_points(neighb_future[:, :, self.loader.coors_row], agent_from_glob_raster)
        neighb_future_localM = transform_points(neighb_future_globalM[:, :, :2],
                                                global_center_to_local_center.transformation_matrix)

        target_globalM = transform_points(agent_future[:, self.loader.coors_row], agent_from_glob_raster)
        target_localM = transform_points(target_globalM[:, :2], global_center_to_local_center.transformation_matrix)[:,
                        :2]
        agent_hist_localM, neigh_localM = self.calc_speed_accel(agent_hist_localM, neigh_localM, agent_hist_avail,
                                                                hist_avail)

        output.agent_pose = agent_hist_localM
        output.target = target_localM
        output.neighb_poses = neigh_localM
        output.neighb_target = neighb_future_localM

        # res = [img.astype(np.uint8), mask, agent_hist_localM, agent_hist_avail, target_localM, target_avil,
        #        neigh_localM,
        #        hist_avail, np.linalg.inv(agent_from_raster), raster_from_world, world_from_agent, agent_from_world,
        #        np.linalg.inv(pix_to_m) @ world_from_raster, transform_points(forces, rotation_matrix_cropped),
        #        rotation_matrix_cropped,
        #        np.array([tl_x_intermidiate, tl_y_intermidiate, br_x_intermidiate, br_y_intermidiate]),
        #        agent_history[:, self.loader.coors_row], agent_future[:, self.loader.coors_row], pix_to_m]
        return output


def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam / traj_cam[2])
    return traj_uvz[:, :2].astype(int)


class NeighboursHistory:

    def __init__(self, history_agents):
        self.history_agents = history_agents
        self.bs = len(self.history_agents)

    def get_current_poses(self):
        poses = []
        for i in range(self.bs):
            poses.append(self.history_agents[i][0])
        return poses

    def get_history(self):
        poses = []
        for batch in range(self.bs):
            poses.append([self.history_agents[batch][i] for i in range(0, len(self.history_agents[batch]))])
        return poses


import time


class UnifiedInterface:
    def __init__(self, data):
        """

        :param data: List of list
        example:
        data[0] = [img.astype(np.uint8), mask, agent_hist_localM, agent_hist_avail, target_localM, target_avil,
               neigh_localM, hist_avail, np.linalg.inv(agent_from_raster), raster_from_world, world_from_agent,
               agent_from_world, np.linalg.inv(pix_to_m) @ world_from_raster,
               transform_points(forces, rotation_matrix_cropped), rotation_matrix_cropped, np.array([tl_x_intermidiate,
               tl_y_intermidiate, br_x_intermidiate, br_y_intermidiate]), agent_history[:, self.loader.coors_row],
               agent_future[:, self.loader.coors_row], pix_to_m, file]
        """
        st = time.time()
        if data[0][0] is None:
            self.image = None
        else:
            self.image = np.array([item[0] for item in data])
            # self.image = np.stack([np.array(data[i]["img"]) for i in range(len(data))], axis=0)

        try:
            # self.segm = np.stack([np.array(data[i]["segm"]) for i in range(len(data))], axis=0)
            if data[0][1] is None:
                self.segm = None
            else:
                self.segm = np.array([item[1] for item in data])
        except KeyError:
            self.segm = None
        self.history_positions = np.array([item[2] for item in data])
        # self.history_positions = np.stack([np.array(data[i]["agent_hist"]) for i in range(len(data))],
        #                                   axis=0)
        self.history_av = np.array([item[3] for item in data])
        # self.history_av = np.stack([np.array(data[i]["agent_hist_avail"]) for i in range(len(data))],
        #                            axis=0)

        try:
            self.forces = np.array([item[13] for item in data])
            # self.forces = np.stack([np.array(data[i]["forces"]) for i in range(len(data))],
            #                        axis=0)
        except KeyError:
            self.forces = None

        try:
            self.map_affine = np.array([item[14] for item in data])
            # self.map_affine = np.stack([np.array(data[i]["map_affine"]) for i in range(len(data))],
            #                             axis=0)
        except KeyError:
            self.map_affine = None

        try:
            self.cropping_points = np.array([item[15] for item in data])
            # self.cropping_points = np.stack([np.array(data[i]["cropping_points"]) for i in range(len(data))],
            #                             axis=0)
        except KeyError:
            self.cropping_points = None

        self.history_agents = NeighboursHistory([(data[i][6]) for i in range(len(data))]).get_history()
        # self.history_agents = NeighboursHistory([(data[i]["neighb"]) for i in range(len(data))]).get_history()

        self.history_agents_avail = np.array([item[7] for item in data], dtype=object)
        # self.history_agents_avail = [np.array(data[i]["neighb_avail"]) for i in range(len(data))]

        self.tgt = np.array([item[4] for item in data])
        # self.tgt = np.stack([np.array(data[i]["target"]) for i in range(len(data))], axis=0)

        self.neighb_tgt = None
        self.neighb_tgt_av = None
        if data[0][20] is not None:
            self.neighb_tgt = NeighboursHistory([(data[i][20]) for i in range(len(data))]).get_history()
            self.neighb_tgt_av = np.array([item[21] for item in data], dtype=object)

        self.tgt_avail = np.array([item[5] for item in data])
        # self.tgt_avail = np.stack([np.array(data[i]["target_avil"]) for i in range(len(data))], axis=0)

        self.raster_from_agent = np.array([item[8] for item in data])
        # self.raster_from_agent = np.stack([np.array(data[i]["raster_from_agent"]) for i in range(len(data))],
        #                                   axis=0)
        self.raster_from_world = np.array([item[9] for item in data])
        # self.raster_from_world = np.stack([np.array(data[i]["raster_from_world"]) for i in range(len(data))],
        #                                   axis=0)
        self.agent_from_world = np.array([item[11] for item in data])
        # self.agent_from_world = np.stack([np.array(data[i]["agent_from_world"]) for i in range(len(data))],
        #                                  axis=0)
        self.world_from_agent = np.array([item[10] for item in data])
        # self.world_from_agent = np.stack([np.array(data[i]["world_from_agent"]) for i in range(len(data))],
        #                                  axis=0)
        self.loc_im_to_glob = np.array([item[12] for item in data])
        # self.loc_im_to_glob = np.stack([np.array(data[i]["loc_im_to_glob"]) for i in range(len(data))],
        #                                axis=0)
        self.files = [dat.file for dat in data]
        self.world_to_image = None  # torch.stack([torch.tensor(data[i]["world_to_image"]) for i in range(len(data))], dim=0)
        self.centroid = None
        self.extent = None
        self.yaw = None
        self.speed = None
        # print(time.time() - st)

    def __str__(self) -> str:
        sp = "\t"
        sn = "\t\n"
        ff = "{:10.3f}"
        output = "UnifiedInterface batch state" + sn
        for batch in range(len(self.history_agents)):
            output += sn + sp + "batch: " + str(batch) + sn
            output += sp + "agent" + sn
            output += sp + "history_positions" + sp + "history_av" + sn
            # "image"+sp+\
            # "segm"+sp+\
            # "forces"+sp+\
            # "map_affine"+sp+\
            # "cropping_points"+sp+\
            # "tgt"+sp+\
            # "tgt_avail"+sp+\
            # "raster_from_agent"+sp+\
            # "raster_from_world"+sp+\
            # "agent_from_world"+sp+\
            # "world_from_agent"+sp+\
            # "loc_im_to_glob"+sp+\
            # "world_to_image"+sp+\
            # "centroid"+sp+\
            # "extent"+sp+\
            # "yaw"+sp+\
            # "speed"+sp+\
            # sn
            # output += table_legend
            for step in range(len(self.history_positions[batch])):
                output += sp + ff.format(self.history_positions[batch][step][0]) + sp + \
                          ff.format(self.history_positions[batch][step][1]) + sp + \
                          str(self.history_av[batch][step]) + sn
            for neigh in range(len(self.history_agents[batch])):
                output += sp + "neigh:" + str(neigh) + sn
                output += sp + "history_positions" + sp + "history_av" + sn
                for step in range(len(self.history_agents[batch][neigh])):
                    output += sp + ff.format(self.history_agents[batch][neigh][step][0]) + sp + \
                              ff.format(self.history_agents[batch][neigh][step][1]) + sp + \
                              str(self.history_agents_avail[batch][neigh][step]) + sn
        return output


def collate_wrapper(batch):
    return UnifiedInterface(batch)




if __name__ == "__main__":
    import cv2
    from utils import preprocess_data
    from torch.utils.data import DataLoader

    path = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    cfg["one_ped_one_traj"] = False
    cfg["raster_params"]["use_segm"] = True


    files = [
             "biwi_eth/biwi_eth.txt",
             "eth_hotel/eth_hotel.txt",
             "UCY/zara02/zara02.txt",
             "UCY/zara01/zara01.txt",
             "UCY/students01/students01.txt",
             "UCY/students03/students03.txt",
             ]
    dataset = DatasetFromTxt(path, files, cfg_=cfg)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0, collate_fn=collate_wrapper)  # , prefetch_factor=3)
    for i in range(100):

        data = next(iter(dataloader))
        print(i, data.files)
        img, segm = preprocess_data(data, cfg)
        plt.imshow(np.flip((img[0].reshape(3, img.shape[2], img.shape[3]).permute(1, 2, 0)).numpy(), axis=2))
        # plt.show()
        plt.savefig('foo'+str(i)+'.png')
        plt.clf()
        assert img.max() > 0
        img_n = img.numpy()
        transform_points(data.history_positions[0][:, :2], data.raster_from_agent[0])
        pass
    exit
    exit()
