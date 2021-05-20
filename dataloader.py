import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

import sys
# sys.path.insert(0, './')
try:
    from force_from_txt import Force_from_txt
    from trajenetloader import TrajnetLoader
    from utils import crop_image_crowds
    from utils import trajectory_orientation, rotate_image
    from utils import sdd_crop_and_rotate, transform_points
    from config import cfg
    from transformations import ChangeOrigin
except:
    # relative import
    from .force_from_txt import Force_from_txt
    from .trajenetloader import TrajnetLoader
    from .utils import crop_image_crowds
    from .utils import trajectory_orientation, rotate_image
    from .utils import sdd_crop_and_rotate, transform_points
    from .config import cfg
    from .transformations import ChangeOrigin

import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# idk = 0
# max_size_x = 0
# max_size_y = 0
# torch.multiprocessing.set_sharing_strategy('file_system')

class DataStructure:
    def __init__(self):
        self.agent_pose = None
        self.agent_pose_av = None

        self.neighb_poses = None
        self.neighb_poses_av = None

        self.img = None
        self.mask = None
        self.target = None
        self.target_av = None
        self.map_affine = None
        self.cropping_points = None
        self.raster_from_agent = None
        self.raster_from_world = None
        self.agent_from_world = None
        self.world_from_agent = None
        self.loc_im_to_glob = None
        self.world_to_image = None
        self.centroid = None
        self.extent = None
        self.yaw = None
        self.speed = None
        self.forces = None
        self.rot_mat = None
        self.pix_to_m = None
        self.loc_im_to_glob = None
        self.orig_pixels_hist = None
        self.orig_pixels_future = None

    def __getitem__(self, i):
        res = [self.img, self.mask, self.agent_pose, self.agent_pose_av, self.target, self.target_av,
               self.neighb_poses, self.agent_pose_av, self.raster_from_agent, self.raster_from_world, self.world_from_agent, self.agent_from_world,
               self.loc_im_to_glob, self.forces, self.rot_mat,
               self.cropping_points, self.orig_pixels_hist, self.orig_pixels_future, self.pix_to_m]
        return res[i]

class DatasetFromTxt(torch.utils.data.Dataset):

    def __init__(self, path, files, cfg=None, use_forces=False, forces_file=None):
        super(DatasetFromTxt, self).__init__()
        self.loader = TrajnetLoader(path, files, cfg)
        self.index = 0
        self.cfg = cfg
        self.files = files
        self.use_forces = use_forces
        if use_forces:
            self.force_from_txt = Force_from_txt(forces_file)

    def __len__(self):
        return self.loader.data_len

    def __getitem__(self, index: int):

        dataset_index = self.loader.get_subdataset_ts_separator(index)
        file = self.files[dataset_index]
        ped_id, ts = self.loader.get_pedId_and_timestamp_by_index(dataset_index, index)

        indexes = self.loader.get_all_agents_with_timestamp(dataset_index, ts)

        try:
            argsort_inexes = self.loader.argsort_inexes[file]
        except:
            argsort_inexes = None
        agents_history = self.loader.get_agent_history(dataset_index, ped_id, ts, indexes, argsort_inexes)
        agents_future = self.loader.get_agent_future(dataset_index, ped_id, ts, indexes, argsort_inexes)
        # agents_history = np.ones((10,8,4), dtype=np.float32)
        # agents_future = np.ones((10,12,4), dtype=np.float32)
        agent_history = agents_history[0]
        agent_future = agents_future[0]
        time_sorted_hist = np.array(agents_history[1:])  # sort_neigh_history(others_history)
        time_sorted_future = np.array(agents_future[1:])  # sort_neigh_future(others_future)

        forces = np.zeros(6)
        if self.use_forces:
            forces = self.force_from_txt.get(index)

        assert len(time_sorted_future) != 0
        assert len(time_sorted_hist) != 0

        agent_hist_avail = (agent_history[:, 0] != -1).astype(int)
        target_avil = (agent_future[:, 0] != -1).astype(int)
        hist_avail = (time_sorted_hist[:, :, 0] != -1).astype(int)
        neighb_future_avail = (time_sorted_future[:, :, 0] != -1).astype(int)
        img, mask, transform = self.loader.get_map(dataset_index, ped_id, ts)

        if not self.cfg["raster_params"]["use_map"]:
            if "zara" in file:
                pix_to_m = self.cfg["zara_h"]
                agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
            elif "students" in file:
                pix_to_m = self.cfg["student_h"]
                agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                    time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
            elif ("stanford" in file) or ("SDD" in file):
                dataset = file[file.index("/") + 1:file.index(".")]
                pix_to_m = np.eye(3) * self.cfg['SDD_scales'][dataset]["scale"]
                pix_to_m[2, 2] = 1
                pix_to_m = {"scale": pix_to_m}
            elif "biwi_eth" in file:
                pix_to_m = self.cfg['eth_univ_h']
                agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(time_sorted_hist[:, :, self.loader.coors_row],
                                                np.linalg.inv(pix_to_m["scale"]))
            elif "eth_hotel" in file:
                pix_to_m = self.cfg['eth_hotel_h']
                agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                    time_sorted_hist[:, :, self.loader.coors_row],
                    np.linalg.inv(pix_to_m["scale"]))

            elif "ros" in file:
                pix_to_m = self.cfg['ros_h']
                agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                    time_sorted_hist[:, :, self.loader.coors_row],
                    np.linalg.inv(pix_to_m["scale"]))




            agent_history = transform_points(agent_history[:, 2:], pix_to_m["scale"])
            agent_future = transform_points(agent_future[:, 2:], pix_to_m["scale"])
            to_localM_transform = np.eye(3)
            if self.cfg["raster_params"]["normalize"]:
                co_operator = ChangeOrigin(new_origin=agent_history[0][:2], rotation=np.eye(2))
                to_localM_transform = co_operator.transformation_matrix
            else:
                co_operator = ChangeOrigin(new_origin=np.array([0.,0]), rotation=np.eye(2))
                to_localM_transform = co_operator.transformation_matrix
            agent_history = transform_points(agent_history, to_localM_transform)
            agent_future = transform_points(agent_future, to_localM_transform)

            # time_sorted_future[:,:,2:] -= translation
            neigh_localM = transform_points(time_sorted_hist[:, :, self.loader.coors_row], co_operator.transformation_matrix@pix_to_m["scale"])
            raster_from_agent = np.linalg.inv(pix_to_m["scale"]) @ np.linalg.inv(co_operator.transformation_matrix)

            agent_history, neigh_localM = self.calc_speed_accel(agent_history, neigh_localM, agent_hist_avail,
                                                                hist_avail)
            res = [img, mask, agent_history, agent_hist_avail, agent_future, target_avil,
                   neigh_localM, hist_avail, raster_from_agent, np.eye(3), np.eye(3), np.eye(3),
                   np.eye(3), forces, None, None]
            return res
            # return {"img": None,
            #         "agent_hist": agent_history,
            #         "agent_hist_avail": agent_hist_avail,
            #         "target": agent_future,
            #         "target_avil": target_avil,
            #         "neighb": neigh_localM,
            #         "neighb_avail": hist_avail,
            #
            #         "raster_from_agent": np.eye(3),
            #         "raster_from_world": np.eye(3),  # raster_from_world,
            #         "world_from_agent": np.eye(3),  # world_from_agent,
            #         "agent_from_world": np.eye(3),  # agent_from_world
            #         "forces": np.eye(3),
            #         "loc_im_to_glob": np.eye(3),
            #         "file": file[file.index("/") + 1:]
            #         }

        # if map:
        if "UCY" in file or "eth" in file:
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
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                    time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))

            elif "students" in file:
                pix_to_image = self.cfg["students_pix_to_image_cfg"]
                pix_to_m = self.cfg["student_h"]
                agent_history[:, 2:] = transform_points(agent_history[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                agent_future[:, 2:] = transform_points(agent_future[:, 2:], np.linalg.inv(pix_to_m["scale"]))
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                    time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))

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
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                    time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
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
                time_sorted_hist[:, :, self.loader.coors_row] = transform_points(
                    time_sorted_hist[:, :, self.loader.coors_row], np.linalg.inv(pix_to_m["scale"]))
            else:
                raise NotImplemented

            if self.cfg["raster_params"]["normalize"]:
                img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
                border_width = int(abs(img_pil.size[0] - img_pil.size[1])*1.5)
                img_pil = ImageOps.expand(img_pil, (border_width, border_width))
                if mask is not None:
                    mask_pil = Image.fromarray(np.asarray(mask, dtype="uint8"))
                    mask_pil = ImageOps.expand(mask_pil, (border_width, border_width))
                if self.cfg["raster_params"]["draw_hist"]:

                    draw = ImageDraw.Draw(img_pil)
                    R = 5


                    for number, pose in enumerate(agent_history[:, 2:]):
                        if agent_hist_avail[number]:
                            rgb = (0, 0, 255//(number+1))
                            draw.ellipse((pix_to_image["coef_x"] * pose[0] - R + pix_to_image["displ_x"]+border_width,
                                          pix_to_image["coef_y"] * pose[1] - R + pix_to_image["displ_y"]+border_width,
                                          pix_to_image["coef_x"] * pose[0] + R + pix_to_image["displ_x"]+border_width,
                                          pix_to_image["coef_y"] * pose[1] + R + pix_to_image["displ_y"]+border_width
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
                                                       center=center + np.array([border_width, border_width]), mask=mask_pil)

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
                    pix_to_image_matrix)  @ img_to_rotated @ imBorder_to_image

                image_to_word = pix_to_m["scale"] @ np.linalg.inv(pix_to_image_matrix) @ imBorder_to_image

                center_img = pix_to_image_matrix @ (np.array([agent_history[0][2], agent_history[0][3], 1])) \
                             + np.array([border_width, border_width, 0])

                center_img_wb = pix_to_image_matrix @ (np.array([agent_history[0][2], agent_history[0][3], 1]))
                agent_history_m = pix_to_m["scale"] @ np.array([agent_history[0][2], agent_history[0][3], 1])
                error = Rimage_to_word @ (center_img) - agent_history_m

                crop_img, scale, mask_pil_crop = crop_image_crowds(img_pil, self.cfg["cropping_cfg"],
                                                    agent_center_img=center_img,
                                                    transform=Rimage_to_word, rot_mat=to_rot_mat, file=file, mask=mask_pil)

                world_to_agent_matrix = np.eye(3)
                world_to_agent_matrix[:, 2] = -agent_history_m

                pix_to_agent = world_to_agent_matrix @ pix_to_m["scale"]
                agent_hist_localM = transform_points(agent_history[:, self.loader.coors_row], pix_to_agent)
                target_localM = transform_points(agent_future[:, self.loader.coors_row], pix_to_agent)
                neigh_localM = transform_points(time_sorted_hist[:, :, self.loader.coors_row], pix_to_agent)
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

            else:
                agent_history, time_sorted_hist = self.calc_speed_accel(agent_history, time_sorted_hist,
                                                                        agent_hist_avail,
                                                                        hist_avail)  # TODO: check time_sorted_hist

                res = {"img": img,
                       "segm": mask,
                       "agent_hist": agent_history[:, self.loader.coors_row],
                       "agent_hist_avail": agent_hist_avail,
                       "target": agent_future[:, self.loader.coors_row],
                       "target_avil": target_avil,
                       "neighb": time_sorted_hist,
                       "neighb_avail": hist_avail,
                       "glob_to_local": None,
                       "image_to_local": None,
                       "raster_from_agent": None,
                       "raster_from_world": None,  # raster_from_world,
                       "world_from_agent": None,  # world_from_agent,
                       "agent_from_world": None,  # agent_from_world
                       "loc_im_to_glob": None,
                       "forces": forces
                       }
                return res

        if ("stanford" in file) or ("SDD" in file):
            if self.cfg["raster_params"]["normalize"]:
                # print(os.getpid(), 2, time.time())
                # self.border_datastes["SDD"] // self.resize_datastes["SDD"]
                # agent_future[:, 2:] += self.loader.border_datastes["SDD"] // self.loader.resize_datastes["SDD"]
                # agent_history[:, 2:] += self.loader.border_datastes["SDD"] // self.loader.resize_datastes["SDD"]
                # time_sorted_hist[:, :, 2:] += self.loader.border_datastes["SDD"] // self.loader.resize_datastes["SDD"]
                # time_sorted_future[:, :, 2:] += self.loader.border_datastes["SDD"] // self.loader.resize_datastes["SDD"]
                res = self.crop_and_normilize(agent_future, agent_hist_avail, agent_history, file, hist_avail, img,
                                              target_avil, time_sorted_hist, forces, mask,
                                              border_width=self.loader.border_datastes["SDD"], transform=transform)
                # print(os.getpid(), 3, time.time())

            else:

                dataset = file[:file.index("/")]
                file = file[file.index("/") + 1:file.index(".")]
                global_to_rast = transform+0
                pix_to_m = (np.eye(3) * self.cfg['SDD_scales'][file]["scale"])
                pix_to_m[2,2] = 1


                agent_center_local_pix = transform @ (np.append(agent_history[:, self.loader.coors_row][0],1))
                co_operator = ChangeOrigin(new_origin=agent_center_local_pix[:2], rotation=np.eye(2))
                raster_to_agent = pix_to_m @ co_operator.transformation_matrix
                agent_hist_localM = transform_points(agent_history[:, self.loader.coors_row],  raster_to_agent @ transform)
                target_localM = transform_points(agent_future[:, self.loader.coors_row],  raster_to_agent@ transform)
                neigh_localM = transform_points(time_sorted_hist[:, :, self.loader.coors_row],  raster_to_agent@ transform)
                neigh_futureM = transform_points(time_sorted_future[:, :, self.loader.coors_row],  raster_to_agent@ transform)
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
                #
                # res = {"img": np.copy(img),
                #        "agent_speed": speed,
                #        "agent_orientation": orientation,
                #        "agent_hist": agent_hist_localM,
                #        "agent_hist_avail": agent_hist_avail,
                #        "target": target_localM,
                #        "target_avil": target_avil,
                #        "neighb": neigh_localM,  # history -  shape[num_peds, time. coords]
                #        "neighb_avail": hist_avail,  # history availability -  shape[num_peds, time]
                #        "neigh_future": neigh_futureM,  # future -  shape[num_peds, time, coords]
                #        "neigh_future_avail": neighb_future_avail,  # future availability -  shape[num_peds, time]
                #        "neigh_speed": neigh_speed,  # history speeds -  shape[num_peds, time, speeds (vx, vy)]
                #        "neigh_orientation": neigh_orient,  # history orientation -  shape[num_peds, time, orient]
                #
                #        "raster_from_agent": raster_from_agent,
                #        "raster_from_world": raster_from_world,
                #        "world_from_agent": world_from_agent,
                #        "agent_from_world": agent_from_world,
                #        "forces": forces,
                #        }
            res.append(file)
            return res

    def calc_speed_accel(self, agent_history, neigh_localM, agent_history_av, neigh_localM_av):
        real_agent_history = 1*agent_history
        real_agent_history[np.sum(agent_history_av):] = real_agent_history[np.sum(agent_history_av) - 1]
        speed = -np.gradient(real_agent_history, axis=0)/0.4
        speed[:, 0][speed[:, 0] == 0] += 1e-6
        speed[agent_history_av == 0] *= 0
        acc = -np.gradient(speed, axis=0)
        acc[:, 0][acc[:, 0] == 0] += 1e-6
        acc[agent_history_av == 0] *= 0
        # TODO: unprecise [1, 0, 0] -> grad: [not0, not0, 0]
        # real_neigh_localM = 1 * neigh_localM
        # real_neigh_localM[:, np.sum(neigh_localM_av, axis=0):] = real_neigh_localM[:, np.sum(neigh_localM_av, axis=0) - 1]
        neigh_speed = -np.gradient(neigh_localM, axis=1) / 0.4
        neigh_speed[:, :, 0][neigh_speed[:, :, 0] == 0] += 1e-6
        neigh_speed[neigh_localM_av == 0] *= 0
        neigh_acc = -np.gradient(neigh_speed, axis=1)
        neigh_acc[:, :, 0][neigh_acc[:, :, 0] == 0] += 1e-6
        neigh_acc[neigh_localM_av == 0] *= 0
        agent_history = np.concatenate((agent_history, speed, acc), axis=1)
        neigh_localM = np.concatenate((neigh_localM, neigh_speed, neigh_acc), axis=2)
        return agent_history, neigh_localM


    def crop_and_normilize(self, agent_future, agent_hist_avail, agent_history, file, hist_avail, img, target_avil,
                           time_sorted_hist, forces, mask, border_width, transform):

        # rotate in a such way that last hisory points are horizontal (elft to right),
        # crop to spec in cfg area and resize
        #  calcultate transformation matrixes for pix to meters
        #  transform poses from pix to meters (local CS)

        file = file[file.index("/") + 1:file.index(".")]

        globPix_to_locraster = np.eye(3)
        tl_y, tl_x, br_y, br_x = (0, 0, 0, 0)
        map_to_local = None
        if img is None:

            scale = np.eye(3)
        else:
            img, globPix_to_locraster, scale,\
            mask, map_to_local,\
            (tl_y, tl_x, br_y, br_x), rotation_matrix = sdd_crop_and_rotate(img, agent_history[:, 2:],
                                                                   border_width=border_width,
                                                                   draw_traj=1,
                                                                   pix_to_m_cfg=self.cfg['SDD_scales'],
                                                                   cropping_cfg=self.cfg['cropping_cfg'], file=file,
                                                                   mask=mask,
                                                                   scale_factor=self.loader.resize_datastes["SDD"],
                                                                   transform=transform, neighb_hist=time_sorted_hist,
                                                                   neighb_hist_avail=hist_avail)

        import cv2

        ###
        agent_center_intermidiate = transform @ np.append(agent_history[:, 2:][0], 1)
        scale_ = self.cfg['SDD_scales'][file]["scale"]
        pix_to_meters = (np.eye(3) * scale_)
        radius = np.sqrt(2) * (round(
            max(np.linalg.norm((transform @ np.append(agent_history[:, 2:][0], 1))[:2] - np.array([tl_x, tl_y])),
                np.linalg.norm((transform @ np.append(agent_history[:, 2:][0], 1))[:2] - np.array([br_x, br_y])))))

        img_ = img[int(agent_center_intermidiate[1] - radius): int(agent_center_intermidiate[1] + radius),
              int(agent_center_intermidiate[0] - radius): int(agent_center_intermidiate[0] + radius)]
        if mask is not None:
            mask = mask[int(agent_center_intermidiate[1] - radius): int(agent_center_intermidiate[1] + radius),
                      int(agent_center_intermidiate[0] - radius): int(agent_center_intermidiate[0] + radius)]
        tl_x_intermidiate = tl_x - (agent_center_intermidiate[0] - radius)
        tl_y_intermidiate = tl_y - (agent_center_intermidiate[1] - radius)
        br_y_intermidiate = br_y - (agent_center_intermidiate[1] - radius)
        br_x_intermidiate = br_x - (agent_center_intermidiate[0] - radius)

        # TODO: depend at pic size in meters!
        size_constant = self.cfg["cropping_cfg"]["image_area_meters"][0]*0.08
        img_size = (int(size_constant*self.loader.img_size["SDD"][0]), int(size_constant*self.loader.img_size["SDD"][1]))
        new_im = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        new_im[:img_.shape[0], :img_.shape[1]] = np.copy(img_)
        img = new_im
        del img_
        co_operator = ChangeOrigin(new_origin=((agent_center_intermidiate[0] - radius, agent_center_intermidiate[1] - radius)), rotation=np.eye(2))
        rotation_matrix_cropped = co_operator.transformation_matrix @ rotation_matrix @ np.linalg.inv(co_operator.transformation_matrix)

        tl_co_operator = ChangeOrigin(
            new_origin=((tl_x, tl_y)),
            rotation=np.eye(2))

        intermidiate_to_locraster = scale @ tl_co_operator.transformation_matrix @ rotation_matrix
        # cv2.warpAffine(img, intermidiate_to_locraster[:2, :], (img.shape[1], img.shape[0]))

        if mask is not None:
            new_mask = np.zeros((img_size[0], img_size[1], 1), dtype=np.uint8)
            new_mask[:mask.shape[0], :mask.shape[1]] = np.copy(mask)
            mask = new_mask.astype(np.uint8)


        pix_to_m = np.eye(3) * self.cfg['SDD_scales'][file]["scale"]
        pix_to_m[2, 2] = 1
        initial_resize = transform.copy()
        initial_resize[:2, 2:] *= 0
        new_origin_operator = ChangeOrigin(new_origin=(intermidiate_to_locraster @ agent_center_intermidiate)[:2], rotation=np.eye(2))
        no_tm = new_origin_operator.transformation_matrix

        agent_from_raster = np.linalg.inv(initial_resize) @ np.linalg.inv(scale) @ pix_to_m @ no_tm

        global_agent_pix_pose_co_operator = ChangeOrigin(
            new_origin=(agent_history[:, self.loader.coors_row][0]),
            rotation=np.eye(2))
        # agent_from_glob_raster = np.linalg.inv(scale) @ pix_to_m @ new_origin_operator.transformation_matrix @ scale @ tl_co_operator.transformation_matrix @ rotation_matrix @ transform #@ np.append(agent_history[:, self.loader.coors_row][0], 1) @ np.append(agent_history[:, self.loader.coors_row][0], 1)
        agent_from_glob_raster = pix_to_m @ global_agent_pix_pose_co_operator.transformation_matrix @ np.linalg.inv(transform)  @ rotation_matrix @ transform
        # global_pix_from_raster = np.linalg.inv(transform) @ np.linalg.inv(globPix_to_locraster)
        world_from_raster = pix_to_m @ np.linalg.inv(intermidiate_to_locraster @ transform)

        raster_from_world = np.linalg.inv(world_from_raster)
        world_from_agent = world_from_raster @ np.linalg.inv(agent_from_raster)
        agent_from_world = np.linalg.inv(world_from_agent)

        agent_hist_localM = transform_points(agent_history[:, self.loader.coors_row],
                                        # pix_to_m @ intermidiate_to_locraster @ transform)
                                        #agent_from_raster   @ tl_co_operator.transformation_matrix @ rotation_matrix @ transform)
                                        agent_from_glob_raster)

        neigh_localM = transform_points(time_sorted_hist[:, :, self.loader.coors_row],
                                        # pix_to_m @ intermidiate_to_locraster @ transform) - agent_hist_M[0
                                        # np.linalg.inv(scale) @ agent_from_raster @ scale @ tl_co_operator.transformation_matrix @ rotation_matrix @ transform)
                                        agent_from_glob_raster)

        target_localM = transform_points(agent_future[:, self.loader.coors_row],
                                         # pix_to_m @ intermidiate_to_locraster @ transform) - agent_hist_M[0]
                                         # np.linalg.inv(scale) @ agent_from_raster @ scale @ tl_co_operator.transformation_matrix @ rotation_matrix @ transform)
                                         agent_from_glob_raster)

        agent_hist_localM, neigh_localM = self.calc_speed_accel(agent_hist_localM, neigh_localM, agent_hist_avail,
                                                            hist_avail)

        res = [img.astype(np.uint8), mask, agent_hist_localM, agent_hist_avail, target_localM, target_avil, neigh_localM,
               hist_avail, np.linalg.inv(agent_from_raster), raster_from_world, world_from_agent, agent_from_world,
               np.linalg.inv(pix_to_m) @ world_from_raster, transform_points(forces, rotation_matrix_cropped),
               rotation_matrix_cropped,
               np.array([tl_x_intermidiate, tl_y_intermidiate, br_x_intermidiate, br_y_intermidiate]),
               agent_history[:, self.loader.coors_row], agent_future[:, self.loader.coors_row], pix_to_m]
        return res

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
            poses.append([self.history_agents[batch][i] for i in range(1, len(self.history_agents[batch]))])
        return poses

import time

class UnifiedInterface:
    def __init__(self, data):
        # return None
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

        self.history_agents_avail = np.array([item[7] for item in data])
        # self.history_agents_avail = [np.array(data[i]["neighb_avail"]) for i in range(len(data))]

        self.tgt = np.array([item[4] for item in data])
        # self.tgt = np.stack([np.array(data[i]["target"]) for i in range(len(data))], axis=0)
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

        self.world_to_image = None  # torch.stack([torch.tensor(data[i]["world_to_image"]) for i in range(len(data))], dim=0)
        self.centroid = None
        self.extent = None
        self.yaw = None
        self.speed = None
        # print(time.time() - st)

def collate_wrapper(batch):
    return UnifiedInterface(batch)


if __name__ == "__main__":
    pass
    path_ = "/media/robot/hdd1/hdd_repos/pedestrian_forecasting_dataloader/data/train/"
    # cfg["raster_params"]["use_map"] = True
    # cfg["raster_params"]["normalize"] = False
    # # files = ["eth_hotel/eth_hotel.txt",
    # #         #"biwi_eth/biwi_eth.txt",
    # #          # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    # #          # "crowds/students003.txt",
    # #          # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    # #          # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    # #          # "stanford/coupa_3.txt",
    # #          # "stanford/deathCircle_0.txt",
    # #          ]
    # # #
    # # dataset = DatasetFromTxt(path_, files, cfg)
    # # img = dataset[0]["img"]
    # # # TODO: plot all trajes
    # # img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
    # # img_pil = img_pil.resize([int(720*2), int(576*2)])
    # # img_pil = img_pil.rotate(90, center=(img_pil.size[0]/2, img_pil.size[1]/2))
    # # bord = 600
    # # img_pil = ImageOps.expand(img_pil, (bord, bord))
    # # draw = ImageDraw.Draw(img_pil)
    # # R = 2
    # # pix_to_image = dataset.cfg["eth_hotel_pix_to_image_cfg"]
    # # pix_to_m = dataset.cfg['eth_hotel_h']
    # # for i in range(0, len(dataset), 2):
    # #     # ind = int(1000 * torch.rand(1).item())
    # #     data = dataset[i]
    # #     agent_history = data["agent_hist"][:, :2]
    # #     agent_history = transform_points(agent_history, np.linalg.inv(pix_to_m["scale"]))
    # #     # agent_history = np.flip(agent_history, axis=1)
    # #     for number, pose in enumerate(agent_history):
    # #         if data["agent_hist_avail"][number]:
    # #             draw.ellipse((pix_to_image["coef_x"] * pose[0] - R + pix_to_image["displ_x"] + bord,
    # #                           pix_to_image["coef_y"] * pose[1] - R + pix_to_image["displ_y"] + bord,
    # #                           pix_to_image["coef_x"] * pose[0] + R + pix_to_image["displ_x"] + bord,
    # #                           pix_to_image["coef_y"] * pose[1] + R + pix_to_image["displ_y"] + bord
    # #                           ), fill='blue', outline='blue')
    # #
    # # img_pil.show()
    #
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # for _ in range(100):
    #     ind = int(1000 * torch.rand(1).item())
    #     data = dataset[ind]
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # cfg["raster_params"]["use_map"] = True
    # cfg["raster_params"]["normalize"] = False
    # files = ["biwi_eth/biwi_eth.txt",
    #          # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #          # "crowds/students003.txt",
    #          # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #          # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #          # "stanford/coupa_3.txt",
    #          # "stanford/deathCircle_0.txt",
    #          ]
    # # dataset = DatasetFromTxt(path_, files, cfg)
    # # ind = int(1000 * torch.rand(1).item())
    # # data = dataset[ind]
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # cfg["raster_params"]["use_map"] = False
    # cfg["raster_params"]["normalize"] = True
    # files = ["biwi_eth/biwi_eth.txt",
    #          # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #          # "crowds/students003.txt",
    #          # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #          # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #          # "stanford/coupa_3.txt",
    #          # "stanford/deathCircle_0.txt",
    #          ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # cfg["raster_params"]["use_map"] = False
    # cfg["raster_params"]["normalize"] = False
    # files = ["biwi_eth/biwi_eth.txt",
    #          # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #          # "crowds/students003.txt",
    #          # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #          # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #          # "stanford/coupa_3.txt",
    #          # "stanford/deathCircle_0.txt",
    #          ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]
    #
    # files = [  # "biwi_eth/biwi_eth.txt",
    #     # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
    #     # "crowds/students003.txt",
    #     "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    #     # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
    #     # "stanford/coupa_3.txt",
    #     # "stanford/deathCircle_0.txt",
    # ]
    # dataset = DatasetFromTxt(path_, files, cfg)
    # ind = int(1000 * torch.rand(1).item())
    # data = dataset[ind]

    cfg["raster_params"]["use_map"] = True
    cfg["raster_params"]["normalize"] = True
    cfg["raster_params"]["use_segm"] = True

    files = [  # "biwi_eth/biwi_eth.txt",
        "SDD/bookstore_0.txt", "SDD/coupa_1.txt", "SDD/deathCircle_4.txt", "SDD/gates_1.txt"
        # "crowds/students001.txt",        "crowds/students003.txt",
        # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
        # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
        # "stanford/coupa_3.txt",
        # "stanford/deathCircle_0.txt",
    ]
    import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 2, tight_layout=True)

    dataset = DatasetFromTxt(path_, files, cfg)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=0, collate_fn=collate_wrapper)

    threshold = 200
    speeds_sdd = np.zeros(0)
    for i, data in enumerate((dataloader)):
        speed = np.linalg.norm(data.history_positions[:, :, 2:4], axis=2)[data.history_av == 1].reshape(-1)
        speeds_sdd = np.concatenate((speeds_sdd, speed[speed > 1e-6]))
        if i > threshold:
            print("crowds speed average:", np.mean(speeds_sdd))
            break
    n_bins = 500
    plt.hist(speeds_sdd, n_bins, alpha=0.5, label='speeds_sdd')
    # plt.hist(speeds_biwi_eth, n_bins, alpha=0.5, label='speeds_biwi_eth')
    # plt.hist(speeds_stanford, n_bins, alpha=0.5, label='speeds_stanford')
    # plt.hist(speeds_students, n_bins, alpha=0.5, label='speeds_students')
    # plt.hist(speeds_eth_hot, n_bins, alpha=0.5, label='speeds_eth_hot')

    plt.legend(loc='upper right')
    plt.savefig('sdd.png')
    plt.show()
    exit()
    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["normalize"] = False
    files = [
        # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt",
        "UCY/students01/students01.txt",
    ]
    dataset = DatasetFromTxt(path_, files, cfg)

    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=8, collate_fn=collate_wrapper)

    speeds_students = np.zeros(0)
    for i, data in enumerate((dataloader)):
        speed = np.linalg.norm(data.history_positions[:, :, 2:4], axis=2)[data.history_av == 1].reshape(-1)
        speeds_students = np.concatenate((speeds_students, speed[speed>1e-6]))
        if i > threshold:
            print("crowds wo_norm speed average:", np.mean(speeds_students))
            break

    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["normalize"] = False
    files = [
        # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt",
        "eth_hotel/eth_hotel.txt"
    ]
    dataset = DatasetFromTxt(path_, files, cfg)

    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=8, collate_fn=collate_wrapper)

    speeds_eth_hot = np.zeros(0)
    for i, data in enumerate((dataloader)):
        speed = np.linalg.norm(data.history_positions[:, :, 2:4], axis=2)[data.history_av == 1].reshape(-1)
        speeds_eth_hot = np.concatenate((speeds_eth_hot, speed[speed > 1e-6]))
        if i > threshold:
            print("eth_hotel wo_norm speed average:", np.mean(speeds_eth_hot))
            break

    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["normalize"] = False
    files = ["biwi_eth/biwi_eth.txt"]
    dataset = DatasetFromTxt(path_, files, cfg)
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=8, collate_fn=collate_wrapper)
    speeds_biwi_eth = np.zeros(0)
    for i, data in enumerate(dataloader):
        speed = np.linalg.norm(data.history_positions[:, :, 2:4], axis=2)[data.history_av == 1].reshape(-1)
        speeds_biwi_eth = np.concatenate((speeds_biwi_eth, speed[speed>1e-6]))
        if i > threshold:
            print("biwi_eth speed average:", np.mean(speeds_biwi_eth))
            break

    cfg["raster_params"]["use_map"] = False
    cfg["raster_params"]["normalize"] = False
    files = [  # "biwi_eth/biwi_eth.txt",
        # "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
        # "crowds/students003.txt",
        "SDD/bookstore_0.txt", "SDD/bookstore_1.txt",
        "SDD/bookstore_2.txt", "SDD/bookstore_3.txt",
        "SDD/coupa_3.txt",
        "SDD/deathCircle_0.txt",
    ]
    dataset = DatasetFromTxt(path_, files, cfg)

    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=8, collate_fn=collate_wrapper)

    speeds_stanford = np.zeros(0)
    for i, data in enumerate(tqdm(dataloader)):
        speed = np.linalg.norm(data.history_positions[:, :, 2:4], axis=2)[data.history_av == 1].reshape(-1)
        speeds_stanford = np.concatenate((speeds_stanford, speed[speed>1e-6]))
        if i > threshold:
            print("stanf speed average:", np.mean(speed))
            break
    n_bins = 20

    plt.hist(speeds_zara, n_bins, alpha=0.5, label='speeds_zara')
    plt.hist(speeds_biwi_eth, n_bins, alpha=0.5, label='speeds_biwi_eth')
    plt.hist(speeds_stanford, n_bins, alpha=0.5, label='speeds_stanford')
    plt.hist(speeds_students, n_bins, alpha=0.5, label='speeds_students')
    plt.hist(speeds_eth_hot, n_bins, alpha=0.5, label='speeds_eth_hot')

    plt.legend(loc='upper right')
    plt.savefig('foo.png')
    plt.show()

    # plt.savefig('foo.png')
    pass
