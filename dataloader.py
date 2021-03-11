import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

from force_from_txt import Force_from_txt
from trajenetloader import TrajnetLoader

from config import cfg

from utils import sdd_crop_and_rotate, transform_points

torch.multiprocessing.set_sharing_strategy('file_system')


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
        ped_id, ts = self.loader.get_pedId_and_timestamp_by_index(dataset_index, index)
        agent_history = self.loader.get_agent_history(dataset_index, ped_id, ts)
        agent_future = self.loader.get_agent_future(dataset_index, ped_id, ts)
        indexes = self.loader.get_all_agents_with_timestamp(dataset_index, ts)
        indexes = indexes[indexes != ped_id]
        others_history = []
        others_future = []
        forces = np.zeros(6)
        if self.use_forces:
            forces = self.force_from_txt.get(index)

        for index in indexes:
            history = self.loader.get_agent_history(dataset_index, index, ts)
            others_history.append(history)
            future = self.loader.get_agent_future(dataset_index, index, ts)
            others_future.append(future)

        if others_history == []:
            others_history = np.zeros((0, 9, 4))
            others_future = np.zeros((0, 12, 4))

        time_sorted_hist = np.array(others_history)  # sort_neigh_history(others_history)
        time_sorted_future = np.array(others_future)  # sort_neigh_future(others_future)
        agent_hist_avail = (agent_history[:, 0] != -1).astype(int)
        target_avil = (agent_future[:, 0] != -1).astype(int)
        hist_avail = (time_sorted_hist[:, :, 0] != -1).astype(int)
        neighb_future_avail = (time_sorted_future[:, :, 0] != -1).astype(int)
        img = self.loader.get_map(dataset_index, ped_id, ts)
        transf = None
        file = self.files[dataset_index]

        if "crowds" in file or "eth" in file:

            if not self.cfg["raster_params"]["use_map"]:
                if "zara" in file:
                    pix_to_image = self.cfg["zara2_pix_to_image_cfg"]
                    pix_to_m = self.cfg["zara_h"]
                elif "students" in file:
                    pix_to_image = self.cfg["students_pix_to_image_cfg"]
                    pix_to_m = self.cfg["student_h"]
                elif "eth" in file:
                    pix_to_image = self.cfg["students_pix_to_image_cfg"]
                    pix_to_m = {"scale": np.eye(3)}
                else:
                    raise NotImplemented
                agent_history = transform_points(agent_history[:, 2:], pix_to_m["scale"])
                translation = 1*agent_history[0]
                agent_history -= translation
                agent_future = transform_points(agent_future[:, 2:], pix_to_m["scale"])
                agent_future -= translation

                # time_sorted_future[:,:,2:] -= translation
                neigh_localM = transform_points(time_sorted_hist[:, :, self.loader.coors_row], pix_to_m["scale"])
                neigh_localM -= translation

                speed = -np.gradient(agent_history, axis=0)
                speed[:, 0][speed[:, 0] == 0] += 1e-6
                orientation = np.arctan((speed[:, 1]) / (speed[:, 0]))
                acc = -np.gradient(speed, axis=0)
                acc[:, 0][acc[:, 0] == 0] += 1e-6

                neigh_speed = -np.gradient(neigh_localM, axis=1)
                neigh_speed[:, :, 0][neigh_speed[:, :, 0] == 0] += 1e-6
                neigh_acc = -np.gradient(neigh_speed, axis=1)
                neigh_acc[:, :, 0][neigh_acc[:, :, 0] == 0] += 1e-6
                agent_history = np.concatenate((agent_history, speed, acc), axis=1)
                neigh_localM = np.concatenate((neigh_localM, neigh_speed, neigh_acc), axis=2)
                return {"img": None,
                       "agent_hist": agent_history,
                       "agent_hist_avail": agent_hist_avail,
                       "target": agent_future,
                       "target_avil": target_avil,
                       "neighb": neigh_localM,
                       "neighb_avail": hist_avail,

                       "raster_from_agent": np.eye(3),
                       "raster_from_world": np.eye(3), #raster_from_world,
                       "world_from_agent": np.eye(3), #world_from_agent,
                       "agent_from_world": np.eye(3), #agent_from_world
                       "forces": np.eye(3),
                       "loc_im_to_glob" : np.eye(3)
                       }

            elif self.cfg["raster_params"]["normalize"]:
                #draw
                img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
                pix_to_image = {}
                pix_to_m = np.eye(3)
                if "zara" in file:
                    pix_to_image = self.cfg["zara2_pix_to_image_cfg"]
                    pix_to_m = self.cfg["zara_h"]
                elif "students" in file:
                    pix_to_image = self.cfg["students_pix_to_image_cfg"]
                    pix_to_m = self.cfg["student_h"]
                else:
                    raise NotImplemented

                if self.cfg["raster_params"]["draw_hist"]:

                    draw = ImageDraw.Draw(img_pil)
                    R = 10
                    for pose in agent_history[:, 2:]:
                        if np.linalg.norm(pose - np.array([-1., -1.])) > 1e-6:
                            draw.ellipse((pix_to_image["coef_x"] * pose[0] - R + pix_to_image["displ_x"],
                                          pix_to_image["coef_y"] * pose[1] - R + pix_to_image["displ_y"],
                                          pix_to_image["coef_x"] * pose[0] + R + pix_to_image["displ_x"],
                                          pix_to_image["coef_y"] * pose[1] + R + pix_to_image["displ_y"]
                                          ), fill='blue', outline='blue')

                #expand

                from utils import crop_image_crowds
                from utils import trajectory_orientation, rotate_image
                import math
                border_width = 600
                img_pil = ImageOps.expand(img_pil, (border_width, border_width))
                # angle_deg
                angle_deg = -trajectory_orientation(agent_history[0][2:], agent_history[1][2:])
                if agent_hist_avail[1] == 0:
                    angle_deg = 0.
                angle_rad = angle_deg / 180 * math.pi

                # rotate_image -> transf matrix!
                center = np.array([pix_to_image["coef_x"] * agent_history[0, 2] + pix_to_image["displ_x"],
                                   pix_to_image["coef_y"] * agent_history[0, 3] + pix_to_image["displ_y"]])
                # angle_deg = 0

                # in imape CS
                img_pil, img_to_rotated = rotate_image(img_pil, angle_deg,
                                                       center=center + np.array([border_width, border_width]))

                # np.array([agent_history[0][2] + border_width, agent_history[0][3] + border_width, 1])

                pix_to_image_matrix = np.array([[pix_to_image["coef_x"], 0, pix_to_image["displ_x"]],
                                                [0, pix_to_image["coef_y"], pix_to_image["displ_y"]],
                                                [0, 0, 1]])
                #                                  img_to_rotated
                imBorder_to_image = np.eye(3)
                imBorder_to_image[:2, 2] = -border_width

                pix_to_pibxborder = np.eye(3)
                pix_to_pibxborder[:2, 0] = pix_to_image["coef_x"]*(-border_width)
                pix_to_pibxborder[:2, 1] = pix_to_image["coef_y"] * (-border_width)

                to_rot_mat = img_to_rotated.copy()
                to_rot_mat[:2, 2] = 0
                Rimage_to_word = pix_to_m["scale"] @ np.linalg.inv(pix_to_image_matrix) @ imBorder_to_image @ img_to_rotated
                image_to_word = pix_to_m["scale"] @ np.linalg.inv(pix_to_image_matrix) @ imBorder_to_image

                center_img = pix_to_image_matrix @  (np.array([agent_history[0][2], agent_history[0][3], 1])) \
                             + np.array([border_width, border_width, 0])

                center_img_wb = pix_to_image_matrix @ (np.array([agent_history[0][2], agent_history[0][3], 1]))
                agent_history_m = pix_to_m["scale"] @ np.array([agent_history[0][2], agent_history[0][3], 1])
                error = Rimage_to_word @ (center_img) - agent_history_m

                crop_img, scale = crop_image_crowds(img_pil, self.cfg["cropping_cfg"],
                                                    agent_center_img=center_img,
                                                    transform=Rimage_to_word, rot_mat=to_rot_mat, file=file)
                world_to_agent_matrix = np.eye(3)
                world_to_agent_matrix[:,2] = -agent_history_m

                pix_to_agent = world_to_agent_matrix @ pix_to_m["scale"]
                agent_hist_localM = transform_points(agent_history[:, self.loader.coors_row], pix_to_agent)
                target_localM = transform_points(agent_future[:, self.loader.coors_row], pix_to_agent)
                neigh_localM = transform_points(time_sorted_hist[:, :, self.loader.coors_row], pix_to_agent)
                # crop -> image, new_transf
                #
                agent_form_raster = pix_to_agent @ np.linalg.inv(pix_to_image_matrix)

                agent_form_raster_ = np.eye(3)
                agent_form_raster_[:2,2] = -np.array(self.cfg["cropping_cfg"]["agent_center"]) * np.array(self.cfg["cropping_cfg"]["image_shape"])
                agent_form_raster__ = np.eye(3)
                agent_form_raster__[:2, :2] = Rimage_to_word[:2, :2] / np.array(scale)
                agent_form_raster = agent_form_raster__ @ agent_form_raster_

                raster_from_agent = np.linalg.inv(agent_form_raster)
                raster_from_world = agent_form_raster @ world_to_agent_matrix
                res = {"img": np.copy(crop_img),
                       "agent_hist": agent_hist_localM,
                       "agent_hist_avail": agent_hist_avail,
                       "target": target_localM,
                       "target_avil": target_avil,
                       "neighb": neigh_localM,
                       "neighb_avail": hist_avail,

                       "raster_from_agent": raster_from_agent,
                       "raster_from_world": raster_from_world, #raster_from_world,
                       "world_from_agent": np.linalg.inv(world_to_agent_matrix), #world_from_agent,
                       "agent_from_world": world_to_agent_matrix, #agent_from_world
                       "forces": transform_points(forces, to_rot_mat),
                       }
                return res

            else:
                res = {"img": img,
                       "agent_hist": agent_history[:, self.loader.coors_row],
                       "agent_hist_avail": agent_hist_avail,
                       "target": agent_future[:, self.loader.coors_row],
                       "target_avil": target_avil,
                       "neighb": time_sorted_hist,
                       "neighb_avail": hist_avail,
                       "glob_to_local": None,
                       "image_to_local": None,
                       "forces": forces}
                return res


        if "stanford" in file:
            if self.cfg["raster_params"]["normalize"]:
                res = self.crop_and_normilize(agent_future, agent_hist_avail, agent_history, file, hist_avail, img,
                                              target_avil, time_sorted_hist, forces)

            else:

                #  calcultate transformation matrixes for pix to meters
                #  transform poses from pix to meters

                dataset = file[:file.index("/")]
                file = file[file.index("/") + 1:file.index(".")]
                raster_to_agent = np.eye(3) * self.cfg['SDD_scales'][file]["scale"]
                raster_to_agent[2, 2] = 1
                agent_hist_localM = transform_points(agent_history[:, self.loader.coors_row], raster_to_agent)
                target_localM = transform_points(agent_future[:, self.loader.coors_row], raster_to_agent)
                neigh_localM = transform_points(time_sorted_hist[:, :, self.loader.coors_row], raster_to_agent)
                neigh_futureM = transform_points(time_sorted_future[:, :, self.loader.coors_row], raster_to_agent)
                raster_from_agent = np.linalg.inv(raster_to_agent)
                raster_from_world = raster_from_agent
                world_from_agent = np.eye(3)
                agent_from_world = np.eye(3)
                speed = -np.gradient(agent_hist_localM, axis=0) / (12*self.loader.delta_t[dataset])  # TODO: 12?
                speed[:, 0][speed[:, 0] == 0] += 1e-6
                orientation = np.arctan((speed[:, 1]) / (speed[:, 0]))

                neigh_speed = -np.gradient(neigh_localM, axis=1) / (12 * self.loader.delta_t[dataset])  #
                neigh_speed[:, :, 0][neigh_speed[:,:, 0] == 0] += 1e-6
                neigh_orient = np.arctan((neigh_speed[:, :, 1]) / (neigh_speed[:, :, 0]))
                res = {"img": np.copy(img),
                       "agent_speed": speed,
                       "agent_orientation": orientation,
                       "agent_hist": agent_hist_localM,
                       "agent_hist_avail": agent_hist_avail,
                       "target": target_localM,
                       "target_avil": target_avil,
                       "neighb": neigh_localM,  # history -  shape[num_peds, time. coords]
                       "neighb_avail": hist_avail,  # history availability -  shape[num_peds, time]
                       "neigh_future": neigh_futureM,  # future -  shape[num_peds, time, coords]
                       "neigh_future_avail": neighb_future_avail,  # future availability -  shape[num_peds, time]
                       "neigh_speed": neigh_speed,  # history speeds -  shape[num_peds, time, speeds (vx, vy)]
                       "neigh_orientation": neigh_orient,  # history orientation -  shape[num_peds, time, orient]

                       "raster_from_agent": raster_from_agent,
                       "raster_from_world": raster_from_world,
                       "world_from_agent": world_from_agent,
                       "agent_from_world": agent_from_world,
                       "forces": forces,
                       }
            return res

    def crop_and_normilize(self, agent_future, agent_hist_avail, agent_history, file, hist_avail, img, target_avil,
                           time_sorted_hist, forces):

        # rotate in a such way that last hisory points are horizontal (elft to right),
        # crop to spec in cfg area and resize
        #  calcultate transformation matrixes for pix to meters
        #  transform poses from pix to meters (local CS)

        file = file[file.index("/") + 1:file.index(".")]

        globPix_to_locraster = np.eye(3)
        if img is None:
            pass
        else:
            img, globPix_to_locraster, scale = sdd_crop_and_rotate(img, agent_history[:, 2:],
                                                                   border_width=600,
                                                                   draw_traj=1,
                                                                   pix_to_m_cfg=self.cfg['SDD_scales'],
                                                                   cropping_cfg=self.cfg['cropping_cfg'], file=file)

        pix_to_m = np.eye(3) * self.cfg['SDD_scales'][file]["scale"]
        pix_to_m[:2, :2] = pix_to_m[:2, :2] @ np.array([[1/scale[0], 0], [0, 1/scale[1]]])  #np.linalg.inv(globPix_to_locraster[:2, :2])
        local_hist_pix = transform_points(agent_history[:, self.loader.coors_row], globPix_to_locraster)
        local_neigh_pix = transform_points(time_sorted_hist[:, :, self.loader.coors_row], globPix_to_locraster)
        local_target_pix = transform_points(agent_future[:, self.loader.coors_row], globPix_to_locraster)
        agent_from_glraster = np.eye(3)

        # TODO :generalize
        if img is not None:
            agent_from_glraster[:2, 2] = -np.array(self.cfg['cropping_cfg']["image_shape"]) * np.array(
                self.cfg['cropping_cfg']["agent_center"])
        else:
            agent_from_glraster[:2, 2] = -np.array([local_hist_pix[0, 0], local_hist_pix[0, 1]])

        agent_from_raster = pix_to_m @ agent_from_glraster
        agent_from_raster[2, 2] = 1
        raster_from_agent = np.eye(3) / self.cfg['SDD_scales'][file]["scale"]
        raster_from_agent[2, 2] = 1

        rotation_matrix = np.eye(3)
        if img is not None:
            raster_from_agent[:2, 2] = np.array(self.cfg['cropping_cfg']["image_shape"]) * np.array(
                self.cfg['cropping_cfg']["agent_center"])
        else:
            from utils import trajectory_orientation
            angle = trajectory_orientation(local_hist_pix[0],local_hist_pix[1])
            if agent_hist_avail[1] != 1:
                angle = 0
            rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0],
                                        [-np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
            agent_from_raster = rotation_matrix @ agent_from_raster
            raster_from_agent = np.linalg.inv(agent_from_raster)

        agent_from_loraster = pix_to_m @ np.linalg.inv(raster_from_agent)
        agent_hist_localM = transform_points(local_hist_pix, agent_from_raster)
        neigh_localM = transform_points(local_neigh_pix, agent_from_raster)
        target_localM = transform_points(local_target_pix, agent_from_raster)
        world_from_agent = globPix_to_locraster @ raster_from_agent
        agent_from_world = pix_to_m @ globPix_to_locraster
        raster_from_world = raster_from_agent @ agent_from_world
        res = {"img": np.copy(img),
               "agent_hist": agent_hist_localM,
               "agent_hist_avail": agent_hist_avail,
               "target": target_localM,
               "target_avil": target_avil,
               "neighb": neigh_localM,
               "neighb_avail": hist_avail,

               "raster_from_agent": np.linalg.inv(agent_from_raster),
               "raster_from_world": raster_from_world,
               "world_from_agent": world_from_agent,
               "agent_from_world": agent_from_world,
               "loc_im_to_glob": np.linalg.inv(globPix_to_locraster),
               "forces": transform_points(forces, rotation_matrix)}
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




class UnifiedInterface:
    def __init__(self, data):
        if data[0]["img"] is None:
            self.image = None
        else:
            self.image = np.stack([np.array(data[i]["img"]) for i in range(len(data))], axis=0)

        self.history_positions = np.stack([np.array(data[i]["agent_hist"]) for i in range(len(data))],
                                          axis=0)
        self.history_av = np.stack([np.array(data[i]["agent_hist_avail"]) for i in range(len(data))],
                                   axis=0)

        try:
            self.forces = np.stack([np.array(data[i]["forces"]) for i in range(len(data))],
                                   axis=0)
        except:
            self.forces = None
        self.history_agents = NeighboursHistory([(data[i]["neighb"]) for i in range(len(data))]).get_history()
        self.history_agents_avail = [np.array(data[i]["neighb_avail"]) for i in range(len(data))]
        self.tgt = np.stack([np.array(data[i]["target"]) for i in range(len(data))], axis=0)
        self.tgt_avail = np.stack([np.array(data[i]["target_avil"]) for i in range(len(data))], axis=0)
        self.world_to_image = None #torch.stack([torch.tensor(data[i]["world_to_image"]) for i in range(len(data))], dim=0)
        self.raster_from_agent = np.stack([np.array(data[i]["raster_from_agent"]) for i in range(len(data))],
                                            axis=0)
        self.raster_from_world = np.stack([np.array(data[i]["raster_from_world"]) for i in range(len(data))],
                                             axis=0)
        self.agent_from_world = np.stack([np.array(data[i]["agent_from_world"]) for i in range(len(data))],
                                            axis=0)
        self.world_from_agent = np.stack([np.array(data[i]["world_from_agent"]) for i in range(len(data))],
                                            axis=0)
        self.loc_im_to_glob = np.stack([np.array(data[i]["loc_im_to_glob"]) for i in range(len(data))],
                                            axis=0)
        self.centroid = None
        self.extent = None
        self.yaw = None
        self.speed = None

def collate_wrapper(batch):
    return UnifiedInterface(batch)




if __name__ == "__main__":
    pass
    path_ = "data/train/"
    files = [ "biwi_eth/biwi_eth.txt", "crowds/crowds_zara02.txt", "crowds/crowds_zara03.txt", "crowds/students001.txt",
         "crowds/students003.txt",
        # "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
        # "stanford/bookstore_2.txt", "stanford/bookstore_3.txt",
        # "stanford/coupa_3.txt",
        # "stanford/deathCircle_0.txt",
        ]
    dataset = DatasetFromTxt(path_, files, cfg)
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=8, collate_fn=collate_wrapper)

    from tqdm import tqdm
    for i, data in enumerate(tqdm(dataloader)):
        pass
        if i > 200:
            break
        continue
