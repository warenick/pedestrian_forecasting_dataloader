import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import re

try:
    from transformations import Resize, AddBorder
except:
    from .transformations import Resize, AddBorder


class TrajnetLoader:

    def __init__(self, path, data_files, cfg):
        self.stochastic = False
        self.data_files = data_files
        self.path = path
        self.dataset_index = []
        self.index_row = 1
        self.argsort_inexes = {}

        # 1/fps
        self.delta_t = {"biwi": 1 / 25,
                        "biwi_eth": 1 / 15,
                        "eth_hotel": 1 / 25,
                        "UCY": 1 / 25,
                        "stanford": 1 / 30,
                        "SDD": 1 / 30,
                        "ros": 1 / 1000.,
                        }

        #  delta frames between sequential data in text file in
        self.delta_timestamp = {"stanford": 12,
                                "SDD": 12,
                                # "crowds": 10,
                                "UCY": 10,
                                "biwi_eth": 6,
                                "eth_hotel": 10,
                                "ros": 400,  # delta MS
                                }

        self.ts_row = 0  # timestamp row
        self.coors_row = [2, 3]
        self.history_len = 3.2  # sec
        self.pred_len = 4.8  # sec

        self.data = {}
        self.homography = {}
        self.data_len = 0
        self.sub_data_len = [0]
        self.cfg = cfg
        self.resize_datastes = {"SDD": 10, "ETH/UCY": 6}
        self.border_datastes = {"SDD": 2000, "ETH/UCY": 1600}
        self.img_size = {"SDD": (int(2100 / self.resize_datastes["SDD"]), int(2000 / self.resize_datastes["SDD"])),
                         "ETH/UCY": (
                             # int(720 / self.resize_datastes["ETH/UCY"]), int(720 / self.resize_datastes["ETH/UCY"]))}
                             int(1520*1.6 / self.resize_datastes["ETH/UCY"]), int(1520*1.6 / self.resize_datastes["ETH/UCY"]))}
        self.loaded_imgs = {}
        self.img_transf = {}
        self.unique_ped_ids = {}
        self.unique_ped_ids_len = 0
        self.sub_data_len_by_ped_ids = [0]

        for file in (data_files):
            dataset = "ETH/UCY"
            if path[-1] != "/":
                path += "/"
            name = path + file

            ## IF SDD: load from npy (faster), skip types (load only ped id , ts, bboxes, flag of being visible)
            if "SDD" in file:
                dataset = "SDD"
                try:
                    new_name = name[:name.index(".")] + ".npy"
                    self.data[file] = np.load(new_name).astype(np.float32)
                except:
                    self.data[file] = np.loadtxt(path + "/" + file, delimiter=' ',
                                                 usecols=[0, 1, 2, 3, 4, 5, 6]).astype(np.float32)
                    self.data[file] = self.data[file][self.data[file][:, 6] == 0]

                # filter each 12th data point (to be 0.4 fps)
                self.data[file] = self.data[file][
                    (self.data[file][:, 5] + (self.data[file][:, 5].min() % 12)) % 12 == 0]

                # rearrange: ts, ped_id, bboxes
                self.data[file] = self.data[file][:, (5, 0, 1, 2, 3, 4)]

                # from bboxes to center points
                self.data[file][:, 2] = (self.data[file][:, 2] + self.data[file][:, 4]) / 2
                self.data[file][:, 3] = (self.data[file][:, 3] + self.data[file][:, 5]) / 2
                self.data[file] = self.data[file][:, :4]

            ## IF NOT SDD
            if "SDD" not in file:
                self.data[file] = np.genfromtxt(path + "/" + file, delimiter='').astype(np.float64)
                self.argsort_inexes[file] = None
                homography_path = path + "/" + file[:[m.start() for m in re.finditer(r"/", file)][-1]] + "/H.txt"
                # if "UCY":
                #     homography_path = path + "/" + file[:file.index("/")] + "/H.txt"

                self.homography[file] = np.genfromtxt(homography_path, delimiter='').astype(np.float64)
                if "students" in file:
                    self.homography[file] = np.array([[0.00000000e+00, 2.10465100e-2, 1.99500000e-05],
                 [-2.38659800e-02, 0.00000000e+00, 1.40331962e+01],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                if 'ros' in file:
                    # self.argsort_inexes[file] = np.argsort(self.data[file][:,self.index_row])
                    self.data[file] = self.data[file][np.argsort(self.data[file][:, self.ts_row])]
                    self.data[file] = self.data[file][np.argsort(self.data[file][:, self.index_row], kind='mergesort')]

            ### save ped ids
            if cfg["one_ped_one_traj"]:
                self.unique_ped_ids[file] = np.unique(self.data[file][:, 1])
                self.unique_ped_ids_len += len(self.unique_ped_ids[file])
                self.sub_data_len_by_ped_ids.append(self.sub_data_len_by_ped_ids[-1] + len(self.unique_ped_ids[file]))

            ## load images
            if cfg["raster_params"]["use_map"]:
                img_format = ".png"
                if dataset == "SDD":
                    img_format = ".jpg"
                img = cv2.imread(name[:name.index(".")] + img_format).astype(np.uint8)
                mask = np.ones_like(img)[:, :, 0]

                ## load segmentations
                if cfg["raster_params"]["use_segm"]:
                    mask = np.load(name[:name.index(".")] + "_s.npy").astype(np.uint8)

                # initial resize of image (for speed-up)
                transf = np.eye(3)
                resize_transf = Resize((1 / self.resize_datastes[dataset], 1 / self.resize_datastes[dataset]))
                img = resize_transf.apply(img)
                mask = resize_transf.apply(mask)
                transf = resize_transf.transformation_matrix @ transf

                # in order to keep all images at the same "buffer" need to use unified img size
                unified_img = np.zeros((self.img_size[dataset][0], self.img_size[dataset][1], 3), dtype=np.int16)
                unified_img[:img.shape[0], :img.shape[1], :] = img
                unified_mask = np.zeros((self.img_size[dataset][0], self.img_size[dataset][1], 1), dtype=np.int16)
                unified_mask[:mask.shape[0], :mask.shape[1], 0] = mask
                img = unified_img
                mask = unified_mask

                # apply border  transformations
                border = self.border_datastes[dataset] // self.resize_datastes[dataset]
                transf_border = AddBorder(border)
                img = transf_border.apply(img)
                mask = transf_border.apply(mask)
                transf = transf_border.transformation_matrix @ transf

                ## save to buffer
                self.loaded_imgs[name[:name.index(".")] + img_format] = img
                self.img_transf[name[:name.index(".")] + img_format] = transf
                self.loaded_imgs[name[:name.index(".")] + "_s.npy"] = mask

            self.data_len += len(self.data[file])
            self.sub_data_len.append(self.data_len)

    def get_all_agents_with_timestamp(self, dataset_ind: int, timestamp: float) -> np.array:

        """
         :param dataset_ind: index of file
         :param timestamp:  timestamp from txt file
         :return: numpy array of agents IDs that exist on scene with specified dataset_ind&timestamp
        """

        file = self.data_files[dataset_ind]
        data = self.data[file]
        return data[data[:, self.ts_row] == timestamp][:, self.index_row]

    def get_agent_history(self, dataset_ind: int, ped_id: int, timestamp: float, neighb_indexes,
                          argsort_inexes=None) -> np.array:

        """
         :param dataset_ind: index of file
         :param ped_id: ID of agent
         :param timestamp:  timestamp from txt file. Observed history is [timespemp - history_len, timespemp]
         :param neighb_indexes: IDs of neghbors
         :param argsort_inexes: Optional array of integer indices that sort array a into ascending order. They are typically the result of argsort.
         :return: observed trajectory of specified agent np.array shape(self.history_len+1,4).
        """
        neighb_indexes = neighb_indexes[neighb_indexes != ped_id]
        # import time
        # st = time.time()
        file = self.data_files[dataset_ind]

        data = self.data[file]
        if "eth" in file or "UCY" in file:
            data = data[np.argsort(data[:, 1], axis=0, kind="mergesort")]
        # TODO: Refactor start_ts, timecoef, self.delta_timestamp
        start_ts = timestamp - (
                self.history_len / self.delta_t[file[0:file.index("/")]])  # *self.delta_t[file[0:file.index("/")]])

        timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
        hist_horizon_steps = int(self.history_len / timecoef)
        full_hist_scene = (np.zeros((hist_horizon_steps, 4)) - 1)[np.newaxis]

        start_stop_ped_ind = [ped_id, ped_id + 1]
        for ind in neighb_indexes:
            start_stop_ped_ind.append(ind)
            start_stop_ped_ind.append(ind + 1)

        ind = np.searchsorted(data[:, self.index_row], start_stop_ped_ind, sorter=argsort_inexes)

        indexes_start = ind[::2]
        # ind_stop = np.searchsorted(data[:, self.index_row], ped_id + 1)
        indexes_stop = ind[1::2]

        for ind_start, ind_stop in zip(indexes_start, indexes_stop):
            data_ = data[ind_start:ind_stop, :]  # filter by index

            data_ = data_[(data_[:, self.ts_row] > start_ts)]  # filter by timestamp
            data_ = data_[data_[:, self.ts_row] <= timestamp]

            if "eth" in file:
                data_ = data_[:, (0, 1, 2, 4)]
            if ("zara01" in file) or ("zara02" in file):
                data_ = data_[:, (0, 1, 2, 4)]

            timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
            out = np.zeros((int(self.history_len / timecoef), 4)) - 1
            out[0:len(data_), :] = np.flip(data_[-out.shape[0]:], axis=0)
            full_hist_scene = np.concatenate((full_hist_scene, out[np.newaxis]), axis=0)

        return full_hist_scene[1:]

    def get_agent_future(self, dataset_ind: int, ped_id: int, timestamp: float, neighb_indexes,
                         argsort_inexes=None) -> np.array:
        """
         :param dataset_ind: index of file
         :param ped_id: ID of agent
         :param timestamp:  timestamp from txt file. Target future is [timespemp, timespemp + pred_len]
         :param neighb_indexes: IDs of neghbors
         :param argsort_inexes: Optional array of integer indices that sort array a into ascending order. They are typically the result of argsort.
         :return: future(target) trajectory of specified agent np.array shape(self.history_len+1,4).
        """
        neighb_indexes = neighb_indexes[neighb_indexes != ped_id]
        file = self.data_files[dataset_ind]
        end_timestamp = timestamp + (
                self.pred_len / self.delta_t[file[0:file.index("/")]])  # self.delta_t[file[0:file.index("/")]])
        data = self.data[file]
        if "eth" in file or "UCY" in file:
            data = data[np.argsort(data[:, 1], axis=0, kind="mergesort")]
        timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
        full_future_scene = (np.zeros((round(self.pred_len / timecoef), 4)) - 1)[np.newaxis]
        start_stop_ped_ind = [ped_id, ped_id + 1]
        for ind in neighb_indexes:
            start_stop_ped_ind.append(ind)
            start_stop_ped_ind.append(ind + 1)
        ind = np.searchsorted(data[:, self.index_row], start_stop_ped_ind, sorter=argsort_inexes)
        indexes_start = ind[::2]
        indexes_stop = ind[1::2]

        # ind_start = np.searchsorted(data[:, self.index_row], ped_id)
        # ind_stop = np.searchsorted(data[:, self.index_row], ped_id + 1)
        # data = data[ind_start:ind_stop, :]  # filter by index
        for ind_start, ind_stop in zip(indexes_start, indexes_stop):
            data_ = data[ind_start:ind_stop, :]  # filter by index

            data_ = data_[(data_[:, self.ts_row] <= end_timestamp)]  # filter by timestamp
            data_ = data_[data_[:, self.ts_row] > timestamp]
            if "eth" in file:
                data_ = data_[:, (0, 1, 2, 4)]
            if ("zara01" in file) or ("zara02" in file):
                data_ = data_[:, (0, 1, 2, 4)]
            # if ("students03" in file):
            #     data = data[:, (0, 1, 2, 4)]

            out = np.zeros((int(round(self.pred_len / timecoef)), 4)) - 1
            out[0:len(data_), :] = np.array(data_)
            full_future_scene = np.concatenate((full_future_scene, out[np.newaxis]), axis=0)
        return full_future_scene[1:]

    def get_subdataset_ts_separator(self, index: int) -> int:
        """
            :param index: index of data (row) in whole(combined) dataset
            :return: index of dataset file
        """
        if self.cfg["one_ped_one_traj"]:
            dataset_ind, = np.where(np.array(self.sub_data_len_by_ped_ids) <= index)
            return dataset_ind[-1]

        dataset_ind, = np.where(np.array(self.sub_data_len) <= index)
        return dataset_ind[-1]

    def get_pedId_and_timestamp_by_index(self, dataset_ind: int, index: int) -> (int, int):
        """
        :param dataset_ind: index of dataset file (from get_subdataset_ts_separator)
        :param index: index of data (row) in whole(combined) dataset
        :return:
            ped_id:  pedestrinan id
            ts: timestamp

        """
        file = self.data_files[dataset_ind]
        data = self.data[file]
        # when shorter dataset with one trajectory for each ped use self.sub_data_len_by_ped_ids as separator
        if self.cfg["one_ped_one_traj"]:
            index = index - self.sub_data_len_by_ped_ids[dataset_ind]
            ped_id = self.unique_ped_ids[file][index]
            num_of_observations = self.data[file][self.data[file][:, 1] == ped_id].shape[0]

            # start_from_index = num_of_observations // 20 * 8 #min(num_of_observations, 8)
            if num_of_observations > 2:
                if self.stochastic:
                    start_from_index = np.random.randint(1, num_of_observations - 1)
                if not self.stochastic:
                    start_from_index = num_of_observations // 2
            else:
                start_from_index = 0
            ts = self.data[file][self.data[file][:, 1] == ped_id][start_from_index, 0]
            return ped_id, ts

        index = index - self.sub_data_len[dataset_ind]
        ts = data[index, self.ts_row]
        ped_id = data[index, self.index_row]
        return ped_id, ts

    def get_map(self, dataset_ind: int, ped_id: int, timestamp: float):
        # import cv2
        """
         :param dataset_ind: index of file
         :param ped_id: ID of agent
         :param timestamp:  timestamp from txt file. Target future is [timespemp, timespemp + pred_len]
         :return:
        """

        if not self.cfg["raster_params"]["use_map"]:
            return None, None, np.eye(3)

        txt_file = self.data_files[dataset_ind]

        if self.cfg["raster_params"]["use_segm"]:
            segm_file = self.path + txt_file[0:txt_file.index(".")] + "_s.npy"

        if "SDD" in txt_file:
            img_type = ".jpg"
        else:
            img_type = ".png"

        img_file = self.path + txt_file[0:txt_file.index(".")] + img_type

        if img_file not in self.loaded_imgs.keys():
            raise Exception("run time loading!")
            # img = cv2.imread(img_file).astype(np.int16)
            # if "SDD" in img_file:
            #     img = cv2.resize(img, (img.shape[1]//5, img.shape[0]//5), interpolation=0)
            # self.loaded_imgs[img_file] = img

        segm = None
        if self.cfg["raster_params"]["use_segm"]:
            if segm_file not in self.loaded_imgs.keys():
                raise Exception("run time loading!")
                # segm = np.load(segm_file)
                # if "SDD" in img_file:
                #     segm = cv2.resize(segm, (segm.shape[1] // 5, segm.shape[0] // 5), interpolation=0)
                # self.loaded_imgs[segm_file] = segm

        if self.cfg["raster_params"]["use_segm"]:
            return np.copy(self.loaded_imgs[img_file]), np.copy(self.loaded_imgs[segm_file]), self.img_transf[img_file]

        return np.copy(self.loaded_imgs[img_file]), None, self.img_transf[img_file]
