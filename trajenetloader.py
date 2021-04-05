import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

class TrajnetLoader:

    def __init__(self, path, data_files, cfg):

        self.data_files = data_files
        self.path = path
        self.index_row = 1
        self.delta_t = {"biwi": 1 / 25,
                        "biwi_eth": 1 / 15,
                        "eth_hotel": 1 / 25,
                        "UCY": 1 / 25,
                        "stanford": 1 / 30,
                        "SDD": 1 / 30,
                        # "crowds": 1 / 25,
                        # "students": 1 / 25,
                        }  # 10*msec

        self.delta_timestamp = {"stanford": 12,
                                "SDD": 12,
                                # "crowds": 10,
                                "UCY": 10,
                                "biwi_eth": 6,
                                "eth_hotel": 10,
                                }
        self.ts_row = 0  # timestamp row
        self.coors_row = [2, 3]
        self.history_len = 3.2  # sec
        self.pred_len = 4.8  # sec

        self.data = {}
        self.data_len = 0
        self.sub_data_len = [0]
        self.cfg = cfg
        self.resize_datastes = {"SDD": 2}
        self.border_datastes = {"SDD": 600}
        self.img_size = {"SDD": (1200, 1200)}
        self.loaded_imgs = {}
        print("loading files")
        for file in tqdm(data_files):

            if "SDD" in file:
                try:
                    if path[-1] != "/":
                        path+="/"
                    name = path + file
                    new_name = name[:name.index(".")] + ".npy"
                    self.data[file] = np.load(new_name).astype(np.float32)

                except:
                    self.data[file] = np.loadtxt(path + "/" + file, delimiter=' ', usecols=[0, 1, 2, 3, 4, 5]).astype(np.float32)

                self.data[file] = self.data[file][
                    (self.data[file][:, 5] + (self.data[file][:, 5].min() % 12)) % 12 == 0]
                self.data[file] = self.data[file][:, (5, 0, 1, 2, 3, 4)]
                self.data[file][:, 2] = (self.data[file][:, 2] + self.data[file][:, 4]) / 2
                self.data[file][:, 3] = (self.data[file][:, 3] + self.data[file][:, 5]) / 2


                self.data[file] = self.data[file][:, :4]

                if cfg["raster_params"]["use_map"]:
                    img = cv2.imread(name[:name.index(".")] + ".jpg").astype(np.uint8)
                    img = cv2.resize(img, (img.shape[1] // self.resize_datastes["SDD"], img.shape[0] // self.resize_datastes["SDD"]), interpolation=0)

                    unified_img = np.zeros((self.img_size["SDD"][0], self.img_size["SDD"][1], 3), dtype=np.int16)
                    unified_img[:img.shape[0], :img.shape[1], :] = img

                    border = self.border_datastes["SDD"] // self.resize_datastes["SDD"]
                    sh = unified_img.shape
                    img_b = np.zeros((sh[0] + 2 * border, sh[1] + 2 * border, sh[2]), dtype=np.uint8)
                    img_b[border:-border, border:-border] = unified_img

                    self.loaded_imgs[name[:name.index(".")] + ".jpg"] = img_b

                if cfg["raster_params"]["use_segm"]:
                    img = np.load(name[:name.index(".")] + "_s.npy").astype(np.uint8)
                    img = cv2.resize(img, (img.shape[1]  // self.resize_datastes["SDD"], img.shape[0]  // self.resize_datastes["SDD"]), interpolation=0)
                    unified_img = np.zeros((self.img_size["SDD"][0], self.img_size["SDD"][1]), dtype=np.uint8)
                    unified_img[:img.shape[0], :img.shape[1]] = img

                    border = self.border_datastes["SDD"] // self.resize_datastes["SDD"]
                    sh = unified_img.shape
                    img_b = np.zeros((sh[0] + 2 * border, sh[1] + 2 * border), dtype=np.uint8)
                    img_b[border:-border, border:-border] = unified_img
                    self.loaded_imgs[name[:name.index(".")] + "_s.npy"] = img_b
            else:
                self.data[file] = np.genfromtxt(path + "/" + file, delimiter='').astype(np.float32)
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

    def get_agent_history(self, dataset_ind: int, ped_id: int, timestamp: float, neighb_indexes) -> np.array:

        """
         :param dataset_ind: index of file
         :param ped_id: ID of agent
         :param timestamp:  timestamp from txt file. Observed history is [timespemp - history_len, timespemp]
         :return: observed trajectory of specified agent np.array shape(self.history_len+1,4).
        """
        neighb_indexes = neighb_indexes[neighb_indexes != ped_id]
        # import time
        # st = time.time()
        file = self.data_files[dataset_ind]

        data = self.data[file]

        start_ts = timestamp - (
                self.history_len / self.delta_t[file[0:file.index("/")]])  # *self.delta_t[file[0:file.index("/")]])
        #         if self.cfg["raster_params"]["use_map"] == True:
        # print(st - time.time())
        # st = time.time()
        timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
        full_hist_scene = (np.zeros((int(self.history_len / timecoef), 4)) - 1)[np.newaxis]
        start_stop_ped_ind = [ped_id, ped_id+1]
        for ind in neighb_indexes:
            start_stop_ped_ind.append(ind)
            start_stop_ped_ind.append(ind+1)

        ind = np.searchsorted(data[:, self.index_row], start_stop_ped_ind)
        # ind_start = np.searchsorted(data[:, self.index_row], start_stop_ped_ind)
        indexes_start = ind[::2]
        # ind_stop = np.searchsorted(data[:, self.index_row], ped_id + 1)
        indexes_stop = ind[1::2]

        for ind_start, ind_stop in zip(indexes_start, indexes_stop):
            data_ = data[ind_start:ind_stop, :]  # filter by index



            data_ = data_[(data_[:, self.ts_row] > start_ts)]  # filter by timestamp
            data_ = data_[data_[:, self.ts_row] <= timestamp]
            # data = np.argwhere(data[:, self.ts_row] > start_ts)
            # data = np.argwhere(data[:, self.index_row] == ped_id)
            # data = np.argwhere(data[:, self.ts_row] <= timestamp)

            # print(st - time.time())
            # st = time.time()
            if "eth" in file:
                data_ = data_[:, (0, 1, 2, 4)]
            if ("zara01" in file) or ("zara02" in file):
                data_ = data_[:, (0, 1, 2, 4)]
            # if ("SDD" in file):

            # if ("students03" in file):
            #     data = data[:, (0, 1, 2, 4)]
            timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
            out = np.zeros((int(self.history_len / timecoef), 4)) - 1
            out[0:len(data_), :] = np.flip(data_, axis=0)
            full_hist_scene = np.concatenate((full_hist_scene, out[np.newaxis]), axis=0)

        return full_hist_scene[1:]

    def get_agent_future(self, dataset_ind: int, ped_id: int, timestamp: float, neighb_indexes) -> np.array:
        """
         :param dataset_ind: index of file
         :param ped_id: ID of agent
         :param timestamp:  timestamp from txt file. Target future is [timespemp, timespemp + pred_len]
         :return: future(target) trajectory of specified agent np.array shape(self.history_len+1,4).
        """

        file = self.data_files[dataset_ind]
        end_timestamp = timestamp + (
                self.pred_len / self.delta_t[file[0:file.index("/")]])  # self.delta_t[file[0:file.index("/")]])
        data = self.data[file]
        timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
        full_future_scene = (np.zeros((round(self.pred_len / timecoef), 4)) - 1)[np.newaxis]
        start_stop_ped_ind = [ped_id, ped_id + 1]
        for ind in neighb_indexes:
            start_stop_ped_ind.append(ind)
            start_stop_ped_ind.append(ind + 1)
        ind = np.searchsorted(data[:, self.index_row], start_stop_ped_ind)
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
        dataset_ind, = np.where(np.array(self.sub_data_len) <= index)
        #         print(dataset_ind)
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
        img = None
        if self.cfg["raster_params"]["use_map"]:
            txt_file = self.data_files[dataset_ind]
            if ("eth" not in txt_file) and ("UCY" not in txt_file):
                img_file = self.path + txt_file[0:txt_file.index(".")] + ".jpg"
                if self.cfg["raster_params"]["use_segm"]:
                    segm_file = self.path + txt_file[0:txt_file.index(".")] + "_s.npy"
            else:
                img_file = self.path + txt_file[0:txt_file.index(".")] + ".png"
                if self.cfg["raster_params"]["use_segm"]:
                    segm_file = self.path + txt_file[0:txt_file.index(".")] + "_s.npy"

            if img_file not in self.loaded_imgs.keys():
                print("run_time loading!")
                img = cv2.imread(img_file).astype(np.int16)
                if "SDD" in img_file:
                    img = cv2.resize(img, (img.shape[1]//5, img.shape[0]//5), interpolation=0)
                self.loaded_imgs[img_file] = img
            segm = None
            if self.cfg["raster_params"]["use_segm"]:
                if segm_file not in self.loaded_imgs.keys():
                    print("run_time loading!")
                    segm = np.load(segm_file)
                    if "SDD" in img_file:
                        segm = cv2.resize(segm, (segm.shape[1] // 5, segm.shape[0] // 5), interpolation=0)
                    self.loaded_imgs[segm_file] = segm

            # img = Image.open(img_file)
            # img = np.asarray(img, dtype="int32")
        if self.cfg["raster_params"]["use_segm"]:
            return np.copy(self.loaded_imgs[img_file]), np.copy(self.loaded_imgs[segm_file])
        else:
            return np.copy(self.loaded_imgs[img_file]), None


