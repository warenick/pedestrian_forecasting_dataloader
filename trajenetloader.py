import numpy as np
from tqdm import tqdm


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
        self.uniq_ids = {}
        self.uniq_ids_len = 0
        self.sub_data_len = [0]
        self.sub_data_len_ids = [0]
        self.cfg = cfg
        print("loading files")
        for file in tqdm(data_files):

            if "SDD" in file:
                try:
                    name = path + "/" + file
                    new_name = name[:name.index(".")] + ".npy"
                    self.data[file] = np.load(new_name)
                except:    
                    self.data[file] = np.loadtxt(path + "/" + file, delimiter=' ', usecols=[0, 1, 2, 3, 4, 5])
                    if self.cfg["use_only_pedestrian"]:
                        types = np.genfromtxt(path + "/" + file ,usecols=[9],dtype='str')
                        only_peds = (types == "\"Pedestrian\"")
                        self.data[file] = self.data[file][only_peds]
                       
                self.data[file] = self.data[file][
                    (self.data[file][:, 5] + (self.data[file][:, 5].min() % 12)) % 12 == 0]
                
                self.data[file] = self.data[file][:, (5, 0, 1, 2, 3, 4)]
                self.data[file][:, 2] = (self.data[file][:, 2] + self.data[file][:, 4]) / 2
                self.data[file][:, 3] = (self.data[file][:, 3] + self.data[file][:, 5]) / 2
                self.data[file] = self.data[file][:, :4]
                if self.cfg["uniq_traj_for_agents"]:
                    ids, indexes, counts = np.unique(self.data[file][:,1],return_index = True,return_counts = True)
                    ids = ids[counts>=20]
                    indexes = indexes[counts>=20]
                    tss = (self.data[file][:,0])[indexes]
                    tss = tss + 12 * 7 # calc time at 8 step of trajectory
                    self.uniq_ids[file] = np.array([ids,tss])    #np.unique(np.concatenate([self.data[file][:,0][None],self.data[file][:,1][None]]),)#
                    self.uniq_ids_len+=len(ids)
                    self.sub_data_len_ids.append(self.uniq_ids_len)
            else:
                self.data[file] = np.genfromtxt(path + "/" + file, delimiter='')
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

    def get_agent_history(self, dataset_ind: int, ped_id: int, timestamp: float) -> np.array:

        """
         :param dataset_ind: index of file
         :param ped_id: ID of agent
         :param timestamp:  timestamp from txt file. Observed history is [timespemp - history_len, timespemp]
         :return: observed trajectory of specified agent np.array shape(self.history_len+1,4).
        """

        file = self.data_files[dataset_ind]
        data = self.data[file]

        start_ts = timestamp - (
                self.history_len / self.delta_t[file[0:file.index("/")]])  # *self.delta_t[file[0:file.index("/")]])
        #         if self.cfg["raster_params"]["use_map"] == True:

        data = data[data[:, self.index_row] == ped_id]  # filter by index
        data = data[(data[:, self.ts_row] > start_ts)]  # filter by timestamp
        data = data[data[:, self.ts_row] <= timestamp]
        # out = np.zeros((self.history_len + 1, 4)) - 1
        if "eth" in file:
            data = data[:, (0, 1, 2, 4)]
        if ("zara01" in file) or ("zara02" in file):
            data = data[:, (0, 1, 2, 4)]
        # if ("SDD" in file):

        # if ("students03" in file):
        #     data = data[:, (0, 1, 2, 4)]
        timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
        out = np.zeros((int(self.history_len / timecoef), 4)) - 1
        out[0:len(data), :] = np.flip(data, axis=0)
        return out

    def get_agent_future(self, dataset_ind: int, ped_id: int, timestamp: float) -> np.array:
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
        data = data[data[:, self.index_row] == ped_id]  # filter by index
        data = data[(data[:, self.ts_row] <= end_timestamp)]  # filter by timestamp
        data = data[data[:, self.ts_row] > timestamp]
        if "eth" in file:
            data = data[:, (0, 1, 2, 4)]
        if ("zara01" in file) or ("zara02" in file):
            data = data[:, (0, 1, 2, 4)]
        # if ("students03" in file):
        #     data = data[:, (0, 1, 2, 4)]
        timecoef = self.delta_timestamp[file[0:file.index("/")]] * self.delta_t[file[0:file.index("/")]]
        out = np.zeros((int(round(self.pred_len / timecoef)), 4)) - 1
        out[0:len(data), :] = np.array(data)
        return out

    def get_subdataset_ts_separator(self, index: int) -> int:
        """
            :param index: index of data (row) in whole(combined) dataset
            :return: index of dataset file
        """
        if self.cfg["uniq_traj_for_agents"]:
            dataset_ind, = np.where(np.array(self.sub_data_len_ids) <= index)
        else:
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

    def get_uniq_pedId_and_timestamp_by_index(self, dataset_ind: int, index: int) -> (int, int):
        """
        :param dataset_ind: index of dataset file (from get_subdataset_ts_separator)
        :param index: index of data (agent id) in whole(combined) dataset
        :return:
            ped_id:  pedestrinan id
            ts: timestamp

        """
        file = self.data_files[dataset_ind]
        # data = self.data[file]
        index = index - self.sub_data_len_ids[dataset_ind]
        ped_id = self.uniq_ids[file][0,index]
        ts = self.uniq_ids[file][1,index]

        # ts = data[index, self.ts_row]
        # ped_id = data[index, self.index_row]
        return ped_id, ts


    def get_map(self, dataset_ind: int, ped_id: int, timestamp: float):
        import cv2
        """
         :param dataset_ind: index of file
         :param ped_id: ID of agent
         :param timestamp:  timestamp from txt file. Target future is [timespemp, timespemp + pred_len]
         :return:
        """
        # 1. find file and image
        # 2. rotate to allign with agent motion
        # 3. crop with specified in cfg area (aka 5x5meters?)
        # a. pixels to meters transformation?
        img = None
        if self.cfg["raster_params"]["use_map"]:
            txt_file = self.data_files[dataset_ind]
            if ("eth" not in txt_file) and ("UCY" not in txt_file):
                img_file = self.path + txt_file[0:txt_file.index(".")] + ".jpg"
            else:
                img_file = self.path + txt_file[0:txt_file.index(".")] + ".png"
            img = cv2.imread(img_file).astype(np.int16)
            # img = Image.open(img_file)
            # img = np.asarray(img, dtype="int32")
        return img
