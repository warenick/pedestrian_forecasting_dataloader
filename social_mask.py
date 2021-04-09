import sys
import os
import torch
from pathlib import Path
from dataloader import DatasetFromTxt, collate_wrapper
from config import cfg
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import numpy as np
import glob

cfg["raster_params"]["draw_hist"] = False
cfg["raster_params"]["use_map"] = False
cfg["raster_params"]["normalize"] = True


def find_min_time(t1, t2):
    '''given two time frame arrays, find then min dist (time)'''
    min_d = 9e4
    t1, t2 = t1[:8], t2[:8]

    for t in t2:
        if abs(t1[0]-t)<min_d:
            min_d = abs(t1[0]-t)

    for t in t1:
        if abs(t2[0]-t)<min_d:
            min_d = abs(t2[0]-t)

    return min_d

def find_min_dist2(traj1,traj2):

    dists = np.linalg.norm(traj1-traj2)
    min_d = dists.min()
    return min_d

def find_min_dist(p1x, p1y, p2x, p2y):
    '''given two time frame arrays, find then min dist'''
    min_d = 9e4
    p1x, p1y = p1x[:8], p1y[:8]
    p2x, p2y = p2x[:8], p2y[:8]

    for i in range(len(p1x)):
        for j in range(len(p1x)):
            if ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5 < min_d:
                min_d = ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5

    return min_d

def social_and_temporal_filter(p1_key, p2_key, all_data_dict, time_thresh=48, dist_tresh=100):
    p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])
    p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
    p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
    p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]

    if find_min_time(p1_time, p2_time)>time_thresh:
        return False
    if find_min_dist(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
    # if find_min_dist2(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
        return False

    return True

def mark_similar(mask, sim_list):
    for i in range(len(sim_list)):
        for j in range(len(sim_list)):
            mask[sim_list[i]][sim_list[j]] = 1


def collect_data(set_name, dataset_type = 'image', batch_size=512, time_thresh=48, dist_tresh=100, scene=None, verbose=True, root_path="./"):

    assert set_name in ['train','val','test']

    '''Please specify the parent directory of the dataset. In our case data was stored in:
        root_path/trajnet_image/train/scene_name.txt
        root_path/trajnet_image/test/scene_name.txt
    '''

    # rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name)
    rel_path = 'data/{0}'.format(set_name)

    full_dataset = []
    full_masks = []

    current_batch = []
    mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

    current_size = 0
    social_id = 0
    part_file = '/{}.txt'.format('*' if scene == None else scene)

    for file in glob.glob(root_path + rel_path + part_file):
        # scene_name = file[len(root_path+rel_path)+1:-6] + file[-5]
        # data = np.loadtxt(fname = file, delimiter = ' ')
        scene_name = file[len(root_path+rel_path)+1:-4]
        data = np.loadtxt(fname = file)

        data_by_id = {}
        # for frame_id, person_id, x, y, in data:
            # if person_id not in data_by_id.keys():
            #     data_by_id[person_id] = []
            # data_by_id[person_id].append([person_id, frame_id, x, y])
        for n in range(len(data)):
            if data[n,1] not in data_by_id.keys():
                data_by_id[data[n,1]] = []
            data_by_id[data[n,1]].append([data[n,1],data[n,0],data[n,2],data[n,3]])

        all_data_dict = data_by_id.copy()
        if verbose:
            print("Total People: ", len(list(data_by_id.keys())))
        while len(list(data_by_id.keys()))>0:
            related_list = []
            curr_keys = list(data_by_id.keys())

            if current_size<batch_size:
                pass
            else:
                full_dataset.append(current_batch.copy())
                mask_batch = np.array(mask_batch)
                full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])

                current_size = 0
                social_id = 0
                current_batch = []
                mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

            current_batch.append((all_data_dict[curr_keys[0]]))
            related_list.append(current_size)
            current_size+=1
            del data_by_id[curr_keys[0]]

            for i in range(1, len(curr_keys)):
                if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh, dist_tresh):
                    current_batch.append((all_data_dict[curr_keys[i]]))
                    related_list.append(current_size)
                    current_size+=1
                    del data_by_id[curr_keys[i]]

            mark_similar(mask_batch, related_list)
            social_id +=1


    full_dataset.append(current_batch)
    mask_batch = np.array(mask_batch)
    full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])
    return full_dataset, full_masks


def generate_pooled_data(b_size, t_tresh, d_tresh, train=True, scene=None, verbose=True):
    if train:
        full_train, full_masks_train = collect_data(
            "train",
            batch_size=b_size, 
            time_thresh=t_tresh, 
            dist_tresh=d_tresh, 
            scene=scene, 
            verbose=verbose)
        train_name = "data/social_pool_data/train_{0}_{1}_{2}_{3}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)
        with open(train_name, 'wb') as f:
            pickle.dump([full_train, full_masks_train], f)
        return full_train, full_masks_train

    if not train:
        full_test, full_masks_test = collect_data(
            "test", 
            batch_size=b_size, 
            time_thresh=t_tresh, 
            dist_tresh=d_tresh, 
            scene=scene, 
            verbose=verbose
            )

        test_name = "data/social_pool_data/test_{0}_{1}_{2}_{3}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)# + str(b_size) + "_" + str(t_tresh) + "_" + str(d_tresh) + ".pickle"
        with open(test_name, 'wb') as f:
            pickle.dump([full_test, full_masks_test], f)
        return full_test, full_masks_test




# def create_social_mask(data):
#     num_agents = neigh_localM.shape[0]+1
#     sm = np.zeros((num_agents,num_agents))
    
 
#     return social_mask
 






if __name__=="__main__":
    # generate_pooled_data(512,0,25, train=True, verbose=True, root_path="./")
    traj, mask = generate_pooled_data( 
        b_size=10,
        t_tresh=0,
        d_tresh=25, 
        train=True, 
        verbose=True,
        # scene = 'biwi_eth/biwi_eth__'
        scene = "stanford/bookstore_0"
        )

    print(traj[0],mask[0])
    exit()






# if __name__ == '__main__':
#     if torch.cuda.is_available():
#         dev = torch.device(0)
#     dev = 'cpu'
    
#     path_ = "data/train/"
#     # get all availible data
#     # pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
#     # files = [str(x).replace(path_,"") for x in pathes]
#     files = ['biwi_eth/biwi_eth.txt']
#     dataset = DatasetFromTxt(path_, files, cfg)
#     print("len dataset:", len(dataset))
#     dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, collate_fn=collate_wrapper, pin_memory=True)    
#     pbar = tqdm(dataloader)
#     for batch_num, data in enumerate(pbar):
#         if batch_num > 0.1*len(dataloader):
#             break 
#         # print(data["social_layer"].shape)
#         sm = create_social_mask(data["history_agents"])
#         break 
#         pass
#         # pbar.status_printer("num_batch="+str(batch_num))

#     exit()
