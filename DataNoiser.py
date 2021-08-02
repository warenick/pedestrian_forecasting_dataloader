from PIL.Image import MODES
from skimage.util import random_noise
import torch
from random import shuffle

class DataNoiser():
    MODES = [
            "gaussian",
            "localvar",
            "poisson",
            "salt",
            "pepper",
            "s&p",
            "speckle"]

    def __init__(self, seed=None) -> None:
        self.seed = seed
        # self.generator = torch.Generator()
        
    def batch_img_noise(self, batch, mode=None):
        if mode is None:
            return batch
        if batch.dim()==3:
            batch = batch.permute(1,2,0)
            batch = torch.tensor(random_noise(batch, mode=mode, clip = True, seed=self.seed))
            return batch.permute(2,0,1)
        if batch.dim()==4:
            imgs = []
            batch = batch.permute(0,2,3,1)
            for i in batch:
                imgs.append(random_noise(i, mode=mode, clip = True, seed=self.seed))
            batch = torch.tensor(imgs).permute(0,3,1,2)
            return batch
        return batch


    def img_noise(self, img, mode=None):
        if mode is None:
            return img
        if img.dim()==3:
            return torch.tensor(random_noise(img, mode=mode, clip = True, seed=self.seed))
        if img.dim()==4:
            imgs = []
            for i in img:
                imgs.append(random_noise(i, mode=mode, clip = True, seed=self.seed))
            return torch.tensor(imgs)
        return img

    def pose_noise(self, poses, poses_avail=None):
        if poses_avail is None:
            return poses + torch.randn_like(poses)
        if poses.dim()==3:
            # return poses + torch.randn_like(poses)*poses_avail[None].permute(1,2,0).expand([-1,-1, poses.shape[-1]])
            return poses + torch.randn_like(poses)*poses_avail.unsqueeze(-1).expand([-1,-1, poses.shape[-1]])
        if poses.dim()==4:
            # return poses + torch.randn_like(poses)*poses_avail[None].permute(1,2,3,0).expand([-1,-1,-1, poses.shape[-1]])
            return poses + torch.randn_like(poses)*poses_avail.unsqueeze(-1).expand([-1,-1,-1, poses.shape[-1]])
        return poses

    def id_noise(self,poses, poses_avail=None, num_steps=1):
        if poses_avail is None:
            poses_avail = torch.ones(poses.shape[:poses.dim()-1])
        rand_steps = list(range(poses.shape[-2]))
        shuffle(rand_steps)
        rand_steps = rand_steps[:num_steps]
        for step in rand_steps:
            for chunk in range(poses.shape[0]):
                shuffled_peds = list(range(len(poses[chunk,:,step])))
                shuffle(shuffled_peds)
                poses[chunk,:,step] = poses[chunk,:,step][shuffled_peds]
                poses_avail[chunk,:,step] = poses_avail[chunk,:,step][shuffled_peds]  
        return poses, poses_avail

    def id_batch_noise(self, self_poses, self_poses_av, neighb_poses, neighb_poses_avail, num_steps=1):
        cutted_poses = torch.cat((self_poses.unsqueeze(1), neighb_poses),dim=1)
        cuttet_av = torch.cat((self_poses_av.unsqueeze(1), neighb_poses_avail),dim=1)
        self.id_noise(cutted_poses,cuttet_av,num_steps = num_steps)
        self_poses = cutted_poses[:,0]
        self_poses_av = cuttet_av[:,0]
        neighb_poses = cutted_poses[:,1:]
        neighb_poses_avail = cuttet_av[:,1:]
        return self_poses, self_poses_av, neighb_poses, neighb_poses_avail




if __name__=="__main__":
    from train_test_split import get_dataloaders
    from config import cfg
    from tqdm import tqdm
    from utils import preprocess_data
    import matplotlib.pyplot as plt


    def get_neighb_poses(data):
        num_peds = 0
        for sequence in data.history_agents:
            num_peds = max(num_peds, len(sequence))
        neighb_poses = -100 * torch.ones((len(data.history_agents), num_peds, 8, 6))
        neighb_poses_avail = torch.zeros((len(data.history_agents), num_peds, 8))
        for i in range(len(data.history_agents)):
            for j in range(len(data.history_agents[i])):
                try:
                    neighb_poses[i, j] = torch.tensor(data.history_agents[i][j])
                    neighb_poses_avail[i, j] = torch.tensor(data.history_agents_avail[i][j])
                except:
                    break
        neighb_poses = neighb_poses.float()
        return neighb_poses, neighb_poses_avail

    cfg["one_ped_one_traj"] = False
    cfg["raster_params"]["draw_hist"] = 1
    _, val_dataloader = get_dataloaders(bs=4, num_w=0, path_="data/train/", cfg_=cfg)
    pbar = tqdm(val_dataloader)

    seed = 6
    dn = DataNoiser(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    check_img_noise = False
    check_poses_noise = False
    check_id_nose = True

    with torch.no_grad():
        for batch_num, data in enumerate(pbar):
            if batch_num>0: break
            if check_img_noise:
                imgs, segm = preprocess_data(data, cfg)
                imgs_in_batch_nosed = dn.batch_img_noise(imgs,mode=dn.MODES[0])
                imgs = imgs.permute(0,2,3,1)
                segm = segm.permute(0,2,3,1)
                plt.imshow(imgs[0])
                for mode in dn.MODES:
                    nosed_imgs = dn.img_noise(imgs, mode=mode)
                    nosed_img = dn.img_noise(imgs[0], mode=mode)
                    nosed_segms = dn.img_noise(segm, mode=mode)
                    nosed_segm = dn.img_noise(segm[0], mode=mode)
                    plt.imshow(nosed_img)
                    plt.imshow(nosed_segm)
            self_poses = torch.tensor(data.history_positions, dtype=torch.float32)
            self_poses_av = torch.tensor(data.history_av)
            neighb_poses, neighb_poses_avail = get_neighb_poses(data)
            if check_poses_noise:
                print((self_poses - dn.pose_noise(self_poses, self_poses_av)).mean())
                print((neighb_poses - dn.pose_noise(neighb_poses, neighb_poses_avail)).mean())
            if check_id_nose:
                self_poses, self_poses_av, neighb_poses, neighb_poses_avail = dn.id_batch_noise(self_poses, self_poses_av, neighb_poses, neighb_poses_avail,num_steps=3)
