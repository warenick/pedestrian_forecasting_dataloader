import sys

sys.path.insert(0, '../')
from scripts.dataloader import DatasetFromTxt, collate_wrapper
from scripts.dataloader_prerecorded import DatasetFromPrePorccessedFolder
from scripts.config import cfg
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import models
from models.resnet import GoalPredictorWOhistCVAE, GoalPredictorWOhistCVAE_cond_hist, GoalPredictorWOhist, GoalCond
from models.baseline_dnn import BaseGoalPredictor
from models.trajpred import TrajPredGoalCond
torch.manual_seed(0)
import numpy as np


device = "cuda"
# path_ = "../data/train/"
# files = ["stanford/bookstore_0.txt", "stanford/bookstore_1.txt", "stanford/bookstore_2.txt" ,"stanford/bookstore_3.txt",
#          "stanford/coupa_3.txt", "stanford/deathCircle_0.txt", "stanford/deathCircle_1.txt",
#          "stanford/deathCircle_2.txt", "stanford/deathCircle_3.txt", "stanford/deathCircle_4.txt" ,
#          "stanford/gates_0.txt", "stanford/gates_1.txt", "stanford/gates_3.txt", "stanford/gates_4.txt",
#          "stanford/gates_5.txt", "stanford/gates_6.txt", "stanford/gates_7.txt", "stanford/gates_8.txt",
#          "stanford/hyang_5.txt", "stanford/hyang_6.txt", "stanford/hyang_7.txt", "stanford/hyang_9.txt",
#          "stanford/nexus_1.txt", "stanford/nexus_3.txt", "stanford/nexus_4.txt", "stanford/nexus_7.txt",
#          "stanford/nexus_8.txt", "stanford/nexus_9.txt"]

# dataset = DatasetFromTxt(path_, files, cfg, use_forces=True, forces_file="forces_19.02.txt")
path = "/home/jovyan/traj_data_1/"
dataset = DatasetFromPrePorccessedFolder(path)


train_size = int(0.95 * len(dataset))
test_size = len(dataset) - train_size
print("len dataset:", len(dataset), "   train_size ", train_size, "   test_size:",test_size, "   sum:", test_size+train_size)
# if len(dataset) < 165730:
#     exit()
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

dataloader = DataLoader(train_dataset, batch_size=1526,
                        shuffle=True, num_workers=3, collate_fn=collate_wrapper)

test_dataloader = DataLoader(test_dataset, batch_size=64,
                             shuffle=True, num_workers=0, collate_fn=collate_wrapper)


from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def setup_experiment(title: str, logdir: str = "./tb", experiment_name: Optional[str] = None) -> Tuple[SummaryWriter, str, str, str]:
    """
    :param title: name of experiment
    :param logdir: tb logdir
    :return: writer object,  modified experiment_name, best_model path
    """
    if experiment_name is None:
        experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    folder_path = os.path.join(logdir, experiment_name)
    writer = SummaryWriter(log_dir=folder_path)

    best_model_path = f"{folder_path}/{experiment_name}_best.pth"
    return writer, experiment_name, best_model_path, folder_path


model_name="trajPred1_forces_neig"
writer, experiment_name, best_model_path, folder_path = setup_experiment(model_name, "../logs/"+model_name, "trajPred1_forces_neig")


def distance(prediction, gt, tgt_avail=None):
    if tgt_avail is None:
        return torch.mean(torch.sqrt(torch.sum((prediction - gt) ** 2, dim=1)))
    tgt_avail = torch.tensor(tgt_avail).bool().to(prediction.device)
    error = prediction - gt
    norm = torch.norm(error, dim=1)
    error_masked = norm[tgt_avail]
    if torch.sum(tgt_avail) != 0:
        return torch.mean(error_masked)
    else:
        print("no gt available?")
        return 0


def nll_loss(pred_distr, gt, tgt_avail=None):
    assert tgt_avail.ndim == 1

    log_probs = pred_distr.log_prob(gt)

    log_probs = log_probs[tgt_avail != 0]
    nll = - log_probs
    return torch.mean(nll)


def ade_loss(pred_traj, gt, mask):

    assert pred_traj.ndim == 3
    assert gt.ndim == 3
    assert mask.ndim == 2

    error = pred_traj - gt
    norm = torch.norm(error, dim=2)[mask]
    return torch.mean(norm)




def vis_predict(data, predictions, num_images_to_vis=4, cov=None, traj=None):
    images = []
    predictions = predictions.mean.detach().cpu().numpy()
    if traj is not None:
        traj = traj.detach().cpu().numpy()
#     predictions = predictions.rsample().detach().cpu().numpy()
#     predictions = predictions.detach().cpu().numpy()
    if cov is not None:
        cov = cov.detach().cpu().numpy()
    if torch.sum(data.tgt_avail[:, -1]).item() == 0:
        return None
    bs = data.image.shape[0]

    if torch.sum(data.tgt_avail[:, -1]).item() <= 4:
        num_images_to_vis = torch.sum(data.tgt_avail[:, -1]).item()

    counter = 0
    for i in range(bs):

        if data.tgt_avail[i, -1]:
            counter += 1
            np_im = data.image[i].detach().cpu().numpy()
            np_im = np.transpose(np_im, (1, 2, 0))
            pil_im = Image.fromarray(np.asarray(np_im, dtype="uint8"))
            draw = ImageDraw.Draw(pil_im)

            pose_raster = data.raster_from_agent[i].numpy() @ np.array([predictions[i, 0], predictions[i, 1], 1.])
            r = 4
            draw.ellipse((pose_raster[0] - r, pose_raster[1] - r, pose_raster[0] + r, pose_raster[1] + r), fill='red',
                             outline='red')
            if cov is not None:
                draw = ImageDraw.Draw(pil_im, "RGBA")
                for j in range(1,4):
                    r_cov_x = np.array([j*np.sqrt(cov[i,0,0]), 0, 0])
                    r_cov_y = np.array([0, j*np.sqrt(cov[i,1,1]),0])
                    r_cov_min = data.raster_from_agent[i].numpy() @ np.array([predictions[i, 0] - r_cov_x[0],
                                                                      predictions[i, 1] - r_cov_y[1],
                                                                     1])
                    r_cov_max = data.raster_from_agent[i].numpy() @ np.array([predictions[i, 0] + r_cov_x[0],
                                                                      predictions[i, 1] + r_cov_y[1],
                                                                      1])
                    draw.ellipse((r_cov_min[0], r_cov_min[1], r_cov_max[0], r_cov_max[1]),
                                 fill=(100, 100, 170, 75),
                                 outline='#878c43')
            r = 2
            gt_raster = data.raster_from_agent[i].numpy() @ np.array([data.tgt[i, -1, 0], data.tgt[i, -1, 1], 1.])
            draw = ImageDraw.Draw(pil_im)
            draw.ellipse((gt_raster[0] - r, gt_raster[1] - r, gt_raster[0] + r, gt_raster[1] + r), fill='#33cc33',
                         outline='#33cc33')
            if traj is not None:

                for pose in traj[i]:
                    pose_raster = data.raster_from_agent[i].numpy() @ np.array([pose[0], pose[1], 1.])
                    r = 2
                    draw.ellipse((pose_raster[0] - r, pose_raster[1] - r, pose_raster[0] + r, pose_raster[1] + r),
                                 fill='red', outline='red')
            images.append(pil_im)

            if counter >= num_images_to_vis:
                break


    return images


from IPython.display import clear_output
from tqdm import tqdm


model = TrajPredGoalCond(True)
model.train()
device = "cuda"
model = model.to(device)
pass

lr = 5e-4
optimizer = optim.Adam(model.parameters(), lr=lr)


try:
    checkpoint = torch.load(folder_path + "/last_epoch.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("succesfully loaded")

except Exception as e:
    print (str(e))
    print("training from scratch")
    epoch = 0


model.train()
pass

from scripts.utils import transform_points


def pix_distance(pred, gt, transforms, mask=None):
    raster_from_agent, loc_img_to_global = transforms
    bs = pred.mean.shape[0]
    pred_pix = np.ones((bs, 2))
    gt_pix = np.ones((bs, 2))
    for i in range(bs):
        tranform = loc_img_to_global[i].numpy() @ raster_from_agent[i].numpy()
        assert tranform[2, 2] == 1

        pred_pix[i] = transform_points(pred.mean[i:i + 1].detach().cpu().numpy(), tranform)
        gt_pix[i] = transform_points(gt[i:i + 1].numpy(), tranform)

    error = distance(torch.tensor(pred_pix), torch.tensor(gt_pix), mask)
    return error


import numpy as np

def validation(model, loader, epoch):
    model.eval()
    nlls = []
    dist_err = []
    pix_errs = []

    ades = []
    for data in tqdm(loader):

        if torch.sum(data.tgt_avail[:, -1]) == 0:
            continue
        data.image = torch.transpose(data.image, 1, 3)
        data.image = torch.transpose(data.image, 2, 3)
        data.image = data.image.float().to(device)
        
        predictions, kl, traj = model(data.image, data.history_positions.float().to(device), data.forces.float().to(device))
        mask_goal = data.tgt_avail[:, -1] == 1
        mask_future = data.tgt_avail == 1

        ade = ade_loss(traj, data.tgt.to(traj.device), mask_future)
        ades.append(ade.item())
        nll = nll_loss(predictions, data.tgt[:, -1, :].float().to(device), mask_goal.to(device))  # , data.tgt_avail[:,-1]
        nlls.append(nll.item())
        distance_loss = distance(predictions.mean.detach().cpu(), data.tgt[:, -1, :], mask_goal)  # , data.tgt_avail[:,-1]
        dist_err.append(distance_loss.item())

        pix_d = pix_distance(predictions, data.tgt[:, -1, :], [data.raster_from_agent, data.loc_im_to_glob], mask_goal)

        pix_errs.append(pix_d)
    mean_ades = sum(ades) / len(ades)
    mean_nll_error = sum(nlls) / len(nlls)
    mean_distance_loss = sum(dist_err) / len(dist_err)
    mean_pix_error = sum(pix_errs) / len(pix_errs)

    writer.add_scalar('test/ade', mean_ades, i+(epoch*len(dataloader)))
    writer.add_scalar('test/nll', mean_nll_error, epoch)
    writer.add_scalar('test/distance_loss', mean_distance_loss, epoch)
    writer.add_scalar('test/pix_distance_loss', mean_pix_error, epoch)
    return mean_nll_error, mean_pix_error


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


from PIL import Image, ImageDraw

count = 0
for epoch in range(epoch, 200):
    pbar = tqdm(dataloader)

    for i, data in enumerate(pbar):

        if torch.sum(data.tgt_avail[:, -1]) == 0:
            continue
        model.train()
        optimizer.zero_grad()
        data.image = torch.transpose(data.image, 1, 3)
        data.image = torch.transpose(data.image, 2, 3)
        data.image = data.image.float().to(device)
        mask_goal = data.tgt_avail[:, -1] == 1
        mask_future = data.tgt_avail == 1
#         with torch.autograd.detect_anomaly():
        predictions, kl, traj = model(data.image, data.history_positions.float().to(device),
                                      goal=data.tgt[:, -1, :].float().to(device), forces=data.forces.float().to(device))
        if torch.sum(data.forces) > 0:
            count+=1
            if count == 1:
                print("forces ok")
        cov_size_loss = torch.sum(predictions.covariance_matrix[mask_goal].det())
        nll_ = nll_loss(predictions, data.tgt[:, -1, :].float().to(device), mask_goal.to(device))
        ade = ade_loss(traj, data.tgt.to(device), mask_future.to(device))
        dist_err = distance(predictions.mean.cpu(), data.tgt[:, -1, :].cpu(), mask_goal.cpu()).to(device)
        # , data.tgt_avail[:,-1]
        kl_weight = min(((epoch ** 2 + 1) * 0.01), 10)
        loss = 0.5*ade + 0.1 * cov_size_loss + kl_weight * kl + 1 * nll_ + 1 * (dist_err)  # , data.tgt_avail[:,-1]

        loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        
        pix_dist = pix_distance(predictions, data.tgt[:, -1, :], [data.raster_from_agent, data.loc_im_to_glob], mask_goal)
        writer.add_scalar("training/ade", ade, i + (epoch * len(dataloader)))
        writer.add_scalar("training/kl_weight", kl_weight, i + (epoch * len(dataloader)))
        writer.add_scalar("training/lr", get_lr(optimizer), i + (epoch * len(dataloader)))
        writer.add_scalar('training/distance_loss', dist_err, i + (epoch * len(dataloader)))
        writer.add_scalar('training/kl_loss', kl, i + (epoch * len(dataloader)))
        writer.add_scalar('training/nll', nll_, i + (epoch * len(dataloader)))
        writer.add_scalar('training/pix_distance', pix_dist, i + (epoch * len(dataloader)))

        pbar.set_postfix({'loss': loss.item()})
        if (i + 1) % 10 == 0:
            images = vis_predict(data, predictions, 4, predictions.covariance_matrix, traj=traj)

            np_images = np.ones((0, data.image.shape[2], 3))
            for j in range(len(images)):
                np_img = np.asarray(images[j], dtype="uint8")
                np_img = np_img[:, :, ::-1]
                np_images = np.concatenate((np_images, np_img / 255), axis=0)

            writer.add_images("train_predictions", np_images, i + (epoch * len(dataloader)), dataformats="HWC")
    pbar.close()

    PATH = folder_path + "/last_epoch.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)
    with torch.no_grad():
        validation(model, test_dataloader, epoch)
