import sys
sys.path.insert(0,'../')
from scripts.dataloader import DatasetFromTxt, collate_wrapper
from scripts.config import cfg
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import models
from models.resnet import GoalPredictorWOhistDist
torch.manual_seed(0)
import numpy as np

import numpy as np
from PIL import Image, ImageDraw


path_ = "../data/train/"
files = [
    "stanford/bookstore_0.txt", "stanford/bookstore_1.txt",
    "stanford/bookstore_2.txt", "stanford/bookstore_3.txt", "stanford/coupa_3.txt",
    "stanford/deathCircle_0.txt", "stanford/deathCircle_1.txt",
    "stanford/deathCircle_2.txt", "stanford/deathCircle_3.txt", "stanford/deathCircle_4.txt",
    "stanford/gates_0.txt", "stanford/gates_1.txt",

    "stanford/gates_3.txt", "stanford/gates_4.txt", "stanford/gates_5.txt", "stanford/gates_6.txt",
    "stanford/gates_7.txt", "stanford/gates_8.txt", "stanford/coupa_3.txt", "stanford/hyang_4.txt",
    "stanford/hyang_5.txt", "stanford/hyang_6.txt", "stanford/hyang_7.txt", "stanford/hyang_9.txt",
    "stanford/nexus_0.txt", "stanford/nexus_1.txt", "stanford/nexus_3.txt", "stanford/nexus_4.txt",
    "stanford/nexus_7.txt", "stanford/nexus_8.txt", "stanford/nexus_9.txt",
]

dataset = DatasetFromTxt(path_, files, cfg)

train_size = int(0.95 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

dataloader = DataLoader(train_dataset, batch_size=32,
                        shuffle=True, num_workers=8, collate_fn=collate_wrapper)

test_dataloader = DataLoader(test_dataset, batch_size=32,
                             shuffle=True, num_workers=8, collate_fn=collate_wrapper)


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

model_name="resnet_CVAE"
writer, experiment_name, best_model_path, folder_path = setup_experiment(model_name, "../logs/"+model_name, "resnet_CVAE@02.12.21")


def distance(prediction, gt, tgt_avail=None):
    if tgt_avail is None:
        return torch.mean(torch.sqrt(torch.sum((prediction - gt) ** 2, dim=1)))
    #     print (prediction.shape, gt.shape, tgt_avail.shape)
    error = torch.sum((prediction - gt) ** 2, dim=1)
    error_masked = error * tgt_avail
    if torch.sum(tgt_avail) != 0:
        return torch.sum(torch.sqrt(error_masked + 1e-6)) / torch.sum(tgt_avail)
    else:
        print("no gt available?")
        return 0


def nll_loss(pred_distr, gt, tgt_avail=None):
    assert tgt_avail.ndim == 1

    log_probs = pred_distr.log_prob(gt)

    log_probs = log_probs[tgt_avail != 0]
    #     print(log_probs)
    #     print (log_probs.shape)
    nll = - log_probs
    return torch.mean(nll)


def vis_predict(data, predictions, num_images_to_vis=4):
    images = []
    predictions = predictions.mean.detach().cpu().numpy()
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

            pose_raster = data.raster_from_agent[i] @ np.array([predictions[i, 0], predictions[i, 1], 1.])
            r = 4
            draw.ellipse((pose_raster[0] - r, pose_raster[1] - r, pose_raster[0] + r, pose_raster[1] + r), fill='red',
                         outline='red')
            r = 2
            gt_raster = data.raster_from_agent[i] @ np.array([data.tgt[i, -1, 0], data.tgt[i, -1, 1], 1.])
            draw.ellipse((gt_raster[0] - r, gt_raster[1] - r, gt_raster[0] + r, gt_raster[1] + r), fill='#33cc33',
                         outline='#33cc33')
            images.append(pil_im)
            if counter >= num_images_to_vis:
                break

    return images


from IPython.display import clear_output
from tqdm import tqdm
from models.resnet import GoalPredictorWOhistCVAE
model = GoalPredictorWOhistCVAE()
model.train()
lr = 4e-4
device = "cuda"
optimizer = optim.Adam(model.parameters(), lr=lr)
model = model.to(device)

try:
    checkpoint = torch.load(folder_path + "/last_epoch.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("succesfully loaded")
except:
    print("training from scratch")
    epoch = 0
model.train()
pass

from scripts.utils import transform_points


def pix_distance(pred, gt, raster_from_agent, mask=None):
    bs = pred.mean.shape[0]
    pred_pix = np.ones((bs, 2))
    gt_pix = np.ones((bs, 2))
    for i in range(bs):
        pred_pix[i] = transform_points(pred.mean[i:i + 1].detach().cpu().numpy(), raster_from_agent[i].numpy())
        gt_pix[i] = transform_points(gt[i:i + 1].numpy(), raster_from_agent[i])

    error = distance(torch.tensor(pred_pix), torch.tensor(gt_pix), mask)
    return error





def validation(model, loader, epoch):
    model.eval()
    nlls = []
    dist_err = []
    pix_errs = []
    for data in tqdm(loader):

        if torch.sum(data.tgt_avail[:, -1]) == 0:
            continue
        #         optimizer.zero_grad()
        data.image = torch.transpose(data.image, 1, 3)
        data.image = torch.transpose(data.image, 2, 3)
        data.image = data.image.float().to(device)

        predictions, kl = model(data.image, data.history_positions.float().cuda())
        mask = data.tgt_avail[:, -1]
        nll = nll_loss(predictions, data.tgt[:, -1, :].float().to(device), mask.to(device))  # , data.tgt_avail[:,-1]
        nlls.append(nll.item())
        distance_loss = distance(predictions.mean.detach().cpu(), data.tgt[:, -1, :], mask)  # , data.tgt_avail[:,-1]
        dist_err.append(distance_loss.item())

        pix_d = pix_distance(predictions.mean.detach().cpu(), data.tgt[:, -1, :], data.raster_from_agent, mask)
        pix_errs.append(pix_d)
    mean_nll_error = sum(nlls) / len(nlls)
    mean_distance_loss = sum(dist_err) / len(dist_err)
    mean_pix_error = sum(pix_errs) / len(pix_errs)

    #     print(mean_error)
    writer.add_scalar('test/nll', mean_nll_error, epoch)
    writer.add_scalar('test/distance_loss', mean_distance_loss, epoch)
    writer.add_scalar('test/pix_distance_loss', mean_pix_error, epoch)
    return mean_nll_error, mean_pix_error




# pbar = tqdm(dataloader)

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

        predictions, kl = model(data.image, data.history_positions.float().cuda())
        mask = data.tgt_avail[:, -1]

        loss = kl + nll_loss(predictions, data.tgt[:, -1, :].float().to(device), mask.to(device))  # , data.tgt_avail[:,-1]

        loss.backward()
        optimizer.step()
        dist_err = distance(predictions.mean.detach().cpu(), data.tgt[:, -1, :].cpu(), mask)  # , data.tgt_avail[:,-1]
        pix_dist = pix_distance(predictions, data.tgt[:, -1, :], data.raster_from_agent, mask)

        writer.add_scalar('training/distance_loss', dist_err, i+(epoch*len(dataloader)))
        writer.add_scalar('training/nll', loss, i+(epoch*len(dataloader)))
        writer.add_scalar('training/pix_distance', pix_dist, i+(epoch*len(dataloader)))

        pbar.set_postfix({'loss': loss.item()})
        if (i+1) % 50 == 0:
            images = vis_predict(data, predictions, 4)
            np_images = np.ones((0, 224, 3))
            for i in range(len(images)):
                np_img = np.asarray(images[i], dtype="uint8")
                np_img = np_img[:, :, ::-1]
                np_images = np.concatenate((np_images, np_img/255), axis=0)

            writer.add_images("train_predictions", np_images, i+(epoch*len(dataloader)), dataformats="HWC")
    pbar.close()
    validation(model, test_dataloader, epoch)
    PATH = folder_path + "/last_epoch.pth"
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
