import math
from typing import List, Union

import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import torch
import matplotlib.pyplot as plt

try:
    from config import SDD_scales, cropping_cfg
except:
    from .config import SDD_scales, cropping_cfg
    

def transform_points2d(points: np.array, transform: np.array):
    """
        :param points: trajectory or in general set of 2d points to be transformed, shape [N, 2]
        :param transform: transformation matrix 3x3
    """
    path_ = np.ones([points.shape[0], points.shape[1] + 1])
    path_[:, :2] = points
    return np.einsum("ki, ji->jk", transform, path_)[:, :2]


def transform_points3d(points: np.array, transform: np.array):
    """
        :param points: trajectory or in general set of 3d points to be transformed, shape [X, N, 2]
        :param transform: transformation matrix 3x3
    """
    path_ = np.ones([points.shape[0], points.shape[1], points.shape[2] + 1])
    path_[:, :, :2] = points
    return np.einsum("ki, xji->xjk", transform, path_)[:, :, :2]


def transform_points(points: np.array, transform: np.array):
    """
        :param points: trajectory or in general set of 2d or 3d points to be transformed
        :param transform: transformation matrix 3x3
    """
    if points.ndim == 3:
        return transform_points3d(points, transform)
    if points.ndim == 2:
        return transform_points2d(points, transform)


def sort_neigh_history(unsorted: List[np.array]):
    '''
        :param unsorted - List of neighb observations.
                   neighb observations is np.array of shape X(x<=obs_len, 4)
        :return: "sorted" List (of np.arrays) with len of timestamps of observation(<= obs.len)
                          consist of np.arrays with shape of (numb_agents at given timestamp, 4)
    '''
    unsorted = np.array(unsorted)
    time_sorted = np.zeros((9, len(unsorted), 4)) - 1
    unique_tss = np.array(np.unique(unsorted[:, :, 0]))
    unique_tss = unique_tss[unique_tss != -1][::-1]
    for i, ts in enumerate(unique_tss):
        poses_at_ts = unsorted[unsorted[:, :, 0] == ts]
        time_sorted[i, :len(poses_at_ts)] = poses_at_ts
    return time_sorted

def sort_neigh_future (unsorted: List[np.array]):
    '''
        :param unsorted - List of neighb observations.
                   neighb observations is np.array of shape X(x<=obs_len, 4)
        :return: "sorted" List (of np.arrays) with len of timestamps of observation(<= obs.len)
                          consist of np.arrays with shape of (numb_agents at given timestamp, 4)
    '''
    unsorted = np.array(unsorted)
    time_sorted = np.zeros((12, len(unsorted), 4)) - 1
    unique_tss = np.array(np.unique(unsorted[:, :, 0]))
    unique_tss = unique_tss[unique_tss != -1] #[::-1]
    for i, ts in enumerate(unique_tss):
        poses_at_ts = unsorted[unsorted[:, :, 0] == ts]
        time_sorted[i, :len(poses_at_ts)] = poses_at_ts
    return time_sorted


def trajectory_orientation(last_pose, prev_pose):
    dy = last_pose[1] - prev_pose[1]
    dx = last_pose[0] - prev_pose[0]
    angle = math.atan2(dy, dx) / math.pi * 180.0
    return angle


def rotate_image(img: PIL.Image, angle, center: Union[np.array, list, tuple], mask:PIL.Image=None):
    new_img = img.rotate(angle, center=(center[0], center[1]))
    new_mask = None
    if mask is not None:
        new_mask = mask.rotate(angle, center=(center[0], center[1]))
    img_to_agentpix = np.array([[1, 0, -center[0]],
                                [0, 1, -center[1]],
                                [0, 0, 1]])

    angle = angle / 180 * np.pi
    map_to_local = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]])

    agent_pix_to_img = np.array([[1, 0, center[0]],
                                 [0, 1, center[1]],
                                 [0, 0, 1]])
    return new_img, agent_pix_to_img @ map_to_local @ img_to_agentpix, new_mask


def crop_image_crowds(img, cfg, agent_center_img: np.array, transform, rot_mat, file, mask=None):
    # transform -> from rot image to world
    size = img.size

    agent_center_m = (transform @ np.array([agent_center_img[0], agent_center_img[1], 1]))

    orientation = 1
    if ("students" in file):
        orientation = -1
        rot_mat = np.linalg.inv(rot_mat)
    if "eth" in file:
        orientation = -1
    tl_disp = rot_mat @ (np.array([orientation*cfg["agent_center"][0]* cfg["image_area_meters"][0],
                         -cfg["agent_center"][1] * cfg["image_area_meters"][1],
                         1]))

    tl_m = (agent_center_m) + tl_disp

    br_displ = rot_mat @ (np.array([orientation*( -1 + cfg["agent_center"][0]) * cfg["image_area_meters"][0],
                          (1 - cfg["agent_center"][1]) * cfg["image_area_meters"][1],
                          1]))
    br_m = (agent_center_m) + br_displ

    tl = (np.linalg.inv(transform)  @ np.array([tl_m[0], tl_m[1], 1]))[:2]
    br = (np.linalg.inv(transform)  @ np.array([br_m[0], br_m[1], 1]))[:2]


    cropped = img.crop((min(tl[0], br[0]), min(tl[1], br[1]), max(tl[0], br[0]), max(tl[1], br[1])))
    mask_cropped = None
    if mask is not None:
        mask_cropped = mask.crop((min(tl[0], br[0]), min(tl[1], br[1]), max(tl[0], br[0]), max(tl[1], br[1])))
        mask_cropped = mask_cropped.resize(cfg["image_shape"], 0)
    image_resize_coef = [cfg["image_shape"][0] / cropped.size[0], cfg["image_shape"][1] / cropped.size[1]]
    cropped = cropped.resize(cfg["image_shape"])
    # cropped.show()
    return cropped, image_resize_coef, mask_cropped


def crop_image(img, cfg, agent_center: np.array, pix_to_met, mask_pil):
    size = img.size
    # np_img = np.asarray(img)

    tl_x = min(size[1], max(0, agent_center[1] -
                            cfg["agent_center"][1] * cfg["image_area_meters"][0] / pix_to_met))
    #     print (tl_y)
    tl_y = min(size[0], max(0, agent_center[0] -
                            cfg["agent_center"][0] * cfg["image_area_meters"][1] / pix_to_met))
    #     print (tl_x)
    br_x = min(size[1], max(0, agent_center[1] +
                            (1 - cfg["agent_center"][1]) * cfg["image_area_meters"][0] / pix_to_met))

    br_y = min(size[0], max(0, agent_center[0] +
                            (1 - cfg["agent_center"][0]) * cfg["image_area_meters"][1] / pix_to_met))
    cropped = img.crop((tl_y, tl_x, br_y, br_x))

    mask_pil_cropped = mask_pil.crop((tl_y, tl_x, br_y, br_x))
    image_resize_coef = [cfg["image_shape"][0] / cropped.size[0], cfg["image_shape"][1] / cropped.size[1]]# cropped = img.crop((tl_y, tl_x, br_y, br_x))  #np_img[int(tl_x): int(br_x), int(tl_y): int(br_y)]

    cropped = cropped.resize(cfg["image_shape"])
    mask_pil_cropped = mask_pil_cropped.resize(cfg["image_shape"], 0)
    return cropped, image_resize_coef, mask_pil_cropped


def calc_transform_matrix(init_coord, angle, scale, output_shape: List):
    # init_coord[:2]+=border_w
    #     print(init_coord)
    a = np.eye(3)
    # a[2,2] = 0
    # border = 200
    a[0, 2] = -init_coord[0]  # -border
    a[1, 2] = -init_coord[1]  # -border
    #     print("a", a)
    # b = np.eye(3)
    b = np.array([[np.cos(angle), np.sin(angle), 0],
                  [-np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])

    c = np.eye(3)
    c[0, 0] *= scale[0]
    c[1, 1] *= scale[1]

    d = np.eye(3)
    d[0, 2] = 0.25 * output_shape[0]  # TODO
    d[1, 2] = 0.5 * output_shape[1]  # TODO
    #     print("d: ",d)
    transf = d @ c @ b @ a
    return transf


def sdd_crop_and_rotate(img: np.array, path, border_width=400, draw_traj=1, pix_to_m_cfg=SDD_scales,
                         cropping_cfg=cropping_cfg, file=None, mask=None):
    img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
    mask_pil = Image.fromarray(np.asarray(mask, dtype="uint8"))
    draw = ImageDraw.Draw(img_pil)
    scale_factor = 5
    border = border_width//scale_factor

    scale = pix_to_m_cfg[file]["scale"]
    if draw_traj:
        R = 1
        for pose in path:
            if np.linalg.norm(pose - np.array([-1., -1.])) > 1e-6:
                draw.ellipse((pose[0]/scale_factor - R, pose[1]/scale_factor - R,
                              pose[0]/scale_factor + R, pose[1]/scale_factor + R),
                             fill='blue', outline='blue')



    # img_pil = img_pil.resize((img_pil.size[0]//scale_factor, img_pil.size[1]//scale_factor))
    # mask_pil = mask_pil.resize((mask_pil.size[0] // scale_factor, mask_pil.size[1] // scale_factor), 0)
    img_pil = ImageOps.expand(img_pil, (border, border))
    mask_pil = ImageOps.expand(mask_pil, (border, border))

    angle_deg = trajectory_orientation(path[0], path[1])
    if np.linalg.norm(path[1] - np.array([-1., -1.])) < 1e-6:
        angle_deg = 0
    angle_rad = angle_deg / 180 * math.pi
    img_pil, map_to_local, mask_pil = rotate_image(img_pil, angle_deg, center=path[0]/scale_factor + border, mask=mask_pil)
    crop_img, scale, crop_mask = crop_image(img_pil, cropping_cfg, agent_center=path[0]/scale_factor + border,
                                            pix_to_met=scale*scale_factor, mask_pil=mask_pil)
    for index in range(len(scale)):
        scale[index] = scale[index]/scale_factor
    transf = calc_transform_matrix(path[0], angle_rad, scale, cropping_cfg["image_shape"])
    #     (init_coord, angle, scale, border_w=0):
    return np.asarray(crop_img), transf, scale, np.asarray(crop_mask)


def vis_pdf(image, distrib, transform_from_pix_to_m):
    dist_img = torch.zeros(image.shape[:2])
    for x in range(len(dist_img)):
        for y in range(len(dist_img[x])):
            coord = transform_from_pix_to_m@torch.tensor([x,y,1.], dtype=torch.float)
            dist_img[y,x] = torch.exp(distrib.log_prob(coord[:2]))
    return dist_img


def heatmap2d_withiImg(arr, img, filename = None):
    fig = plt.figure()
    plt.imshow(img/255, alpha=1)
    plt.imshow(arr.detach().cpu().numpy(), cmap='viridis', alpha=0.4)
    import io
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close()
    # if filename is None:
    #     plt.show()
    #     return
    # plt.savefig(filename)
    return img_arr

def Psidom_mesh(rows, cols):
    return np.dstack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'))


def vis_pdf2(image, distrib, transform_from_pix_to_m, index):

    grid = Psidom_mesh(image.shape[0],image.shape[1])
    grid = grid.reshape(image.shape[0]*image.shape[1], 2)[:, ::-1]
    grid = transform_points(grid,torch.inverse(transform_from_pix_to_m))
    tgrid = torch.tensor(grid,dtype=torch.float)
    mix = torch.distributions.Categorical(distrib.mixture_distribution.logits[index, :])
    mean = distrib.component_distribution.mean[index, :]
    cov_mat = distrib.component_distribution.covariance_matrix[index, :]
    distr = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov_mat)
    gmm = torch.distributions.MixtureSameFamily(mix, distr)
    tgrid = tgrid.to(mean.device)
    distr_image = torch.exp(gmm.log_prob(tgrid)).reshape(image.shape[:2])
    return distr_image


def vis_image_batched_Aleksander(images, distrs, raster_from_agent, goal_avail, tgt, num_to_vis=4):
    bs = images.shape[0]
    images = images.cpu().numpy()
    img = []
    for i in range(bs):
        if goal_avail[i]:
            import cv2
            tgt_ = transform_points(tgt[i, -1:, :], raster_from_agent[i])
            try:
                images[i] = cv2.ellipse(images[i].astype('float32'), ((int(tgt_[0, 0]), int(tgt_[0, 1])), (6, 6), 0), (40, 240, 50), thickness=-1)
            except Exception as E:
                print ("cv2.ellipse: " + str(E))
            # cv2.ellipse(images[i], (tgt[0], tgt[1]), (3, 3), 0, 0, 360, (100, 100, 100), -1)
            visualized_image = vis_image_Aleksander(images[i], distrs, raster_from_agent[i], tgt, index=i)


            img.append(visualized_image)
            num_to_vis -= 1
            if num_to_vis == 0:
                return img

def vis_sigma_ellipses(np_im, pose_rasters, covs):
    pil_im = Image.fromarray(np.asarray(np_im, dtype="uint8"))
    draw = ImageDraw.Draw(pil_im)
    for pose_raster, cov in zip(pose_rasters, covs):
        # pose_raster = data.raster_from_agent[i].numpy() @ np.array([predictions[i, 0], predictions[i, 1], 1.])
        r = 2
        draw.ellipse((pose_raster[0] - r, pose_raster[1] - r, pose_raster[0] + r, pose_raster[1] + r), fill='red',
                     outline='red')
        if cov is not None:
            draw = ImageDraw.Draw(pil_im, "RGBA")
            for j in range(1, 4):
                r_cov_x = np.array([j * np.sqrt(cov[0, 0]), 0, 0])
                r_cov_y = np.array([0, j * np.sqrt(cov[1, 1]), 0])
                r_cov_min = np.array([pose_raster[0] - r_cov_x[0],
                                      pose_raster[1] - r_cov_y[1],
                                      1])
                r_cov_max = np.array([pose_raster[0] + r_cov_x[0],
                                      pose_raster[1] + r_cov_y[1],
                                      1])
                draw.ellipse((r_cov_min[0], r_cov_min[1], r_cov_max[0], r_cov_max[1]),
                             fill=(100, 100, 170, 15),
                             outline=(200, 100, 170, 75))
    return np.asarray(pil_im, dtype="uint8")


def plot2dcov(img, mu, Sigma, color='k', nSigmas=[1, 2, 3]):
    """
    Plots a 2D covariance ellipse given the Gaussian distribution parameters.
    The function expects the mean and covariance matrix to ignore the theta parameter.
    :param mu: The mean of the distribution: 2x1 vector.
    :param Sigma: The covariance of the distribution: 2x2 matrix.
    :param color: The border color of the ellipse and of the major and minor axes.
    :param nSigma: The radius of the ellipse in terms of the number of standard deviations (default: 1).
    :param legend: If not None, a legend label to the ellipse will be added to the plot as an attribute.
    """
    pil_img = Image.fromarray(np.asarray(img * 255, dtype="uint8"))
    draw = ImageDraw.Draw(pil_img)
    mu = np.array(mu)
    assert mu.shape == (2,)
    Sigma = np.array(Sigma)
    assert Sigma.shape == (2, 2)
    n_points = 105
    A = np.linalg.cholesky(Sigma)
    angles = np.linspace(0, 2 * np.pi, n_points)

    for nSigma in nSigmas:
        x_old = nSigma * np.cos(angles)
        y_old = nSigma * np.sin(angles)
        x_y_old = np.stack((x_old, y_old), 1)
        x_y_new = np.matmul(x_y_old, np.transpose(A)) + mu.reshape(1, 2)  # (A*x)T = xT * AT
        #         plt.plot(x_y_new[:, 0], x_y_new[:, 1], color=color, label=legend)
        draw.point([(x_y_new[i, 0], x_y_new[i, 1]) for i in range(len(x_y_new))], fill="red")

    return np.asarray(pil_img, dtype="uint8")

def vis_image_Aleksander(image, distr, raster_from_agent, tgt, index):

    try:
        means_meters = distr.component_distribution.mean.detach().cpu().numpy()[index]
        cov_meters = distr.component_distribution.covariance_matrix.detach().cpu().numpy()[index]
    except:
        means_meters = distr.mean.unsqueeze(1).detach().cpu().numpy()[index]
        cov_meters = distr.covariance_matrix.unsqueeze(1).detach().cpu().numpy()[index]
    pose_rasters = transform_points(means_meters, raster_from_agent)
    scale_raster_from_agent = raster_from_agent.clone()
    scale_raster_from_agent[:2, 2] = 0
    covs = transform_points(cov_meters, raster_from_agent)
    # for pose_raster, cov in zip(pose_rasters, covs):
    #     image = plot2dcov(image, pose_raster, cov)
    image = vis_sigma_ellipses(image, pose_rasters, covs)

    dist_image = vis_pdf2(image, distr, raster_from_agent, index=index)
    img_arr = heatmap2d_withiImg(dist_image, image)
    return img_arr[:,:, :3]

if __name__ == '__main__':
    image = torch.rand((224, 224, 3), dtype=torch.float)
    transform_from_pix_to_m =torch.eye(3)
    distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.rand(2)+100, torch.eye(2)*50)
    img_arr = vis_image_Aleksander(image, distr, transform_from_pix_to_m)
    plt.imshow(img_arr)
    plt.show()