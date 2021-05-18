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
import kornia
import cv2
try:
    from transformations import Resize, AddBorder, Rotate, Crop
except:
    from .transformations import Resize, AddBorder, Rotate, Crop

def preprocess_data(data, cfg, device="cpu") -> torch.tensor:
    imgs_tensor = (torch.tensor(data.image, device=device, dtype=torch.float32).permute(0, 3, 1, 2))/255

    mask_tensor = None
    if data.segm is not None:
        mask_tensor = (torch.tensor(data.segm.squeeze(), device=device, dtype=torch.float32).unsqueeze(1)/255)
        # imgs_tensor = torch.cat((imgs_tensor, mask_tensor), dim=1)
    bs = imgs_tensor.shape[0]
    cropping_points = data.cropping_points
    bboxes = torch.zeros((bs, 4, 2), device=device)
    bboxes[:, 0, :] = torch.tensor([cropping_points[:, 0], cropping_points[:, 1]], device=device).permute(1, 0)
    # bboxes[:, 0, :] = torch.stack([torch.tensor(cropping_points[:, 0]), torch.tensor(cropping_points[:, 1])], dim=1)
    bboxes[:, 1, :] = torch.tensor([cropping_points[:, 2], cropping_points[:, 1]], device=device).permute(1, 0)
    bboxes[:, 2, :] = torch.tensor([cropping_points[:, 2], cropping_points[:, 3]], device=device).permute(1, 0)
    bboxes[:, 3, :] = torch.tensor([cropping_points[:, 0], cropping_points[:, 3]], device=device).permute(1, 0)
    transf = torch.tensor(data.map_affine, device=device)[:, :2, :].float()
    new_size = cfg["cropping_cfg"]["image_shape"]
    dst_bboxes = torch.tensor([[[0., 0], [new_size[0], 0], [new_size[0], new_size[1]], [0, new_size[1]]]], device=device).repeat(bs, 1, 1)
    flags = {}
    flags['interpolation'] = torch.tensor([0]).to(device)
    flags['align_corners'] = torch.ones(1).bool().to(device)

    params = {"src": bboxes,
              "dst": dst_bboxes
              }

    imgs = kornia.warp_affine(imgs_tensor, transf, mode="nearest", dsize=imgs_tensor.shape[2:])
    imgs = kornia.apply_crop(imgs, params, flags)



    if data.segm is not None:
        masks = kornia.warp_affine(mask_tensor, transf, mode="nearest", dsize=imgs_tensor.shape[2:])
        mask_tensor = kornia.apply_crop(masks, params, flags)
    return imgs, mask_tensor


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


def sort_neigh_future(unsorted: List[np.array]):
    '''
        :param unsorted - List of neighb observations.
                   neighb observations is np.array of shape X(x<=obs_len, 4)
        :return: "sorted" List (of np.arrays) with len of timestamps of observation(<= obs.len)
                          consist of np.arrays with shape of (numb_agents at given timestamp, 4)
    '''
    unsorted = np.array(unsorted)
    time_sorted = np.zeros((12, len(unsorted), 4)) - 1
    unique_tss = np.array(np.unique(unsorted[:, :, 0]))
    unique_tss = unique_tss[unique_tss != -1]  # [::-1]
    for i, ts in enumerate(unique_tss):
        poses_at_ts = unsorted[unsorted[:, :, 0] == ts]
        time_sorted[i, :len(poses_at_ts)] = poses_at_ts
    return time_sorted


def trajectory_orientation(last_pose, prev_pose):
    dy = last_pose[1] - prev_pose[1]
    dx = last_pose[0] - prev_pose[0]
    angle = math.atan2(dy, dx) / math.pi * 180.0
    return angle


def rotate_image(img: PIL.Image, angle, center: Union[np.array, list, tuple], mask: PIL.Image = None):
    # new_img = img.rotate(angle, center=(center[0], center[1]))
    new_mask = None
    # if mask is not None:
    #     new_mask = mask.rotate(angle, center=(center[0], center[1]))
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
    return None, agent_pix_to_img @ np.linalg.inv(map_to_local) @ img_to_agentpix, None


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
    tl_disp = rot_mat @ (np.array([orientation * cfg["agent_center"][0] * cfg["image_area_meters"][0],
                                   -cfg["agent_center"][1] * cfg["image_area_meters"][1],
                                   1]))

    tl_m = (agent_center_m) + tl_disp

    br_displ = rot_mat @ (np.array([orientation * (-1 + cfg["agent_center"][0]) * cfg["image_area_meters"][0],
                                    (1 - cfg["agent_center"][1]) * cfg["image_area_meters"][1],
                                    1]))
    br_m = (agent_center_m) + br_displ

    tl = (np.linalg.inv(transform) @ np.array([tl_m[0], tl_m[1], 1]))[:2]
    br = (np.linalg.inv(transform) @ np.array([br_m[0], br_m[1], 1]))[:2]

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
    # size = img.size
    # np_img = np.asarray(img)

    # position of [agent_x - 0.25area_x, agent_y - 0.5area_y] in meters
    tl_meters = pix_to_met @ (np.append(agent_center, 1)) - np.array([cfg["agent_center"][0] * cfg["image_area_meters"][1],
                          cfg["agent_center"][1] * cfg["image_area_meters"][0],
                          0])

    # position of [agent_x + 0.75area_x, agent_y + 0.5area_y]
    br_meters = pix_to_met @ (np.append(agent_center, 1)) + np.array([(1 - cfg["agent_center"][0]) * cfg["image_area_meters"][1],
                          (1 - cfg["agent_center"][1]) * cfg["image_area_meters"][0],
                          0])

    tl_pix = np.linalg.inv(pix_to_met) @ tl_meters
    br_pix = np.linalg.inv(pix_to_met) @ br_meters
    # assert np.allclose((br_meters[:2] - tl_meters[:2]),
    #                    np.array([cfg["image_area_meters"][0], cfg["image_area_meters"][1]]))

    # tl_x = max(0, agent_center[1] - cfg["agent_center"][1] * cfg["image_area_meters"][0] / pix_to_met[0,0])
    # tl_y = max(0, agent_center[0] - cfg["agent_center"][0] * cfg["image_area_meters"][1] / pix_to_met[0,0])
    # br_x = max(0, agent_center[1] + (1 - cfg["agent_center"][1]) * cfg["image_area_meters"][0] / pix_to_met[0,0])
    # br_y = max(0, agent_center[0] + (1 - cfg["agent_center"][0]) * cfg["image_area_meters"][1] / pix_to_met[0,0])
    # assert tl_x * tl_y * br_y * br_x != 0
    image_resize_coef = [cfg["image_shape"][0] / abs(tl_pix[1] - br_pix[1]), cfg["image_shape"][1] / abs(tl_pix[0] - br_pix[0])]

    # cropped = img.crop((tl_y, tl_x, br_y, br_x))
    # mask_pil_cropped = mask_pil.crop((tl_y, tl_x, br_y, br_x))
    # image_resize_coef = [cfg["image_shape"][0] / cropped.size[0], cfg["image_shape"][1] / cropped.size[1]]

    # cropped = cropped.resize(cfg["image_shape"])
    # mask_pil_cropped = mask_pil_cropped.resize(cfg["image_shape"], 0)

    # img[int(tl_pix[1]):int(br_pix[1]), int(tl_pix[0]):int(br_pix[0])]
    return None, image_resize_coef, None, (int(tl_pix[1]), int(tl_pix[0]), int(br_pix[1]), int(br_pix[0]))


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
                        cropping_cfg=cropping_cfg, file=None, mask=None, scale_factor=1, transform=None, neighb_hist=None,
                        neighb_hist_avail=None):
    # print(img.dtype)
    # img_pil = Image.fromarray(np.asarray(img, dtype="uint8"))
    # mask_pil = Image.fromarray(np.asarray(mask, dtype="uint8"))
    # draw = ImageDraw.Draw(img_pil)
    # scale_factor = 2
    scaled_border = border_width // scale_factor


    draw_h(draw_traj, img, path, transform, neighb_hist[:,:,2:], neighb_hist_avail)
    # img_b, mask_b = expand(border, img, mask)
    # mask_pil = ImageOps.expand(mask_pil, (border, border))

    angle_deg = trajectory_orientation(path[0], path[1])
    if np.linalg.norm(path[1] - np.array([-1., -1.])) < 1e-6:
        angle_deg = 0
    angle_rad = angle_deg / 180 * math.pi
    # agent_center = path[0] / scale_factor + scaled_border
    agent_center = (transform @ np.append(path[0], 1))[:2]
    rotate_operator = Rotate(angle_deg, agent_center)
    tm = rotate_operator.transformation_matrix

    # scale -> from image_pix (original) to meters
    scale = pix_to_m_cfg[file]["scale"]
    pix_to_meters = (np.eye(3) * scale)
    pix_to_meters[2, 2] = 1
    # transform -> from image to numpy

    # pix_to_meters from numpy_pix to meters =  (orig_im to meters)  @  (numpy_im to orig_image)
    pix_to_meters = pix_to_meters @ np.linalg.inv(transform)

    crop_img, scale, crop_mask, (tl_y, tl_x, br_y, br_x) = crop_image(img, cropping_cfg,
                                                                      agent_center=agent_center,
                                                                      pix_to_met=pix_to_meters,
                                                                      mask_pil=mask)

    scale_reshaping = np.eye(3)
    scale_reshaping[:2, :2] = scale_reshaping[:2, :2] * np.array(scale)
    # for index in range(len(scale)):
    #     scale[index] = scale[index] / scale_factor
    # transf = calc_transform_matrix(path[0], angle_rad, scale, cropping_cfg["image_shape"])
    cr_operator = Crop((tl_x, tl_y), (br_x, br_y))
    cr_tm = cr_operator.transformation_matrix
    transf = scale_reshaping @ cr_tm @ tm
    #     (init_coord, angle, scale, border_w=0):
    return img, transf, scale_reshaping, mask, tm, (tl_y, tl_x, br_y, br_x), tm


def expand(border, img, mask):
    sh = img.shape
    img_b = np.zeros((sh[0] + 2 * border, sh[1] + 2 * border, sh[2]), dtype=np.float32)
    img_b[border:-border, border:-border] = img
    # img_pil = ImageOps.expand(img_pil, (border, border))
    sh = mask.shape
    mask_b = np.zeros((sh[0] + 2 * border, sh[1] + 2 * border), dtype=np.float32)
    mask_b[border:-border, border:-border] = mask
    return img_b, mask_b


def draw_h(draw_traj, img, path, transform, neighb_hist=None, neighb_hist_avail=None):
    if draw_traj:
        R = 1
        for pose in path:
            if np.linalg.norm(pose - np.array([-1., -1.])) > 1e-6:
                new_pose = transform @ np.array([pose[0], pose[1], 1])
                cv2.circle(img, (int(new_pose[0]), int(new_pose[1])), R, (0, 0, 255), -1)
        if neighb_hist is not None:
            for ped_id, ped in enumerate(neighb_hist):
                for ts, pose in enumerate(ped):
                    if neighb_hist_avail[ped_id, ts]:
                        new_pose = transform @ np.array([pose[0], pose[1], 1])
                        cv2.circle(img, (int(new_pose[0]), int(new_pose[1])), R, (0, int(205*((ts+1))/8), int(155*((ts+1))/8)), -1)


def vis_pdf(image, distrib, transform_from_pix_to_m):
    dist_img = torch.zeros(image.shape[:2])
    for x in range(len(dist_img)):
        for y in range(len(dist_img[x])):
            coord = transform_from_pix_to_m @ torch.tensor([x, y, 1.], dtype=torch.float)
            dist_img[y, x] = torch.exp(distrib.log_prob(coord[:2]))
    return dist_img


def heatmap2d_withiImg(arr, img, alpha1 = 1.0, alpha2 = 0.4,gamma = 0.0, filename=None):
    fig = plt.figure()
    # return cv2.addWeighted(arr.permute(2,3,1).contiguous(), alpha1, img, alpha2, gamma)
    plt.imshow(img / 255, alpha=1)
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
    grid = Psidom_mesh(image.shape[0], image.shape[1])
    grid = grid.reshape(image.shape[0] * image.shape[1], 2)[:, ::-1]
    grid = transform_points(grid, torch.inverse(transform_from_pix_to_m))
    tgrid = torch.tensor(grid, dtype=torch.float)
    mix = torch.distributions.Categorical(distrib.mixture_distribution.logits[index, :])
    mean = distrib.component_distribution.mean[index, :]
    cov_mat = distrib.component_distribution.covariance_matrix[index, :]
    distr = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov_mat)
    gmm = torch.distributions.MixtureSameFamily(mix, distr)
    tgrid = tgrid.to(mean.device)
    distr_image = torch.exp(gmm.log_prob(tgrid)).reshape(image.shape[:2])
    return distr_image


def vis_image_batched_Aleksander(images, distrs, raster_from_agent, goal_avail, tgt, num_to_vis=4):
    '''
    plot batch of multicomponent distribution on images

    :params images: original 2d images  [bs,h,w,channel] channel=3
    :params distr: torch.distributions.multivariate_normal.MultivariateNormal(  
                        Categorical(probs: torch.Size([bs, num_distr]), logits: torch.Size([bs, num_distr])),
                        MultivariateNormal(loc: torch.Size([bs, num_distr, 2]), covariance_matrix: torch.Size([bs, num_distr, 2, 2])))
    :params raster_from_agent: transform matrix [bs,3,3]
    :params goal_avail: mask of batc for plot [bs](bool)
    :params tgt: optional points to drowing green circle
    :params index: index of bs to visualise    
    :params num_to_vis: count of img to be created

    :return images: that function return original image with added visualisation distribution layer  [bs/num_to_vis,h,w,channel]
    '''
    bs = images.shape[0]
    images = images.cpu().numpy()
    img = []
    for i in range(bs):
        if goal_avail[i]:

            try:
                tgt_ = transform_points(tgt[i, -1:, :], raster_from_agent[i])
                # cv2.ellipse((images[i] * 255).astype(np.uint8), center=(int(tgt_[0, 0]), int(tgt_[0, 1])), axes=(6, 6),
                #             angle=0, startAngle=0, endAngle=360, color=(40, 240, 50), thickness=-1)
                images[i] = cv2.ellipse(images[i].astype(np.float32).copy(), ((int(tgt_[0, 0]), int(tgt_[0, 1])), (6, 6), 0),
                                        (0.2, 0.9, 0.2), thickness=-1)
            except Exception as E:
                print("cv2.ellipse: " + str(E))
            # cv2.ellipse(images[i], (tgt[0], tgt[1]), (3, 3), 0, 0, 360, (100, 100, 100), -1)
            visualized_image = vis_image_Aleksander(images[i], distrs, raster_from_agent[i], tgt, index=i)

            img.append(visualized_image)
            num_to_vis -= 1
            if num_to_vis == 0:
                return img


def vis_sigma_ellipses(np_im, pose_rasters, covs):
    pil_im = Image.fromarray(np.asarray(np_im*255, dtype="uint8"))
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


def vis_image_Aleksander(image, distr, raster_from_agent, tgt = None,  index = 0):
    '''
    plot multicomponent distribution on image

    :params image: original 2d image  [h,w,channel] channel=3
    :params distr: torch.distributions.multivariate_normal.MultivariateNormal(  
                        Categorical(probs: torch.Size([bs, num_distr]), logits: torch.Size([bs, num_distr])),
                        MultivariateNormal(loc: torch.Size([bs, num_distr, 2]), covariance_matrix: torch.Size([bs, num_distr, 2, 2])))
    :params raster_from_agent: transform matrix [3,3]
    :params tgt: optional hz
    :params index: index of bs to visualise    

    :return image: that function return original image with added visualisation distribution layer  
    '''
    try:
        means_meters = distr.component_distribution.mean.detach().cpu().numpy()[index]
        cov_meters = distr.component_distribution.covariance_matrix.detach().cpu().numpy()[index]
    except:
        means_meters = distr.mean.unsqueeze(1).detach().cpu().numpy()[index]
        cov_meters = distr.covariance_matrix.unsqueeze(1).detach().cpu().numpy()[index]
    pose_rasters = transform_points(means_meters, raster_from_agent)
    scale_raster_from_agent = raster_from_agent.clone()
    scale_raster_from_agent[:2, 2] = 0
    # covs = transform_points(transform_points(cov_meters, raster_from_agent))

    if distr.mean.shape[0] == 1:
        # mean_pix = torch.cat((distr.mean.detach(), torch.tensor([[0]])), dim=1)
        mean_pix = transform_points(distr.mean.detach(), raster_from_agent)
        image = cv2.ellipse(image.numpy(), ((int(mean_pix[0, 0]), int(mean_pix[0, 1])), (10, 10), 0),
                    (220, 0, 150), thickness=-1)
        image = torch.tensor(image)

        # draw.ellipse()
    # for pose_raster, cov in zip(pose_rasters, covs):
    #     image = plot2dcov(image, pose_raster, cov)
    # image = vis_sigma_ellipses(image, pose_rasters, covs)

    dist_image = vis_pdf2(image, distr, raster_from_agent, index=index)
    cmap = plt.get_cmap('jet')
    dist_image = cmap(2.5*dist_image.detach().numpy())
    img_arr = torch.clamp(0.7*image/255 + 0.5 * dist_image[:, :, :3], max=1)
    # img_arr = heatmap2d_withiImg(dist_image, image)
    return img_arr[:, :, :3]


if __name__ == '__main__':
    bs = 10
    num_distr = 3
    images = torch.rand((bs,224, 224, 3), dtype=torch.float)
    raster_from_agent = torch.eye(3)
    mean_predictions = ((torch.rand((bs,num_distr,2)))*224)
    cov_matrix = torch.eye(2).unsqueeze(0).repeat(num_distr,1,1).unsqueeze(0).repeat(bs,1,1,1) * 50
    mix = torch.distributions.Categorical(torch.rand(bs,num_distr))
    distr = torch.distributions.multivariate_normal.MultivariateNormal(mean_predictions, cov_matrix)
    gmm = torch.distributions.MixtureSameFamily(mix, distr)

    img_arr = vis_image_Aleksander(images[0], gmm, raster_from_agent, index =1)
    plt.imshow(images[0])
    plt.show()
    # plt.savefig("orig.png")
    plt.imshow(img_arr)
    plt.show()
    # plt.savefig("origPlusDistr.png")
    goal_avail = torch.ones(bs)
    tgt = torch.rand(bs,12,2)+100
    raster_from_agent = raster_from_agent.repeat(bs,1,1)
    imgs = vis_image_batched_Aleksander(images,gmm,raster_from_agent,goal_avail,tgt)
    plt.imshow(images[0])
    # plt.savefig("orig.png")
    plt.show()
    plt.imshow(imgs[0])
    plt.show()
    # plt.savefig("origPlusDistr.png")
    


