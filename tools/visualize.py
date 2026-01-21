import os
import numpy as np
import pyvista as pv
from tqdm import tqdm
import multiprocessing
from functools import partial


""" class names:
'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
'pole', 'traffic-sign'
"""
colors = np.array(
    [
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255],
    ]
).astype(np.uint8)


def get_coords():
    resolution = 0.2
    xx = np.arange(0, 256 + 1)
    yy = np.arange(0, 256 + 1)
    zz = np.arange(0, 32 + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(xx[:-1], yy[:-1], zz[:-1])
    coords_grid = np.array([xx, yy, zz])
    coords_grid = coords_grid.transpose([1, 2, 3, 0])
    coords_grid = (coords_grid * resolution) + resolution / 2
    coords_grid = coords_grid + vox_origin.reshape([1, 1, 1, 3])
    return coords_grid


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def _add_camera_wireframe_pyvista(plotter, T_velo_2_cam, vox_origin, img_size, f, d):
    """Minimal replacement of mlab.triangular_mesh wireframe camera."""
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ],
        dtype=np.float32,
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1), dtype=np.float32)])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T

    pts = tri_points[:, :3]
    pts[:, 0] -= vox_origin[0]
    pts[:, 1] -= vox_origin[1]
    pts[:, 2] -= vox_origin[2]

    # triangles (faces)
    triangles = [(0, 1, 2), (0, 1, 4), (0, 3, 4), (0, 2, 3)]
    faces = []
    for a, b, c in triangles:
        faces.extend([3, a, b, c])
    faces = np.array(faces, dtype=np.int64)

    cam_mesh = pv.PolyData(pts, faces)
    plotter.add_mesh(cam_mesh, color="black", style="wireframe", line_width=5)


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
    save_name=None,
    save_root=None,
    video_view=False,
):
    if video_view:
        window_size = (2000, 1000)
    else:
        window_size = (2000, 1500)
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    pv.global_theme.multi_samples = 0  # helps headless stability
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    # Draw the camera
    if T_velo_2_cam is not None:
        _add_camera_wireframe_pyvista(plotter, T_velo_2_cam, vox_origin, img_size, f, d)

    # Build cube glyph
    cube = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)

    # Inside FOV voxels (full color)
    if fov_voxels.shape[0] > 0:
        pts = fov_voxels[:, :3].astype(np.float32)
        lab = fov_voxels[:, 3].astype(np.int32)  # 1..19
        rgba = colors[lab - 1]  # (N,4) uint8

        cloud = pv.PolyData(pts)
        cloud["rgba"] = rgba

        glyphs = cloud.glyph(geom=cube, scale=False, orient=False)
        plotter.add_mesh(glyphs, scalars="rgba", rgba=True, show_scalar_bar=False)

    # Outside FOV voxels (dimmed color)
    if outfov_voxels.shape[0] > 0:
        pts = outfov_voxels[:, :3].astype(np.float32)
        lab = outfov_voxels[:, 3].astype(np.int32)  # 1..19

        outfov_colors = colors.copy()
        outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
        rgba = outfov_colors[lab - 1]

        cloud = pv.PolyData(pts)
        cloud["rgba"] = rgba
        glyphs = cloud.glyph(geom=cube, scale=False, orient=False)
        plotter.add_mesh(glyphs, scalars="rgba", rgba=True, show_scalar_bar=False)


    if video_view:
        plotter.camera.position = [-96.17897208968986, 24.447806140326282, 71.4786454057558]
        plotter.camera.focal_point = [25.59999984735623, 25.59999984735623, 2.1999999904073775]
        plotter.camera.up = [0.4945027163799531, -0.004902474180369383, 0.8691622571417599]
        plotter.camera.view_angle = 23.999999999999993
    else:
        plotter.camera.position = (-50.907238103376244, -51.31911151935225, 104.75510851395386)
        plotter.camera.focal_point = (23.005321731256945, 23.263153155247394, 0.7241134057028675)
        plotter.camera.up = (0.5286546999662366, 0.465851763212298, 0.7095818084728509)
        plotter.camera.view_angle = 19.199999999999996
    
    # Render & save
    os.makedirs(save_root, exist_ok=True)
    save_file = save_name + ".png"
    plotter.show(screenshot=os.path.join(save_root, save_file), auto_close=True)

    return save_file


def get_fov_mask(transform, intr):
    xv, yv, zv = np.meshgrid(range(256), range(256), range(32), indexing="ij")
    vox_coords = (
        np.concatenate(
            [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
        )
        .astype(int)
        .T
    )
    vox_size = 0.2
    offsets = np.array([0.5, 0.5, 0.5]).reshape(1, 3)
    vol_origin = np.array([0, -25.6, -2])
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = vox_coords * vox_size + vox_size * offsets + vol_origin.reshape(1, 3)
    cam_pts = np.hstack([cam_pts, np.ones((len(cam_pts), 1), dtype=np.float32)])
    cam_pts = np.dot(transform, cam_pts.T).T

    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = cam_pts[:, 0:2]
    pix[:, 0] = np.round((pix[:, 0] * fx) / cam_pts[:, 2] + cx).astype(int)
    pix[:, 1] = np.round((pix[:, 1] * fy) / cam_pts[:, 2] + cy).astype(int)
    pix = pix.astype(np.int32)

    pix_z = cam_pts[:, 2]
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    fov_mask = np.logical_and(
        pix_x >= 0,
        np.logical_and(
            pix_x < 1280,
            np.logical_and(pix_y >= 0, np.logical_and(pix_y < 384, pix_z > 0)),
        ),
    )
    return fov_mask


def process_voxel(
    pred_voxel,
    cam_param,
    write_root,
    pred_seq,
    ann_file,
    lidar2cam,
    fov_mask,
    save_name,
    vis_gt=True,
    video_view=False,
):
    save_root = os.path.join(write_root, pred_voxel.split(".")[0])
    pred = np.fromfile(os.path.join(pred_seq, pred_voxel), dtype=np.uint16)
    occ_pred = pred.reshape(256, 256, 32).astype(np.uint16)
    occ_pred[occ_pred == 255] = 0

    vox_origin = np.array([0, -25.6, -2])
    os.makedirs(save_root, exist_ok=True)

    _ = draw(
        occ_pred,
        lidar2cam,
        vox_origin,
        fov_mask,
        img_size=cam_param["img_size"],
        f=cam_param["f"],
        voxel_size=0.2,
        d=cam_param["d"],
        save_name=save_name,
        save_root=save_root,
        video_view=video_view,
    )
    
    if vis_gt:
        occ_gt = np.load(os.path.join(ann_file, pred_voxel.split('.')[0] + '_1_1.npy')).astype(np.uint16)
        gt_img = draw(occ_gt, lidar2cam, vox_origin, fov_mask,
                    img_size=cam_param['img_size'], f=cam_param['f'], voxel_size=0.2, d=cam_param['d'],
                    save_name='gt', save_root=save_root, video_view=video_view)


if __name__ == "__main__":
    pv.start_xvfb(wait=0.5)

    sequence = "08"
    data_root = "datasets/SemanticKITTI/dataset/sequences"
    ann_file = f"datasets/SemanticKITTI/dataset/labels/{sequence}"

    pred_seq = f"pred/sequences/{sequence}/predictions"

    cam_param = {
        "img_size": (1220, 370),
        "f": 707.0912,
        "d": 7,
    }
    vox_origin = np.array([0, -25.6, -2])

    write_root = pred_seq.replace("predictions", "visualizations")

    calib_file = os.path.join(data_root, sequence, "calib.txt")
    calib_all = {}
    with open(calib_file, "r") as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    intrin = np.identity(4)
    intrin[:3, :4] = calib_all["P2"].reshape(3, 4)
    lidar2cam = np.identity(4)  # 4x4 matrix
    lidar2cam[:3, :4] = calib_all["Tr"].reshape(3, 4)
    fov_mask = get_fov_mask(lidar2cam, intrin)

    pred_voxels = os.listdir(pred_seq)
    pred_voxels.sort()
    save_name = "FoundationSSC"

    # multi process
    worker_func = partial(
        process_voxel,
        cam_param=cam_param,
        write_root=write_root,
        pred_seq=pred_seq,
        ann_file=ann_file,
        lidar2cam=lidar2cam,
        fov_mask=fov_mask,
        save_name=save_name,
        vis_gt=True,
        video_view=True,
    )
    with multiprocessing.Pool(processes=16) as pool:
        _ = list(tqdm(pool.imap(worker_func, pred_voxels), total=len(pred_voxels)))
    
    # single process (for debug)
    # for pred_voxel in tqdm(pred_voxels):
    #     process_voxel(
    #         pred_voxel,
    #         cam_param,
    #         write_root,
    #         pred_seq,
    #         ann_file,
    #         lidar2cam,
    #         fov_mask,
    #         save_name,
    #         vis_gt=True,
    #         video_view=False,
    #     )
