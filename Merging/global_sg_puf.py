from math import floor

import numpy as np
import open3d as o3d

from puf_utils import PUF_SG
from global_sg import _rels_to_sparse
from spatial_prior import SpatialPrior

# Global 3D scene graph with PUF uncertainty-aware fusion
class GlobalSG_PUF:
    def __init__(self, args, num_classes=20, num_rel_classes=7, visualize=False):
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes

        # Build spatial prior if requested
        sp = None
        if getattr(args, 'use_spatial_prior', False):
            sp = SpatialPrior(
                num_obj_classes=num_classes,
                num_rel_classes=num_rel_classes,
                class_prior_path=getattr(args, 'class_prior_path', None),
                alpha_strength=getattr(args, 'alpha_strength', 1.0),
                sigma_d=getattr(args, 'sigma_d', 1.0),
            )

        self.global_group = PUF_SG(
            num_rel_class=num_rel_classes,
            num_obj_class=num_classes,
            alpha_prior=args.dirichlet_alpha_prior,
            spatial_prior=sp,
            completion_threshold=getattr(args, 'completion_threshold', 0.1),
            obs_strength=getattr(args, 'obs_strength', 1.0),
            obj_obs_strength=getattr(args, 'obj_obs_strength', 1.0),
            lambda_birth=getattr(args, 'lambda_birth', 0.1),
            tau_birth=getattr(args, 'tau_birth', 0.5),
            beta_min=getattr(args, 'beta_min', 0.05),
            likelihood_sigma_jsd=getattr(args, 'likelihood_sigma_jsd', 1.0),
            l2_gate=getattr(args, 'l2_gate', 3.0),
            alpha_min=getattr(args, 'alpha_min', 0.0),
        )
        self.visualize = visualize
        if visualize:
            self.cur_obj = []
            self.cur_rel = []

    def update(self, classes, bboxes, rels, rel_classes, depth, camera_rot, camera_trans, camera_intrinsic, rel_probs=None, class_probs=None):
        if len(classes) == 0:
            if self.visualize:
                self.cur_obj.append({"classes": self.global_group.classes.copy(), "means": self.global_group.means.copy(), "covs": self.global_group.covs.copy()})
                self.cur_rel.append(_rels_to_sparse(self.global_group.rels))
            return

        camera_rot = camera_rot[None, ...]  # (1, 3, 3)
        camera_trans = camera_trans[None, :, None]  # (1, 3, 1)

        # Filter out objects with invalid depth
        invalid_depth = depth[bboxes[:, 1], bboxes[:, 0]] == 0
        classes = classes[~invalid_depth]
        bboxes = bboxes[~invalid_depth]
        if class_probs is not None:
            class_probs = class_probs[~invalid_depth]
        if len(rels) > 0:
            new_idx = np.cumsum(~invalid_depth) - 1
            new_idx[invalid_depth] = -1
            valid_edge_idx = np.logical_and(new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1)
            rel_classes = rel_classes[valid_edge_idx]
            if rel_probs is not None:
                rel_probs = rel_probs[valid_edge_idx]
            rels = new_idx[rels[valid_edge_idx]]

        if len(classes) == 0:
            return

        # Project 2D to 3D — Eq. (2)
        x = (bboxes[:, 0] - camera_intrinsic.cx) / camera_intrinsic.fx
        y = (bboxes[:, 1] - camera_intrinsic.cy) / camera_intrinsic.fy
        z = np.ones_like(x)
        depth_val = depth[bboxes[:, 1], bboxes[:, 0], None]  # (num_objects, 1)
        camera_coord = (np.stack((x, y, z), axis=-1) * depth_val)[..., None]  # (num_objects, 3, 1)
        mean_3d = (camera_rot @ camera_coord + camera_trans).squeeze(-1)  # (num_objects, 3)

        # Eq. (1)
        zeros = np.zeros_like(bboxes[:, 0])
        cov_2d = np.stack((bboxes[:, 2] ** 2, zeros, zeros, bboxes[:, 3] ** 2), axis=-1).reshape(-1, 2, 2) / 12

        # Eq. (3)
        J = np.array([[[camera_intrinsic.fx, 0, 0],
                       [0, camera_intrinsic.fy, 0]]]).repeat(len(bboxes), axis=0) / depth_val[..., None]
        J[:, 0, 2] = -x * camera_intrinsic.fx / (depth_val[:, 0] ** 2)
        J[:, 1, 2] = -y * camera_intrinsic.fy / (depth_val[:, 0] ** 2)
        J_inv = np.linalg.pinv(J)

        cov_3d = J_inv @ cov_2d @ J_inv.transpose(0, 2, 1)  # Eq. (5)
        cov_3d[:, 2, 2] += (cov_3d[:, 0, 0] + cov_3d[:, 1, 1]) / 2  # Eq. (6)
        cov_3d = camera_rot @ cov_3d @ camera_rot.transpose(0, 2, 1)  # Eq. (7)

        # Extract middle 50% x 50% of the bounding boxes for point cloud
        bboxes_xyxy_50 = np.concatenate((bboxes[:, :2] - bboxes[:, 2:] / 4, bboxes[:, :2] + bboxes[:, 2:] / 4), axis=1)
        proj_coords = []
        proj_count = []
        for x1, y1, x2, y2 in bboxes_xyxy_50:
            for px in range(int(x1), int(x2), 50):
                for py in range(int(y1), int(y2), 50):
                    proj_coords.append([px, py])
            proj_count.append((floor((int(x2) - int(x1)) / 50) + 1) * (floor((int(y2) - int(y1)) / 50) + 1))
        proj_coords = np.array(proj_coords)
        if len(proj_coords) == 0:
            proj_coords = np.zeros((0, 2), dtype=int)
        px = (proj_coords[:, 0] - camera_intrinsic.cx) / camera_intrinsic.fx
        py = (proj_coords[:, 1] - camera_intrinsic.cy) / camera_intrinsic.fy
        pz = np.ones_like(px)
        depth_val_pcd = depth[proj_coords[:, 1], proj_coords[:, 0], None]
        camera_coord_pcd = (np.stack((px, py, pz), axis=-1) * depth_val_pcd)[..., None]
        world_coord = (camera_rot @ camera_coord_pcd + camera_trans).squeeze(-1)
        pcds = np.split(world_coord, np.cumsum(proj_count)[:-1])

        # Add local 3D SG to global 3D SG
        update_idx = self.global_group.add(classes, mean_3d, cov_3d, rels, rel_classes, pcds, rel_probs=rel_probs, class_probs=class_probs)

        # Merge observation nodes into global graph
        self.global_group.fuse(update_idx)

        if self.visualize:
            self.cur_obj.append({"classes": self.global_group.classes.copy(), "means": self.global_group.means.copy(), "covs": self.global_group.covs.copy()})
            self.cur_rel.append(_rels_to_sparse(self.global_group.rels))

