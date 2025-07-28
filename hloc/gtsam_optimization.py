import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

from . import logger

# Try to import GTSAM - fallback to SciPy if not available
try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    warnings.warn("GTSAM not available. Using SciPy-based optimization fallback.")
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation


class GTSAMOptimizer:
    """GTSAM-based pose graph optimization for SfM reconstruction"""
    
    def __init__(self, 
                 huber_threshold: float = 1.0,
                 max_iterations: int = 100,
                 relative_error_tol: float = 1e-5):
        
        self.huber_threshold = huber_threshold
        self.max_iterations = max_iterations  
        self.relative_error_tol = relative_error_tol
        
        if not GTSAM_AVAILABLE:
            logger.warning("GTSAM not available. Using fallback optimization.")
            
    def optimize_poses(self, 
                      poses: Dict[str, Dict],
                      pairwise_poses: List[Dict],
                      fixed_pose_id: Optional[str] = None) -> Dict[str, Dict]:
        """
        Optimize camera poses using pose graph optimization
        
        Args:
            poses: Dictionary of initial poses {image_id: {"R": rotation, "t": translation}}
            pairwise_poses: List of pairwise pose constraints
            fixed_pose_id: ID of pose to fix as reference (None = auto-select first)
            
        Returns:
            Optimized poses dictionary
        """
        
        if GTSAM_AVAILABLE:
            return self._gtsam_optimize_poses(poses, pairwise_poses, fixed_pose_id)
        else:
            return self._scipy_optimize_poses(poses, pairwise_poses, fixed_pose_id)
            
    def _gtsam_optimize_poses(self, poses, pairwise_poses, fixed_pose_id):
        """GTSAM-based pose graph optimization"""
        
        # Create pose graph
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        # Convert poses to GTSAM format and add to initial estimate
        pose_ids = list(poses.keys())
        if fixed_pose_id is None:
            fixed_pose_id = pose_ids[0]
            
        for i, (image_id, pose_data) in enumerate(poses.items()):
            R = pose_data["R"]
            t = pose_data["t"].flatten()
            
            # Convert to GTSAM pose
            rotation = gtsam.Rot3(R)
            translation = gtsam.Point3(t[0], t[1], t[2])
            pose = gtsam.Pose3(rotation, translation)
            
            initial_estimate.insert(i, pose)
            
        # Add pairwise constraints
        noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.huber_threshold),
            gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 0.1)
        )
        
        for constraint in pairwise_poses:
            id1, id2 = constraint["image_ids"]
            idx1 = pose_ids.index(id1)
            idx2 = pose_ids.index(id2)
            
            R_rel = constraint["R"]
            t_rel = constraint["t"].flatten()
            
            # Convert relative pose to GTSAM
            rel_rotation = gtsam.Rot3(R_rel)
            rel_translation = gtsam.Point3(t_rel[0], t_rel[1], t_rel[2])
            relative_pose = gtsam.Pose3(rel_rotation, rel_translation)
            
            # Add between factor
            factor = gtsam.BetweenFactorPose3(idx1, idx2, relative_pose, noise_model)
            graph.push_back(factor)
            
        # Fix reference pose
        fixed_idx = pose_ids.index(fixed_pose_id)
        fixed_pose = initial_estimate.atPose3(fixed_idx)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
        graph.push_back(gtsam.PriorFactorPose3(fixed_idx, fixed_pose, prior_noise))
        
        # Optimize
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iterations)
        params.setRelativeErrorTol(self.relative_error_tol)
        
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        
        # Extract optimized poses
        optimized_poses = {}
        for i, image_id in enumerate(pose_ids):
            optimized_pose = result.atPose3(i)
            R_opt = optimized_pose.rotation().matrix()
            t_opt = optimized_pose.translation()
            
            optimized_poses[image_id] = {
                "R": R_opt,
                "t": np.array([t_opt.x(), t_opt.y(), t_opt.z()]).reshape(3, 1)
            }
            
        logger.info(f"GTSAM optimization completed. Final error: {optimizer.error():.6f}")
        return optimized_poses
        
    def _scipy_optimize_poses(self, poses, pairwise_poses, fixed_pose_id):
        """SciPy-based fallback optimization"""
        
        pose_ids = list(poses.keys())
        if fixed_pose_id is None:
            fixed_pose_id = pose_ids[0]
            
        # Convert poses to parameter vector (axis-angle + translation)
        def pose_to_params(R, t):
            r = Rotation.from_matrix(R)
            rotvec = r.as_rotvec()
            return np.concatenate([rotvec, t.flatten()])
            
        def params_to_pose(params):
            rotvec = params[:3]
            t = params[3:6].reshape(3, 1)
            R = Rotation.from_rotvec(rotvec).as_matrix()
            return R, t
            
        # Build parameter vector (excluding fixed pose)
        param_poses = {id: poses[id] for id in pose_ids if id != fixed_pose_id}
        x0 = []
        param_ids = []
        
        for image_id, pose_data in param_poses.items():
            x0.extend(pose_to_params(pose_data["R"], pose_data["t"]))
            param_ids.append(image_id)
            
        x0 = np.array(x0)
        
        def residual_function(x):
            # Reconstruct poses from parameters
            current_poses = {fixed_pose_id: poses[fixed_pose_id]}
            
            for i, image_id in enumerate(param_ids):
                start_idx = i * 6
                params = x[start_idx:start_idx + 6]
                R, t = params_to_pose(params)
                current_poses[image_id] = {"R": R, "t": t}
                
            # Compute residuals for pairwise constraints
            residuals = []
            
            for constraint in pairwise_poses:
                id1, id2 = constraint["image_ids"]
                if id1 not in current_poses or id2 not in current_poses:
                    continue
                    
                # Get poses
                pose1 = current_poses[id1]
                pose2 = current_poses[id2]
                
                # Compute relative pose
                R_rel_pred = pose2["R"] @ pose1["R"].T
                t_rel_pred = pose2["t"] - R_rel_pred @ pose1["t"]
                
                # Expected relative pose
                R_rel_exp = constraint["R"]
                t_rel_exp = constraint["t"]
                
                # Rotation residual (axis-angle)
                R_error = R_rel_exp @ R_rel_pred.T
                rot_residual = Rotation.from_matrix(R_error).as_rotvec()
                
                # Translation residual
                trans_residual = (t_rel_pred - t_rel_exp).flatten()
                
                # Apply robust weighting (Huber)
                combined_residual = np.concatenate([rot_residual, trans_residual])
                norm = np.linalg.norm(combined_residual)
                
                if norm <= self.huber_threshold:
                    weight = 1.0
                else:
                    weight = self.huber_threshold / norm
                    
                residuals.extend(combined_residual * weight)
                
            return np.array(residuals)
            
        # Optimize
        result = least_squares(
            residual_function, 
            x0, 
            max_nfev=self.max_iterations * len(x0),
            ftol=self.relative_error_tol
        )
        
        # Extract optimized poses
        optimized_poses = {fixed_pose_id: poses[fixed_pose_id]}
        
        for i, image_id in enumerate(param_ids):
            start_idx = i * 6
            params = result.x[start_idx:start_idx + 6]
            R, t = params_to_pose(params)
            optimized_poses[image_id] = {"R": R, "t": t}
            
        logger.info(f"SciPy optimization completed. Final cost: {result.cost:.6f}")
        return optimized_poses
        
    def bundle_adjustment(self, 
                         poses: Dict[str, Dict],
                         points_3d: Dict[int, np.ndarray],
                         observations: List[Dict],
                         intrinsics: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Bundle adjustment optimization
        
        Args:
            poses: Camera poses
            points_3d: 3D points {point_id: xyz}
            observations: List of 2D-3D correspondences
            intrinsics: Camera intrinsic parameters
            
        Returns:
            Tuple of (optimized_poses, optimized_points_3d)
        """
        
        if GTSAM_AVAILABLE:
            return self._gtsam_bundle_adjustment(poses, points_3d, observations, intrinsics)
        else:
            return self._scipy_bundle_adjustment(poses, points_3d, observations, intrinsics)
            
    def _gtsam_bundle_adjustment(self, poses, points_3d, observations, intrinsics):
        """GTSAM-based bundle adjustment"""
        
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        # Add poses to initial estimate
        pose_ids = list(poses.keys())
        for i, (image_id, pose_data) in enumerate(poses.items()):
            R = pose_data["R"]
            t = pose_data["t"].flatten()
            
            rotation = gtsam.Rot3(R)
            translation = gtsam.Point3(t[0], t[1], t[2])
            pose = gtsam.Pose3(rotation, translation)
            
            initial_estimate.insert(gtsam.symbol('x', i), pose)
            
        # Add 3D points to initial estimate
        point_ids = list(points_3d.keys())
        for i, (point_id, xyz) in enumerate(points_3d.items()):
            point = gtsam.Point3(xyz[0], xyz[1], xyz[2])
            initial_estimate.insert(gtsam.symbol('l', i), point)
            
        # Add projection factors
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        
        for obs in observations:
            image_id = obs["image_id"]
            point_id = obs["point_id"]
            uv = obs["uv"]  # 2D observation
            
            pose_idx = pose_ids.index(image_id)
            point_idx = point_ids.index(point_id)
            
            # Camera calibration
            K = intrinsics[image_id]
            cal = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0, K[0, 2], K[1, 2])
            
            # Add projection factor
            factor = gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(uv[0], uv[1]),
                measurement_noise,
                gtsam.symbol('x', pose_idx),
                gtsam.symbol('l', point_idx),
                cal
            )
            graph.push_back(factor)
            
        # Add prior on first pose
        if pose_ids:
            first_pose = initial_estimate.atPose3(gtsam.symbol('x', 0))
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-3)
            graph.push_back(gtsam.PriorFactorPose3(
                gtsam.symbol('x', 0), first_pose, prior_noise))
                
        # Optimize
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iterations)
        
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        
        # Extract results
        optimized_poses = {}
        for i, image_id in enumerate(pose_ids):
            opt_pose = result.atPose3(gtsam.symbol('x', i))
            R_opt = opt_pose.rotation().matrix()
            t_opt = opt_pose.translation()
            
            optimized_poses[image_id] = {
                "R": R_opt,
                "t": np.array([t_opt.x(), t_opt.y(), t_opt.z()]).reshape(3, 1)
            }
            
        optimized_points_3d = {}
        for i, point_id in enumerate(point_ids):
            opt_point = result.atPoint3(gtsam.symbol('l', i))
            optimized_points_3d[point_id] = np.array([
                opt_point.x(), opt_point.y(), opt_point.z()
            ])
            
        logger.info(f"Bundle adjustment completed. Final error: {optimizer.error():.6f}")
        return optimized_poses, optimized_points_3d
        
    def _scipy_bundle_adjustment(self, poses, points_3d, observations, intrinsics):
        """SciPy-based bundle adjustment fallback"""
        
        logger.warning("Using simplified SciPy bundle adjustment")
        
        # Convert poses to parameter vector
        pose_ids = list(poses.keys())
        point_ids = list(points_3d.keys())
        
        def pose_to_params(R, t):
            r = Rotation.from_matrix(R)
            rotvec = r.as_rotvec()
            return np.concatenate([rotvec, t.flatten()])
            
        def params_to_pose(params):
            rotvec = params[:3]
            t = params[3:6].reshape(3, 1)
            R = Rotation.from_rotvec(rotvec).as_matrix()
            return R, t
        
        # Build parameter vector
        x0 = []
        param_poses = []
        param_points = []
        
        # Pose parameters
        for image_id in pose_ids:
            pose_data = poses[image_id]
            x0.extend(pose_to_params(pose_data["R"], pose_data["t"]))
            param_poses.append(image_id)
            
        # Point parameters
        for point_id in point_ids:
            x0.extend(points_3d[point_id])
            param_points.append(point_id)
            
        x0 = np.array(x0)
        
        def residual_function(x):
            # Reconstruct poses and points from parameters
            current_poses = {}
            current_points = {}
            
            # Extract poses
            for i, image_id in enumerate(param_poses):
                start_idx = i * 6
                params = x[start_idx:start_idx + 6]
                R, t = params_to_pose(params)
                current_poses[image_id] = {"R": R, "t": t}
                
            # Extract points
            pose_params_size = len(param_poses) * 6
            for i, point_id in enumerate(param_points):
                start_idx = pose_params_size + i * 3
                current_points[point_id] = x[start_idx:start_idx + 3]
                
            # Compute residuals for observations
            residuals = []
            
            for obs in observations:
                image_id = obs["image_id"]
                point_id = obs["point_id"]
                uv_observed = obs["uv"]
                
                if image_id not in current_poses or point_id not in current_points:
                    continue
                    
                pose = current_poses[image_id]
                point_3d = current_points[point_id]
                
                # Project 3D point
                K = intrinsics[image_id]
                R = pose["R"]
                t = pose["t"]
                
                # Transform to camera coordinates
                point_cam = R @ point_3d.reshape(3, 1) + t
                
                if point_cam[2, 0] <= 0:  # Behind camera
                    residuals.extend([1000.0, 1000.0])  # Large penalty
                    continue
                    
                # Project to image
                point_img = K @ point_cam
                uv_projected = point_img[:2, 0] / point_img[2, 0]
                
                # Residual
                residual = uv_projected - uv_observed
                residuals.extend(residual)
                
            return np.array(residuals)
        
        # Optimize
        try:
            result = least_squares(
                residual_function,
                x0,
                max_nfev=self.max_iterations * len(x0),
                ftol=self.relative_error_tol,
                loss='huber',
                f_scale=self.huber_threshold
            )
            
            # Extract optimized poses and points
            optimized_poses = {}
            optimized_points_3d = {}
            
            # Extract poses
            for i, image_id in enumerate(param_poses):
                start_idx = i * 6
                params = result.x[start_idx:start_idx + 6]
                R, t = params_to_pose(params)
                optimized_poses[image_id] = {"R": R, "t": t}
                
            # Extract points
            pose_params_size = len(param_poses) * 6
            for i, point_id in enumerate(param_points):
                start_idx = pose_params_size + i * 3
                optimized_points_3d[point_id] = result.x[start_idx:start_idx + 3]
                
            logger.info(f"SciPy bundle adjustment completed. Final cost: {result.cost:.6f}")
            return optimized_poses, optimized_points_3d
            
        except Exception as e:
            logger.error(f"SciPy bundle adjustment failed: {e}")
            return poses, points_3d


def create_pose_graph_from_matches(matches_data: Dict, 
                                  pose_estimates: Dict,
                                  confidence_threshold: float = 0.8) -> List[Dict]:
    """
    Create pose graph constraints from feature matches
    
    Args:
        matches_data: Dictionary with pairwise matches
        pose_estimates: Initial pose estimates from DEGENSAC
        confidence_threshold: Minimum confidence for including constraint
        
    Returns:
        List of pairwise pose constraints
    """
    
    constraints = []
    
    for pair_name, pair_data in matches_data.items():
        # Parse image IDs from pair name
        parts = pair_name.split('_')
        if len(parts) < 2:
            continue
            
        id1, id2 = parts[0], '_'.join(parts[1:])
        
        # Check if we have pose estimate for this pair
        pose_key = f"{id1}_{id2}"
        if pose_key not in pose_estimates:
            continue
            
        pose_est = pose_estimates[pose_key]
        if not pose_est.get("success", False):
            continue
            
        # Check confidence (number of inliers)
        if "inliers" in pose_est:
            inlier_ratio = np.sum(pose_est["inliers"]) / len(pose_est["inliers"])
            if inlier_ratio < confidence_threshold:
                continue
                
        constraints.append({
            "image_ids": (id1, id2),
            "R": pose_est["R"],
            "t": pose_est["t"],
            "confidence": inlier_ratio if "inliers" in pose_est else 1.0
        })
        
    logger.info(f"Created pose graph with {len(constraints)} constraints")
    return constraints