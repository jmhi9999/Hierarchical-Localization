import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares

from . import logger


class DirectTriangulation:
    """Direct triangulation implementation without COLMAP dependency"""
    
    def __init__(self, 
                 reprojection_threshold: float = 4.0,
                 min_triangulation_angle: float = 2.0,
                 max_reprojection_error: float = 8.0):
        
        self.reprojection_threshold = reprojection_threshold
        self.min_triangulation_angle = np.radians(min_triangulation_angle)
        self.max_reprojection_error = max_reprojection_error
        
    def triangulate_point_dlt(self, 
                             P1: np.ndarray, 
                             P2: np.ndarray,
                             pt1: np.ndarray, 
                             pt2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D point using Direct Linear Transform (DLT)
        
        Args:
            P1, P2: [3, 4] projection matrices
            pt1, pt2: [2] corresponding 2D points
            
        Returns:
            [3] triangulated 3D point
        """
        
        # Build constraint matrix A
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1], 
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1]
        ])
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        # Convert from homogeneous to 3D coordinates
        if abs(X[3]) < 1e-10:
            return None
            
        return X[:3] / X[3]
        
    def triangulate_point_midpoint(self,
                                  P1: np.ndarray,
                                  P2: np.ndarray, 
                                  pt1: np.ndarray,
                                  pt2: np.ndarray) -> np.ndarray:
        """
        Triangulate using midpoint method (more robust for nearly parallel rays)
        
        Args:
            P1, P2: [3, 4] projection matrices
            pt1, pt2: [2] corresponding 2D points
            
        Returns:
            [3] triangulated 3D point
        """
        
        # Extract camera centers and directions
        C1 = -np.linalg.inv(P1[:, :3]) @ P1[:, 3]
        C2 = -np.linalg.inv(P2[:, :3]) @ P2[:, 3]
        
        # Normalize 2D points 
        pt1_norm = np.array([pt1[0], pt1[1], 1.0])
        pt2_norm = np.array([pt2[0], pt2[1], 1.0])
        
        # Ray directions
        d1 = np.linalg.inv(P1[:, :3]) @ pt1_norm
        d2 = np.linalg.inv(P2[:, :3]) @ pt2_norm
        
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)
        
        # Find closest points on rays
        # Solve: C1 + t1*d1 = C2 + t2*d2 (in least squares sense)
        A = np.column_stack([d1, -d2])
        b = C2 - C1
        
        try:
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            t1, t2 = params[0], params[1]
            
            # Compute 3D points on each ray
            X1 = C1 + t1 * d1
            X2 = C2 + t2 * d2
            
            # Return midpoint
            return (X1 + X2) / 2
            
        except np.linalg.LinAlgError:
            return None
            
    def compute_triangulation_angle(self,
                                   P1: np.ndarray,
                                   P2: np.ndarray,
                                   X: np.ndarray) -> float:
        """
        Compute triangulation angle between two cameras for a 3D point
        
        Args:
            P1, P2: [3, 4] projection matrices
            X: [3] 3D point
            
        Returns:
            Triangulation angle in radians
        """
        
        # Extract camera centers
        C1 = -np.linalg.inv(P1[:, :3]) @ P1[:, 3]
        C2 = -np.linalg.inv(P2[:, :3]) @ P2[:, 3]
        
        # Vectors from cameras to 3D point
        v1 = X - C1
        v2 = X - C2
        
        # Normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)
        
        # Compute angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(abs(cos_angle))
        
        return min(angle, np.pi - angle)
        
    def compute_reprojection_error(self,
                                  P: np.ndarray,
                                  X: np.ndarray, 
                                  pt_observed: np.ndarray) -> float:
        """
        Compute reprojection error for a 3D point
        
        Args:
            P: [3, 4] projection matrix
            X: [3] 3D point
            pt_observed: [2] observed 2D point
            
        Returns:
            Reprojection error in pixels
        """
        
        # Project 3D point
        X_hom = np.append(X, 1.0)
        pt_proj_hom = P @ X_hom
        
        if abs(pt_proj_hom[2]) < 1e-10:
            return float('inf')
            
        pt_proj = pt_proj_hom[:2] / pt_proj_hom[2]
        
        # Compute error
        error = np.linalg.norm(pt_proj - pt_observed)
        return error
        
    def triangulate_points(self,
                          poses: Dict[str, Dict],
                          intrinsics: Dict[str, np.ndarray],
                          matches_data: Dict,
                          track_data: Dict) -> Dict[int, Dict]:
        """
        Triangulate 3D points from multiple view correspondences
        
        Args:
            poses: Camera poses {image_id: {"R": R, "t": t}}
            intrinsics: Camera intrinsics {image_id: K}
            matches_data: Pairwise matches
            track_data: Point tracks across images {track_id: [(image_id, kpt_idx), ...]}
            
        Returns:
            Dictionary of 3D points {point_id: {"xyz": xyz, "observations": obs_data}}
        """
        
        points_3d = {}
        
        # Build projection matrices
        projection_matrices = {}
        for image_id, pose in poses.items():
            if image_id not in intrinsics:
                continue
                
            K = intrinsics[image_id]
            R = pose["R"]
            t = pose["t"].reshape(3, 1)
            
            # P = K [R | t]
            Rt = np.hstack([R, t])
            P = K @ Rt
            projection_matrices[image_id] = P
            
        logger.info(f"Built {len(projection_matrices)} projection matrices")
        
        # Triangulate each track
        successful_triangulations = 0
        
        for track_id, observations in track_data.items():
            # Need at least 2 observations
            if len(observations) < 2:
                continue
                
            # Filter observations with valid projection matrices
            valid_obs = [(img_id, kpt_idx) for img_id, kpt_idx in observations 
                        if img_id in projection_matrices]
            
            if len(valid_obs) < 2:
                continue
                
            # Try triangulation with best pair (most separated cameras)
            best_angle = 0
            best_point = None
            best_obs_pair = None
            
            for i in range(len(valid_obs)):
                for j in range(i + 1, len(valid_obs)):
                    img_id1, kpt_idx1 = valid_obs[i]
                    img_id2, kpt_idx2 = valid_obs[j]
                    
                    P1 = projection_matrices[img_id1]
                    P2 = projection_matrices[img_id2]
                    
                    # Get 2D points (this would need to be extracted from features)
                    # For now, using placeholder - in real implementation, 
                    # extract from keypoints using kpt_idx1, kpt_idx2
                    pt1 = np.array([100.0, 100.0])  # Placeholder
                    pt2 = np.array([110.0, 105.0])  # Placeholder
                    
                    # Triangulate
                    X_dlt = self.triangulate_point_dlt(P1, P2, pt1, pt2)
                    if X_dlt is None:
                        continue
                        
                    # Check triangulation angle
                    angle = self.compute_triangulation_angle(P1, P2, X_dlt)
                    if angle < self.min_triangulation_angle:
                        continue
                        
                    # Check reprojection errors
                    error1 = self.compute_reprojection_error(P1, X_dlt, pt1)
                    error2 = self.compute_reprojection_error(P2, X_dlt, pt2)
                    
                    if (error1 < self.reprojection_threshold and 
                        error2 < self.reprojection_threshold and
                        angle > best_angle):
                        
                        best_angle = angle
                        best_point = X_dlt
                        best_obs_pair = (i, j)
                        
            if best_point is not None:
                # Refine with all observations using bundle adjustment
                refined_point = self._refine_point_ba(
                    best_point, valid_obs, projection_matrices)
                
                if refined_point is not None:
                    points_3d[track_id] = {
                        "xyz": refined_point,
                        "observations": valid_obs,
                        "triangulation_angle": best_angle,
                        "num_observations": len(valid_obs)
                    }
                    successful_triangulations += 1
                    
        logger.info(f"Successfully triangulated {successful_triangulations} points")
        return points_3d
        
    def _refine_point_ba(self, 
                        initial_point: np.ndarray,
                        observations: List[Tuple[str, int]], 
                        projection_matrices: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Refine 3D point using bundle adjustment (point-only)
        
        Args:
            initial_point: [3] initial 3D point estimate
            observations: List of (image_id, kpt_idx) tuples
            projection_matrices: Dictionary of projection matrices
            
        Returns:
            Refined 3D point or None if optimization failed
        """
        
        def residual_function(X):
            residuals = []
            
            for img_id, kpt_idx in observations:
                if img_id not in projection_matrices:
                    continue
                    
                P = projection_matrices[img_id]
                
                # Get observed 2D point (placeholder - should extract from actual keypoints)
                pt_observed = np.array([100.0, 100.0])  # Placeholder
                
                # Compute reprojection error
                error = self.compute_reprojection_error(P, X, pt_observed)
                residuals.append(error)
                
            return np.array(residuals)
            
        try:
            result = least_squares(
                residual_function,
                initial_point,
                loss='huber',
                f_scale=self.reprojection_threshold
            )
            
            if result.success:
                return result.x
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Point refinement failed: {e}")
            return None
            
    def create_point_tracks(self, matches_data: Dict) -> Dict[int, List[Tuple[str, int]]]:
        """
        Create point tracks from pairwise matches
        
        Args:
            matches_data: Dictionary with pairwise match data
            
        Returns:
            Dictionary of tracks {track_id: [(image_id, keypoint_idx), ...]}
        """
        
        # This is a simplified track building - 
        # Full implementation would use Union-Find or graph-based methods
        
        tracks = {}
        track_id = 0
        used_points = set()
        
        for pair_name, pair_data in matches_data.items():
            # Parse image IDs
            parts = pair_name.split('_')
            if len(parts) < 2:
                continue
                
            img_id1, img_id2 = parts[0], '_'.join(parts[1:])
            
            if "matches0" not in pair_data:
                continue
                
            matches = pair_data["matches0"]
            
            for match in matches:
                if match[1] == -1:  # No match
                    continue
                    
                kpt_idx1, kpt_idx2 = int(match[0]), int(match[1])
                
                point1 = (img_id1, kpt_idx1)
                point2 = (img_id2, kpt_idx2)
                
                if point1 not in used_points and point2 not in used_points:
                    # Create new track
                    tracks[track_id] = [point1, point2]
                    used_points.add(point1)
                    used_points.add(point2)
                    track_id += 1
                    
        logger.info(f"Created {len(tracks)} point tracks")
        return tracks


def triangulate_reconstruction(poses: Dict[str, Dict],
                             intrinsics: Dict[str, np.ndarray], 
                             matches_data: Dict,
                             **kwargs) -> Dict[int, Dict]:
    """
    Convenience function for triangulating 3D reconstruction
    
    Args:
        poses: Camera poses
        intrinsics: Camera intrinsics
        matches_data: Pairwise matches
        **kwargs: Additional parameters for triangulation
        
    Returns:
        Dictionary of triangulated 3D points
    """
    
    triangulator = DirectTriangulation(**kwargs)
    
    # Create point tracks from matches
    tracks = triangulator.create_point_tracks(matches_data)
    
    # Triangulate 3D points
    points_3d = triangulator.triangulate_points(
        poses, intrinsics, matches_data, tracks)
        
    return points_3d