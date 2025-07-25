import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from scipy.spatial.distance import cdist

from . import logger


class DeepFundamentalModel(nn.Module):
    """Deep learning model for fundamental matrix estimation"""
    
    def __init__(self, input_dim: int = 9):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9),  # 3x3 fundamental matrix
        )
        
    def forward(self, correspondences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            correspondences: [N, 9] normalized correspondences [x1, y1, 1, x2, y2, 1, x1*x2, x1*y2, y1*x2]
        Returns:
            fundamental_matrix: [3, 3] fundamental matrix
        """
        f_vec = self.network(correspondences.mean(dim=0))
        F = f_vec.view(3, 3)
        
        # Enforce rank-2 constraint via SVD
        U, S, V = torch.svd(F)
        S[-1] = 0  # Set smallest singular value to 0
        F_rank2 = U @ torch.diag(S) @ V.t()
        
        return F_rank2


class DEGENSAC:
    """DEGENSAC: DEep GENeralized RANSAC for robust fundamental matrix estimation"""
    
    def __init__(self, 
                 max_iterations: int = 10000,
                 confidence: float = 0.99,
                 threshold: float = 1.0,
                 min_sample_size: int = 8):
        
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.threshold = threshold
        self.min_sample_size = min_sample_size
        
        # Load pre-trained deep F model (simplified - in practice load actual weights)
        self.deep_f_model = DeepFundamentalModel()
        self.deep_f_model.eval()
        
    def normalize_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize points for numerical stability"""
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        std = np.where(std == 0, 1, std)
        
        T = np.array([
            [1/std[0], 0, -mean[0]/std[0]],
            [0, 1/std[1], -mean[1]/std[1]],
            [0, 0, 1]
        ])
        
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        points_norm = (T @ points_hom.T).T
        
        return points_norm[:, :2], T
        
    def compute_fundamental_8point(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """Classical 8-point algorithm for fundamental matrix"""
        n = pts1.shape[0]
        
        # Build constraint matrix
        A = np.zeros((n, 9))
        for i in range(n):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]
            A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
            
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)
        
        # Enforce rank-2 constraint
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ Vt
        
        return F
        
    def compute_deep_fundamental(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """Use deep learning model for fundamental matrix estimation"""
        n = pts1.shape[0]
        
        # Prepare normalized correspondences
        correspondences = []
        for i in range(n):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]
            corr = [x1, y1, 1, x2, y2, 1, x1*x2, x1*y2, y1*x2]
            correspondences.append(corr)
            
        correspondences = torch.tensor(correspondences, dtype=torch.float32)
        
        with torch.no_grad():
            F = self.deep_f_model(correspondences)
            
        return F.cpu().numpy()
        
    def compute_sampson_distance(self, F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """Compute Sampson distance for fundamental matrix"""
        pts1_hom = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
        
        # Compute epipolar lines
        lines1 = (F @ pts2_hom.T).T  # lines in image 1
        lines2 = (F.T @ pts1_hom.T).T  # lines in image 2
        
        # Sampson distance
        numerator = np.sum(pts1_hom * lines1, axis=1) ** 2
        denominator = (lines1[:, 0] ** 2 + lines1[:, 1] ** 2 + 
                      lines2[:, 0] ** 2 + lines2[:, 1] ** 2)
        
        distances = numerator / (denominator + 1e-8)
        return np.sqrt(distances)
        
    def estimate_pose(self, 
                     keypoints1: np.ndarray, 
                     keypoints2: np.ndarray,
                     matches: np.ndarray,
                     K1: Optional[np.ndarray] = None,
                     K2: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate relative pose using DEGENSAC
        
        Args:
            keypoints1, keypoints2: [N, 2] keypoint coordinates
            matches: [M, 2] match indices
            K1, K2: [3, 3] camera intrinsic matrices
            
        Returns:
            Dictionary with pose estimation results
        """
        
        # Extract matched points
        pts1 = keypoints1[matches[:, 0]]
        pts2 = keypoints2[matches[:, 1]]
        
        if len(pts1) < self.min_sample_size:
            logger.warning(f"Insufficient matches: {len(pts1)} < {self.min_sample_size}")
            return {"success": False}
            
        # Normalize points
        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)
        
        best_F = None
        best_inliers = None
        max_inliers = 0
        
        # RANSAC loop
        for iteration in range(self.max_iterations):
            # Sample minimal set
            sample_idx = np.random.choice(len(pts1_norm), self.min_sample_size, replace=False)
            sample_pts1 = pts1_norm[sample_idx]
            sample_pts2 = pts2_norm[sample_idx]
            
            try:
                # Use deep model for fundamental matrix estimation
                if iteration % 2 == 0:  # Alternate between classical and deep
                    F_norm = self.compute_fundamental_8point(sample_pts1, sample_pts2)
                else:
                    F_norm = self.compute_deep_fundamental(sample_pts1, sample_pts2)
                    
                # Denormalize fundamental matrix
                F = T2.T @ F_norm @ T1
                    
                # Compute inliers
                distances = self.compute_sampson_distance(F, pts1, pts2)
                inliers = distances < self.threshold
                num_inliers = np.sum(inliers)
                
                # Update best model
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_F = F.copy()
                    best_inliers = inliers.copy()
                    
                # Early termination check
                inlier_ratio = num_inliers / len(pts1)
                if inlier_ratio > 0.8:  # High confidence threshold
                    break
                    
            except np.linalg.LinAlgError:
                continue
                
        if best_F is None:
            logger.warning("DEGENSAC failed to find valid fundamental matrix")
            return {"success": False}
            
        # Refine with all inliers using deep model
        if np.sum(best_inliers) >= self.min_sample_size:
            inlier_pts1 = pts1_norm[best_inliers]
            inlier_pts2 = pts2_norm[best_inliers]
            
            refined_F_norm = self.compute_deep_fundamental(inlier_pts1, inlier_pts2)
            refined_F = T2.T @ refined_F_norm @ T1
            best_F = refined_F
            
        # Extract relative pose from fundamental matrix
        pose_result = {"success": True, "F": best_F, "inliers": best_inliers}
        
        if K1 is not None and K2 is not None:
            # Compute essential matrix
            E = K2.T @ best_F @ K1
            
            # Decompose essential matrix to R, t
            inlier_pts1 = pts1[best_inliers]
            inlier_pts2 = pts2[best_inliers]
            
            _, R, t, mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K1)
            
            pose_result.update({
                "E": E,
                "R": R,
                "t": t,
                "pose_inliers": mask.ravel().astype(bool)
            })
            
        logger.info(f"DEGENSAC: {max_inliers}/{len(pts1)} inliers "
                   f"({100*max_inliers/len(pts1):.1f}%)")
        
        return pose_result


def estimate_relative_pose(keypoints1: np.ndarray,
                          keypoints2: np.ndarray, 
                          matches: np.ndarray,
                          camera_params: Optional[Dict] = None,
                          **kwargs) -> Dict:
    """
    Convenience function for relative pose estimation
    
    Args:
        keypoints1, keypoints2: Keypoint arrays
        matches: Match indices
        camera_params: Optional camera parameters
        **kwargs: Additional parameters for DEGENSAC
        
    Returns:
        Pose estimation results
    """
    
    degensac = DEGENSAC(**kwargs)
    
    K1 = K2 = None
    if camera_params is not None:
        K1 = camera_params.get("K1")
        K2 = camera_params.get("K2")
        
    return degensac.estimate_pose(keypoints1, keypoints2, matches, K1, K2)