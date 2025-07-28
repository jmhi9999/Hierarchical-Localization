import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import pydengensac

from . import logger


class DEGENSAC:
    """DEGENSAC: DEep GENeralized RANSAC for robust fundamental matrix estimation using pydengensac"""
    
    def __init__(self, 
                 max_iterations: int = 10000,
                 confidence: float = 0.99,
                 threshold: float = 1.0,
                 min_sample_size: int = 8):
        
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.threshold = threshold
        self.min_sample_size = min_sample_size
        
    def estimate_pose(self, 
                     keypoints1: np.ndarray, 
                     keypoints2: np.ndarray,
                     matches: np.ndarray,
                     K1: Optional[np.ndarray] = None,
                     K2: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate relative pose using pydengensac
        
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
            
        # Use pydengensac for fundamental matrix estimation
        try:
            F, inliers = pydengensac.findFundamentalMatrix(
                pts1, pts2,
                threshold=self.threshold,
                confidence=self.confidence,
                max_iterations=self.max_iterations
            )
            
            if F is None:
                logger.warning("pydengensac failed to find valid fundamental matrix")
                return {"success": False}
                
            # Convert inliers to boolean mask
            inlier_mask = np.zeros(len(pts1), dtype=bool)
            inlier_mask[inliers] = True
            
            # Extract relative pose from fundamental matrix
            pose_result = {"success": True, "F": F, "inliers": inlier_mask}
            
            if K1 is not None and K2 is not None:
                # Compute essential matrix
                E = K2.T @ F @ K1
                
                # Decompose essential matrix to R, t
                inlier_pts1 = pts1[inlier_mask]
                inlier_pts2 = pts2[inlier_mask]
                
                _, R, t, mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K1)
                
                pose_result.update({
                    "E": E,
                    "R": R,
                    "t": t,
                    "pose_inliers": mask.ravel().astype(bool)
                })
                
            logger.info(f"pydengensac: {len(inliers)}/{len(pts1)} inliers "
                       f"({100*len(inliers)/len(pts1):.1f}%)")
            
            return pose_result
            
        except Exception as e:
            logger.error(f"pydengensac error: {e}")
            return {"success": False}


class LoRANSAC:
    """LoRANSAC: Locally Optimized RANSAC for robust fundamental matrix estimation using pydengensac"""
    
    def __init__(self,
                 max_iterations: int = 10000,
                 confidence: float = 0.99,
                 threshold: float = 1.0,
                 min_sample_size: int = 8):
        
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.threshold = threshold
        self.min_sample_size = min_sample_size
        
    def estimate_pose(self,
                     keypoints1: np.ndarray,
                     keypoints2: np.ndarray,
                     matches: np.ndarray,
                     K1: Optional[np.ndarray] = None,
                     K2: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate relative pose using pydengensac LoRANSAC
        
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
            
        # Use pydengensac for fundamental matrix estimation with LoRANSAC
        try:
            F, inliers = pydengensac.findFundamentalMatrix(
                pts1, pts2,
                threshold=self.threshold,
                confidence=self.confidence,
                max_iterations=self.max_iterations,
                method='loransac'  # Use LoRANSAC method
            )
            
            if F is None:
                logger.warning("pydengensac LoRANSAC failed to find valid fundamental matrix")
                return {"success": False}
                
            # Convert inliers to boolean mask
            inlier_mask = np.zeros(len(pts1), dtype=bool)
            inlier_mask[inliers] = True
            
            # Extract relative pose from fundamental matrix
            pose_result = {"success": True, "F": F, "inliers": inlier_mask}
            
            if K1 is not None and K2 is not None:
                # Compute essential matrix
                E = K2.T @ F @ K1
                
                # Decompose essential matrix to R, t
                inlier_pts1 = pts1[inlier_mask]
                inlier_pts2 = pts2[inlier_mask]
                
                _, R, t, mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K1)
                
                pose_result.update({
                    "E": E,
                    "R": R,
                    "t": t,
                    "pose_inliers": mask.ravel().astype(bool)
                })
                
            logger.info(f"pydengensac LoRANSAC: {len(inliers)}/{len(pts1)} inliers "
                       f"({100*len(inliers)/len(pts1):.1f}%)")
            
            return pose_result
            
        except Exception as e:
            logger.error(f"pydengensac LoRANSAC error: {e}")
            return {"success": False}


def estimate_relative_pose(keypoints1: np.ndarray,
                          keypoints2: np.ndarray, 
                          matches: np.ndarray,
                          camera_params: Optional[Dict] = None,
                          method: str = "degensac",
                          **kwargs) -> Dict:
    """
    Convenience function for relative pose estimation using pydengensac
    
    Args:
        keypoints1, keypoints2: Keypoint arrays
        matches: Match indices
        camera_params: Optional camera parameters
        method: Pose estimation method ("degensac" or "loransac")
        **kwargs: Additional parameters for the chosen method
        
    Returns:
        Pose estimation results
    """
    
    # Extract camera intrinsics
    K1 = K2 = None
    if camera_params is not None:
        K1 = camera_params.get("K1")
        K2 = camera_params.get("K2")
    
    # Choose estimator based on method
    if method.lower() == "loransac":
        estimator = LoRANSAC(**kwargs)
        logger.info("Using pydengensac LoRANSAC for pose estimation")
    elif method.lower() == "degensac":
        estimator = DEGENSAC(**kwargs)
        logger.info("Using pydengensac DEGENSAC for pose estimation")
    else:
        raise ValueError(f"Unknown pose estimation method: {method}. "
                        "Choose 'degensac' or 'loransac'")
        
    return estimator.estimate_pose(keypoints1, keypoints2, matches, K1, K2)