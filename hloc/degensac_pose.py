import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List

# Try to import pydengensac, fallback to pydegensac
try:
    import pydengensac
    DENGENSAC_AVAILABLE = True
except ImportError:
    try:
        import pydegensac
        DENGENSAC_AVAILABLE = True
        # Create alias for compatibility
        pydengensac = pydegensac
    except ImportError:
        DENGENSAC_AVAILABLE = False
        print("Warning: Neither pydengensac nor pydegensac found. Using fallback implementation.")

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
        Estimate relative pose using pydengensac/pydegensac
        
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
            
        # Use pydengensac/pydegensac for fundamental matrix estimation
        if not DENGENSAC_AVAILABLE:
            logger.warning("DEGENSAC not available, using fallback implementation")
            return self._fallback_estimate_pose(pts1, pts2, K1, K2)
            
        try:
            F, inliers = pydengensac.findFundamentalMatrix(
                pts1, pts2,
                threshold=self.threshold,
                confidence=self.confidence,
                max_iterations=self.max_iterations
            )
            
            if F is None:
                logger.warning("pydengensac/pydegensac failed to find valid fundamental matrix")
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
                
            logger.info(f"pydengensac/pydegensac: {len(inliers)}/{len(pts1)} inliers "
                       f"({100*len(inliers)/len(pts1):.1f}%)")
            
            return pose_result
            
        except Exception as e:
            logger.error(f"pydengensac/pydegensac error: {e}")
            return self._fallback_estimate_pose(pts1, pts2, K1, K2)
            
    def _fallback_estimate_pose(self, 
                               pts1: np.ndarray, 
                               pts2: np.ndarray,
                               K1: Optional[np.ndarray] = None,
                               K2: Optional[np.ndarray] = None) -> Dict:
        """
        Fallback pose estimation using OpenCV's RANSAC
        
        Args:
            pts1, pts2: [N, 2] point correspondences
            K1, K2: [3, 3] camera intrinsic matrices
            
        Returns:
            Dictionary with pose estimation results
        """
        
        if len(pts1) < 8:
            return {"success": False}
            
        try:
            if K1 is not None and K2 is not None:
                # Use essential matrix method
                E, inliers = cv2.findEssentialMat(
                    pts1, pts2, K1, 
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=self.threshold
                )
                
                if E is None:
                    return {"success": False}
                    
                # Decompose essential matrix
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K1)
                
                # Convert essential matrix to fundamental matrix
                F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
                
            else:
                # Use fundamental matrix method
                F, inliers = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.threshold,
                    confidence=0.99,
                    maxIters=self.max_iterations
                )
                
                if F is None:
                    return {"success": False}
                    
                R = t = mask = None
                
            # Convert inliers to boolean mask
            inlier_mask = np.zeros(len(pts1), dtype=bool)
            inlier_mask[inliers.ravel()] = True
            
            pose_result = {"success": True, "F": F, "inliers": inlier_mask}
            
            if R is not None and t is not None:
                pose_result.update({
                    "R": R,
                    "t": t,
                    "pose_inliers": mask.ravel().astype(bool)
                })
                
            logger.info(f"OpenCV fallback: {inlier_mask.sum()}/{len(pts1)} inliers "
                       f"({100*inlier_mask.sum()/len(pts1):.1f}%)")
            
            return pose_result
            
        except Exception as e:
            logger.error(f"Fallback pose estimation failed: {e}")
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
        Estimate relative pose using pydengensac/pydegensac LoRANSAC
        
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
            
        # Use pydengensac/pydegensac for fundamental matrix estimation with LoRANSAC
        if not DENGENSAC_AVAILABLE:
            logger.warning("LoRANSAC not available, using fallback implementation")
            return self._fallback_estimate_pose(pts1, pts2, K1, K2)
            
        try:
            F, inliers = pydengensac.findFundamentalMatrix(
                pts1, pts2,
                threshold=self.threshold,
                confidence=self.confidence,
                max_iterations=self.max_iterations,
                method='loransac'  # Use LoRANSAC method
            )
            
            if F is None:
                logger.warning("pydengensac/pydegensac LoRANSAC failed to find valid fundamental matrix")
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
                
            logger.info(f"pydengensac/pydegensac LoRANSAC: {len(inliers)}/{len(pts1)} inliers "
                       f"({100*len(inliers)/len(pts1):.1f}%)")
            
            return pose_result
            
        except Exception as e:
            logger.error(f"pydengensac/pydegensac LoRANSAC error: {e}")
            return self._fallback_estimate_pose(pts1, pts2, K1, K2)
            
    def _fallback_estimate_pose(self, 
                               pts1: np.ndarray, 
                               pts2: np.ndarray,
                               K1: Optional[np.ndarray] = None,
                               K2: Optional[np.ndarray] = None) -> Dict:
        """
        Fallback pose estimation using OpenCV's RANSAC (for LoRANSAC)
        
        Args:
            pts1, pts2: [N, 2] point correspondences
            K1, K2: [3, 3] camera intrinsic matrices
            
        Returns:
            Dictionary with pose estimation results
        """
        
        if len(pts1) < 8:
            return {"success": False}
            
        try:
            if K1 is not None and K2 is not None:
                # Use essential matrix method
                E, inliers = cv2.findEssentialMat(
                    pts1, pts2, K1, 
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=self.threshold
                )
                
                if E is None:
                    return {"success": False}
                    
                # Decompose essential matrix
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K1)
                
                # Convert essential matrix to fundamental matrix
                F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
                
            else:
                # Use fundamental matrix method
                F, inliers = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.threshold,
                    confidence=0.99,
                    maxIters=self.max_iterations
                )
                
                if F is None:
                    return {"success": False}
                    
                R = t = mask = None
                
            # Convert inliers to boolean mask
            inlier_mask = np.zeros(len(pts1), dtype=bool)
            inlier_mask[inliers.ravel()] = True
            
            pose_result = {"success": True, "F": F, "inliers": inlier_mask}
            
            if R is not None and t is not None:
                pose_result.update({
                    "R": R,
                    "t": t,
                    "pose_inliers": mask.ravel().astype(bool)
                })
                
            logger.info(f"OpenCV LoRANSAC fallback: {inlier_mask.sum()}/{len(pts1)} inliers "
                       f"({100*inlier_mask.sum()/len(pts1):.1f}%)")
            
            return pose_result
            
        except Exception as e:
            logger.error(f"LoRANSAC fallback pose estimation failed: {e}")
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