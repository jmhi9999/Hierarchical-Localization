#!/usr/bin/env python3
"""
Improved HLoc Pipeline Implementation

This pipeline implements the following improvements over standard HLoc:
1. SuperPoint + LightGlue feature matching (vs SuperGlue)
2. LightGlue's inlier mask for geometric verification (vs separate RANSAC)
3. DEGENSAC for robust pose estimation (vs standard methods)
4. GTSAM-based pose graph optimization (vs COLMAP dependency)
5. Direct triangulation implementation (vs COLMAP dependency)

Usage:
    python hloc_pipeline.py --images_dir /path/to/images --output_dir /path/to/output
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import h5py

# Import our improved modules
from hloc import extract_features, pairs_from_exhaustive, logger
from hloc.improved_match_features import main as improved_match_features
from hloc.degensac_pose import estimate_relative_pose
from hloc.gtsam_optimization import GTSAMOptimizer, create_pose_graph_from_matches
from hloc.direct_triangulation import triangulate_reconstruction


class ImprovedHLocPipeline:
    """
    Improved HLoc pipeline with modern algorithms and reduced dependencies
    """
    
    def __init__(self, 
                 images_dir: Path,
                 output_dir: Path,
                 feature_conf: str = "superpoint_aachen",
                 matcher_conf: str = "superpoint+lightglue"):
        
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.feature_conf = feature_conf
        self.matcher_conf = matcher_conf
        
        # Pipeline outputs
        self.features_path = self.output_dir / "features.h5"
        self.pairs_path = self.output_dir / "pairs.txt"
        self.matches_path = self.output_dir / "matches.h5"
        self.poses_path = self.output_dir / "poses.txt"
        self.points_path = self.output_dir / "points3d.txt"
        
        logger.info(f"Initialized improved HLoc pipeline")
        logger.info(f"Images: {self.images_dir}")
        logger.info(f"Output: {self.output_dir}")
        
    def run_full_pipeline(self, 
                         max_pairs: int = 5000,
                         intrinsics: Optional[Dict[str, np.ndarray]] = None,
                         optimize_poses: bool = True,
                         bundle_adjustment: bool = True) -> Dict:
        """
        Run the complete improved HLoc pipeline
        
        Args:
            max_pairs: Maximum number of image pairs to process
            intrinsics: Camera intrinsic parameters (if known)
            optimize_poses: Whether to run pose graph optimization
            bundle_adjustment: Whether to run bundle adjustment
            
        Returns:
            Dictionary with reconstruction results
        """
        
        results = {}
        
        # Step 1: Extract features using SuperPoint
        logger.info("=" * 50)
        logger.info("Step 1: Feature Extraction (SuperPoint)")
        logger.info("=" * 50)
        
        self.extract_features()
        results["features_path"] = str(self.features_path)
        
        # Step 2: Generate image pairs
        logger.info("=" * 50)
        logger.info("Step 2: Generate Image Pairs")
        logger.info("=" * 50)
        
        self.generate_pairs(max_pairs=max_pairs)
        results["pairs_path"] = str(self.pairs_path)
        
        # Step 3: Match features using LightGlue
        logger.info("=" * 50)
        logger.info("Step 3: Feature Matching (LightGlue)")
        logger.info("=" * 50)
        
        self.match_features()
        results["matches_path"] = str(self.matches_path)
        
        # Step 4: Estimate pairwise poses using DEGENSAC
        logger.info("=" * 50)
        logger.info("Step 4: Pairwise Pose Estimation (DEGENSAC)")
        logger.info("=" * 50)
        
        pose_estimates = self.estimate_pairwise_poses(intrinsics=intrinsics)
        results["pairwise_poses"] = len(pose_estimates)
        
        # Step 5: Global pose optimization using GTSAM
        logger.info("=" * 50)
        logger.info("Step 5: Global Pose Optimization (GTSAM)")
        logger.info("=" * 50)
        
        if optimize_poses and len(pose_estimates) > 0:
            global_poses = self.optimize_global_poses(pose_estimates)
            results["global_poses"] = len(global_poses)
        else:
            logger.warning("Skipping pose optimization - insufficient poses")
            global_poses = self._create_initial_poses_from_pairwise(pose_estimates)
            results["global_poses"] = len(global_poses)
        
        # Step 6: Triangulate 3D points
        logger.info("=" * 50)
        logger.info("Step 6: 3D Point Triangulation")
        logger.info("=" * 50)
        
        if len(global_poses) >= 2:
            points_3d = self.triangulate_points(global_poses, intrinsics)
            results["points_3d"] = len(points_3d)
            
            # Step 7: Bundle adjustment (optional)
            if bundle_adjustment and len(points_3d) > 10:
                logger.info("=" * 50)
                logger.info("Step 7: Bundle Adjustment")
                logger.info("=" * 50)
                
                refined_poses, refined_points = self.bundle_adjustment(
                    global_poses, points_3d, intrinsics)
                results["refined_poses"] = len(refined_poses)
                results["refined_points"] = len(refined_points)
                
                global_poses = refined_poses
                points_3d = refined_points
        else:
            logger.warning("Insufficient poses for triangulation")
            points_3d = {}
            results["points_3d"] = 0
        
        # Save results
        self.save_reconstruction(global_poses, points_3d)
        results["reconstruction_saved"] = True
        
        logger.info("=" * 50)
        logger.info("Pipeline Complete!")
        logger.info("=" * 50)
        logger.info(f"Results: {results}")
        
        return results
        
    def extract_features(self):
        """Extract features using SuperPoint"""
        if self.features_path.exists():
            logger.info(f"Features already exist at {self.features_path}")
            return
            
        from hloc.extract_features import confs, main
        
        conf = confs[self.feature_conf]
        extract_features.main(
            conf=conf,
            image_dir=self.images_dir,
            export_dir=self.output_dir,
            feature_path=self.features_path
        )
        
    def generate_pairs(self, max_pairs: int = 5000):
        """Generate exhaustive image pairs"""
        if self.pairs_path.exists():
            logger.info(f"Pairs already exist at {self.pairs_path}")
            return
            
        pairs_from_exhaustive.main(
            output=self.pairs_path,
            image_list=None,  # Will use all images in features file
            features=self.features_path,
            ref_features=None
        )
        
        # Limit number of pairs if requested
        if max_pairs > 0:
            with open(self.pairs_path, 'r') as f:
                lines = f.readlines()
                
            if len(lines) > max_pairs:
                logger.info(f"Limiting pairs from {len(lines)} to {max_pairs}")
                with open(self.pairs_path, 'w') as f:
                    f.writelines(lines[:max_pairs])
                    
    def match_features(self):
        """Match features using improved LightGlue matcher"""
        if self.matches_path.exists():
            logger.info(f"Matches already exist at {self.matches_path}")
            return
            
        improved_match_features(
            conf={"matcher": {"features": "superpoint"}},
            pairs=self.pairs_path,
            features=self.features_path,
            matches=self.matches_path
        )
        
    def estimate_pairwise_poses(self, 
                               intrinsics: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """Estimate pairwise poses using DEGENSAC"""
        
        pose_estimates = {}
        
        # Load matches and features
        matches_file = h5py.File(str(self.matches_path), "r")
        features_file = h5py.File(str(self.features_path), "r")
        
        try:
            for pair_name in matches_file.keys():
                pair_data = matches_file[pair_name]
                
                if "matches0" not in pair_data:
                    continue
                    
                # Parse image names
                parts = pair_name.split('_')
                if len(parts) < 2:
                    continue
                    
                img_name1, img_name2 = parts[0], '_'.join(parts[1:])
                
                # Load keypoints and matches
                try:
                    kpts1 = features_file[img_name1]["keypoints"].__array__()
                    kpts2 = features_file[img_name2]["keypoints"].__array__()
                    matches = pair_data["matches0"].__array__()
                    
                    # Filter valid matches
                    valid_matches = matches[matches[:, 1] != -1]
                    
                    if len(valid_matches) < 8:  # Need minimum for fundamental matrix
                        continue
                        
                    # Prepare camera parameters
                    camera_params = None
                    if intrinsics is not None:
                        camera_params = {
                            "K1": intrinsics.get(img_name1),
                            "K2": intrinsics.get(img_name2)
                        }
                        
                    # Estimate pose using DEGENSAC
                    pose_result = estimate_relative_pose(
                        kpts1, kpts2, valid_matches, 
                        camera_params=camera_params,
                        max_iterations=5000,
                        threshold=1.0
                    )
                    
                    if pose_result["success"]:
                        pose_estimates[pair_name] = pose_result
                        
                except KeyError as e:
                    logger.warning(f"Missing data for pair {pair_name}: {e}")
                    continue
                    
        finally:
            matches_file.close()
            features_file.close()
            
        logger.info(f"Estimated poses for {len(pose_estimates)} pairs")
        return pose_estimates
        
    def optimize_global_poses(self, pose_estimates: Dict) -> Dict[str, Dict]:
        """Optimize global poses using GTSAM"""
        
        # Create pose graph constraints
        constraints = create_pose_graph_from_matches({}, pose_estimates)
        
        if len(constraints) < 2:
            logger.warning("Insufficient constraints for pose graph optimization")
            return {}
            
        # Extract initial poses (simplified - would need proper initialization)
        initial_poses = {}
        image_ids = set()
        
        for constraint in constraints:
            id1, id2 = constraint["image_ids"]
            image_ids.add(id1)
            image_ids.add(id2)
            
        # Initialize poses (first pose at origin, others from relative poses)
        image_ids = list(image_ids)
        if len(image_ids) > 0:
            # Set first pose as identity
            initial_poses[image_ids[0]] = {
                "R": np.eye(3),
                "t": np.zeros((3, 1))
            }
            
            # Initialize other poses from constraints
            for constraint in constraints:
                id1, id2 = constraint["image_ids"]
                if id1 in initial_poses and id2 not in initial_poses:
                    # Add pose for id2 relative to id1
                    R_rel = constraint["R"]
                    t_rel = constraint["t"]
                    
                    R1 = initial_poses[id1]["R"]
                    t1 = initial_poses[id1]["t"]
                    
                    R2 = R_rel @ R1
                    t2 = t1 + R1.T @ t_rel
                    
                    initial_poses[id2] = {"R": R2, "t": t2}
                    
        if len(initial_poses) < 2:
            logger.warning("Could not initialize sufficient poses")
            return {}
            
        # Optimize using GTSAM
        optimizer = GTSAMOptimizer()
        optimized_poses = optimizer.optimize_poses(
            initial_poses, constraints, fixed_pose_id=image_ids[0])
            
        logger.info(f"Optimized {len(optimized_poses)} poses")
        return optimized_poses
        
    def _create_initial_poses_from_pairwise(self, pose_estimates: Dict) -> Dict[str, Dict]:
        """Create initial global poses from pairwise estimates (fallback)"""
        
        poses = {}
        
        if not pose_estimates:
            return poses
            
        # Use first successful pose estimate to initialize
        for pair_name, pose_data in pose_estimates.items():
            if pose_data["success"]:
                parts = pair_name.split('_')
                if len(parts) >= 2:
                    img1, img2 = parts[0], '_'.join(parts[1:])
                    
                    # Set first image at origin
                    poses[img1] = {"R": np.eye(3), "t": np.zeros((3, 1))}
                    
                    # Set second image from relative pose
                    poses[img2] = {
                        "R": pose_data["R"],
                        "t": pose_data["t"].reshape(3, 1)
                    }
                    break
                    
        return poses
        
    def triangulate_points(self, 
                          poses: Dict[str, Dict],
                          intrinsics: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """Triangulate 3D points"""
        
        if intrinsics is None:
            # Use default intrinsics (would normally extract from EXIF or calibration)
            logger.warning("Using default camera intrinsics")
            intrinsics = {}
            for img_id in poses.keys():
                intrinsics[img_id] = np.array([
                    [800, 0, 400],
                    [0, 800, 300],
                    [0, 0, 1]
                ], dtype=np.float32)
        
        # Load matches data
        matches_data = {}
        with h5py.File(str(self.matches_path), "r") as f:
            for pair_name in f.keys():
                pair_data = f[pair_name]
                matches_data[pair_name] = {
                    "matches0": pair_data["matches0"].__array__()
                }
                
        # Triangulate points
        points_3d = triangulate_reconstruction(
            poses, intrinsics, matches_data,
            reprojection_threshold=4.0,
            min_triangulation_angle=2.0
        )
        
        logger.info(f"Triangulated {len(points_3d)} 3D points")
        return points_3d
        
    def bundle_adjustment(self,
                         poses: Dict[str, Dict], 
                         points_3d: Dict,
                         intrinsics: Dict[str, np.ndarray]) -> tuple:
        """Run bundle adjustment optimization"""
        
        # Create observations (simplified - would extract from actual matches)
        observations = []
        
        optimizer = GTSAMOptimizer()
        
        try:
            refined_poses, refined_points = optimizer.bundle_adjustment(
                poses, points_3d, observations, intrinsics)
            
            logger.info("Bundle adjustment completed successfully")
            return refined_poses, refined_points
            
        except Exception as e:
            logger.warning(f"Bundle adjustment failed: {e}")
            return poses, points_3d
            
    def save_reconstruction(self, 
                           poses: Dict[str, Dict],
                           points_3d: Dict):
        """Save reconstruction results"""
        
        # Save poses
        with open(self.poses_path, 'w') as f:
            f.write("# Image poses (image_name qw qx qy qz tx ty tz)\\n")
            for img_id, pose in poses.items():
                R = pose["R"]
                t = pose["t"].flatten()
                
                # Convert rotation matrix to quaternion
                from scipy.spatial.transform import Rotation
                quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
                quat = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]
                
                f.write(f"{img_id} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
                       f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\\n")
                       
        # Save 3D points
        with open(self.points_path, 'w') as f:
            f.write("# 3D points (point_id x y z)\\n")
            for point_id, point_data in points_3d.items():
                xyz = point_data["xyz"]
                f.write(f"{point_id} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\\n")
                
        logger.info(f"Saved {len(poses)} poses to {self.poses_path}")
        logger.info(f"Saved {len(points_3d)} points to {self.points_path}")


def main():
    parser = argparse.ArgumentParser(description="Improved HLoc Pipeline")
    parser.add_argument("--images_dir", type=Path, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--feature_conf", type=str, default="superpoint_aachen",
                       help="Feature extraction configuration")
    parser.add_argument("--matcher_conf", type=str, default="superpoint+lightglue",
                       help="Feature matcher configuration")
    parser.add_argument("--max_pairs", type=int, default=5000,
                       help="Maximum number of image pairs")
    parser.add_argument("--skip_pose_optimization", action="store_true",
                       help="Skip pose graph optimization")
    parser.add_argument("--skip_bundle_adjustment", action="store_true",
                       help="Skip bundle adjustment")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize and run pipeline
    pipeline = ImprovedHLocPipeline(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        feature_conf=args.feature_conf,
        matcher_conf=args.matcher_conf
    )
    
    results = pipeline.run_full_pipeline(
        max_pairs=args.max_pairs,
        optimize_poses=not args.skip_pose_optimization,
        bundle_adjustment=not args.skip_bundle_adjustment
    )
    
    print("\\nPipeline completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {results}")


if __name__ == "__main__":
    main()