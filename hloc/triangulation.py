import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import pycolmap
from tqdm import tqdm

from . import logger
from .utils.database import COLMAPDatabase
from .utils.geometry import compute_epipolar_errors
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_retrieval


class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = False

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = True


def create_db_from_model(
    reconstruction: pycolmap.Reconstruction, database_path: Path
) -> Dict[str, int]:
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in reconstruction.cameras.items():
        db.add_camera(
            camera.model.value,
            camera.width,
            camera.height,
            camera.params,
            camera_id=i,
            prior_focal_length=True,
        )

    for i, image in reconstruction.images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in reconstruction.images.items()}


def import_features(
    image_ids: Dict[str, int], database_path: Path, features_path: Path
):
    logger.info("Importing features into the database...")
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def import_matches(
    image_ids: Dict[str, int],
    database_path: Path,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
    skip_geometric_verification: bool = False,
):
    logger.info("Importing matches into the database...")

    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()


def estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    logger.info("Performing LoRANSAC geometric verification of the matches...")
    loransac_estimation_and_geometric_verification(database_path, pairs_path, verbose)


def loransac_estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Enhanced geometric verification using true LoRANSAC with PyRANSAC"""
    from .utils.database import image_ids_to_pair_id, array_to_blob, blob_to_array
    
    try:
        from pyransac import LoRANSAC
        from pyransac.base import Model
        import numpy as np
        
        class FundamentalMatrixModel(Model):
            """Fundamental Matrix model for LoRANSAC"""
            def __init__(self):
                super().__init__()
                
            def fit(self, data):
                """Fit fundamental matrix using 8-point algorithm"""
                if len(data) < 8:
                    return None
                    
                pts0, pts1 = data[:, :2], data[:, 2:]
                
                # Normalize points
                pts0_norm = cv2.undistortPoints(pts0.reshape(-1, 1, 2), np.eye(3), None).reshape(-1, 2)
                pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), np.eye(3), None).reshape(-1, 2)
                
                # 8-point algorithm
                A = np.zeros((len(pts0_norm), 9))
                for i in range(len(pts0_norm)):
                    x0, y0 = pts0_norm[i]
                    x1, y1 = pts1_norm[i]
                    A[i] = [x0*x1, x0*y1, x0, y0*x1, y0*y1, y0, x1, y1, 1]
                
                # Solve using SVD
                _, _, Vt = np.linalg.svd(A)
                F = Vt[-1].reshape(3, 3)
                
                # Enforce rank-2 constraint
                U, S, Vt = np.linalg.svd(F)
                S[2] = 0
                F = U @ np.diag(S) @ Vt
                
                return F
                
            def residuals(self, data, model):
                """Compute Sampson distance residuals"""
                if model is None:
                    return np.inf * np.ones(len(data))
                    
                pts0, pts1 = data[:, :2], data[:, 2:]
                
                # Convert to homogeneous coordinates
                pts0_h = np.column_stack([pts0, np.ones(len(pts0))])
                pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
                
                # Compute Sampson distance
                Fx = model @ pts0_h.T
                Ftx = model.T @ pts1_h.T
                
                # Numerator: (x'^T F x)^2
                numerator = (pts1_h * (model @ pts0_h.T).T).sum(axis=1) ** 2
                
                # Denominator: (Fx)_1^2 + (Fx)_2^2 + (F^T x')_1^2 + (F^T x')_2^2
                denominator = Fx[0]**2 + Fx[1]**2 + Ftx[0]**2 + Ftx[1]**2
                
                # Avoid division by zero
                denominator = np.where(denominator < 1e-10, 1e-10, denominator)
                
                return numerator / denominator
                
        class EssentialMatrixModel(Model):
            """Essential Matrix model for LoRANSAC"""
            def __init__(self, K1, K2):
                super().__init__()
                self.K1 = K1
                self.K2 = K2
                
            def fit(self, data):
                """Fit essential matrix using 5-point algorithm"""
                if len(data) < 5:
                    return None
                    
                pts0, pts1 = data[:, :2], data[:, 2:]
                
                # Normalize points using camera intrinsics
                pts0_norm = cv2.undistortPoints(pts0.reshape(-1, 1, 2), self.K1, None).reshape(-1, 2)
                pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K2, None).reshape(-1, 2)
                
                # Use OpenCV's 5-point algorithm
                E, mask = cv2.findEssentialMat(
                    pts0_norm, pts1_norm, 
                    focal=1.0, pp=(0., 0.),
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                
                return E
                
            def residuals(self, data, model):
                """Compute epipolar constraint residuals"""
                if model is None:
                    return np.inf * np.ones(len(data))
                    
                pts0, pts1 = data[:, :2], data[:, 2:]
                
                # Normalize points
                pts0_norm = cv2.undistortPoints(pts0.reshape(-1, 1, 2), self.K1, None).reshape(-1, 2)
                pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K2, None).reshape(-1, 2)
                
                # Convert to homogeneous coordinates
                pts0_h = np.column_stack([pts0_norm, np.ones(len(pts0_norm))])
                pts1_h = np.column_stack([pts1_norm, np.ones(len(pts1_norm))])
                
                # Compute epipolar constraint: x'^T E x = 0
                residuals = np.abs((pts1_h * (model @ pts0_h.T).T).sum(axis=1))
                
                return residuals
        
    except ImportError:
        logger.warning("PyRANSAC not available, falling back to standard RANSAC")
        # Fallback to standard RANSAC implementation
        return standard_ransac_verification(database_path, pairs_path, verbose)
    
    # LoRANSAC configuration
    loransac_config = {
        'threshold': 2.0,  # Inlier threshold in pixels
        'confidence': 0.999,  # Confidence level
        'max_iterations': 10000,  # Maximum iterations
        'min_inlier_ratio': 0.15,  # Minimum inlier ratio
        'min_inliers': 8  # Minimum number of inliers
    }
    
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    db = COLMAPDatabase.connect(database_path)
    
    # Get camera information
    cameras = {}
    for row in db.execute("SELECT camera_id, model, width, height, params FROM cameras"):
        camera_id, model, width, height, params = row
        params = blob_to_array(params, np.float64)
        cameras[camera_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
    
    # Process each image pair with LoRANSAC
    processed_pairs = set()
    successful_verifications = 0
    total_pairs = 0
    
    for name0, name1 in tqdm(pairs, desc="LoRANSAC verification"):
        # Get image IDs and camera info
        image_id0_result = db.execute("SELECT image_id, camera_id FROM images WHERE name=?", (name0,)).fetchone()
        image_id1_result = db.execute("SELECT image_id, camera_id FROM images WHERE name=?", (name1,)).fetchone()
        
        if image_id0_result is None or image_id1_result is None:
            continue
            
        image_id0, camera_id0 = image_id0_result
        image_id1, camera_id1 = image_id1_result
        
        # Skip if already processed
        pair_key = tuple(sorted([image_id0, image_id1]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        # Get pair_id for database queries
        pair_id = image_ids_to_pair_id(image_id0, image_id1)
        
        # Get keypoints and matches
        kps0_result = db.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id0,)).fetchone()
        kps1_result = db.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id1,)).fetchone()
        matches_result = db.execute("SELECT data FROM matches WHERE pair_id=?", (pair_id,)).fetchone()
        
        if kps0_result is None or kps1_result is None or matches_result is None:
            continue
        
        # Decode keypoints and matches
        kps0 = blob_to_array(kps0_result[0], np.float32).reshape(-1, 2)
        kps1 = blob_to_array(kps1_result[0], np.float32).reshape(-1, 2)
        matches = blob_to_array(matches_result[0], np.uint32).reshape(-1, 2)
        
        total_pairs += 1
        
        if len(matches) < 8:
            # Add empty geometry for pairs with insufficient matches
            db.add_two_view_geometry(
                image_id0, image_id1, 
                matches=np.array([]).reshape(0, 2),
                F=np.eye(3), E=np.eye(3), H=np.eye(3),
                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                config=0
            )
            continue
        
        # Extract matched keypoints
        pts0 = kps0[matches[:, 0]]
        pts1 = kps1[matches[:, 1]]
        
        # Prepare data for LoRANSAC
        data = np.column_stack([pts0, pts1])
        
        # Get camera parameters
        cam0 = cameras[camera_id0]
        cam1 = cameras[camera_id1]
        
        # Try essential matrix first if we have camera intrinsics
        if cam0['model'] == 0 and cam1['model'] == 0:  # Pinhole camera
            K0 = np.array([
                [cam0['params'][0], 0, cam0['params'][2]],
                [0, cam0['params'][1], cam0['params'][3]],
                [0, 0, 1]
            ])
            K1 = np.array([
                [cam1['params'][0], 0, cam1['params'][2]],
                [0, cam1['params'][1], cam1['params'][3]],
                [0, 0, 1]
            ])
            
            # Use LoRANSAC with essential matrix model
            try:
                model = EssentialMatrixModel(K0, K1)
                loransac = LoRANSAC(
                    model=model,
                    threshold=loransac_config['threshold'],
                    confidence=loransac_config['confidence'],
                    max_iterations=loransac_config['max_iterations']
                )
                
                E, inliers = loransac.fit(data)
                
                if E is not None and len(inliers) >= loransac_config['min_inliers']:
                    inlier_ratio = len(inliers) / len(matches)
                    
                    if inlier_ratio >= loransac_config['min_inlier_ratio']:
                        inlier_matches = matches[inliers]
                        inlier_pts0 = pts0[inliers]
                        inlier_pts1 = pts1[inliers]
                        
                        # Recover pose from essential matrix
                        _, R, t, _ = cv2.recoverPose(E, inlier_pts0, inlier_pts1, K0, K1)
                        
                        # Convert rotation matrix to quaternion
                        qvec = rotation_matrix_to_quaternion(R)
                        
                        # Compute fundamental matrix from essential matrix
                        F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
                        
                        # Store two-view geometry
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=inlier_matches,
                            F=F, E=E, H=np.eye(3),
                            qvec=qvec, tvec=t.ravel(),
                            config=len(inlier_matches)
                        )
                        
                        successful_verifications += 1
                        
                        if verbose:
                            logger.info(f"LoRANSAC Essential: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                    else:
                        # Try fundamental matrix as fallback
                        model = FundamentalMatrixModel()
                        loransac = LoRANSAC(
                            model=model,
                            threshold=loransac_config['threshold'],
                            confidence=loransac_config['confidence'],
                            max_iterations=loransac_config['max_iterations']
                        )
                        
                        F, inliers = loransac.fit(data)
                        
                        if F is not None and len(inliers) >= loransac_config['min_inliers']:
                            inlier_ratio = len(inliers) / len(matches)
                            
                            if inlier_ratio >= loransac_config['min_inlier_ratio']:
                                inlier_matches = matches[inliers]
                                
                                db.add_two_view_geometry(
                                    image_id0, image_id1,
                                    matches=inlier_matches,
                                    F=F, E=np.eye(3), H=np.eye(3),
                                    qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                    config=len(inlier_matches)
                                )
                                
                                successful_verifications += 1
                                
                                if verbose:
                                    logger.info(f"LoRANSAC Fundamental: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                            else:
                                # Add empty geometry for low inlier ratio
                                db.add_two_view_geometry(
                                    image_id0, image_id1,
                                    matches=np.array([]).reshape(0, 2),
                                    F=np.eye(3), E=np.eye(3), H=np.eye(3),
                                    qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                    config=0
                                )
                        else:
                            # Add empty geometry for failed estimation
                            db.add_two_view_geometry(
                                image_id0, image_id1,
                                matches=np.array([]).reshape(0, 2),
                                F=np.eye(3), E=np.eye(3), H=np.eye(3),
                                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                config=0
                            )
                else:
                    # Essential matrix failed, try fundamental matrix
                    model = FundamentalMatrixModel()
                    loransac = LoRANSAC(
                        model=model,
                        threshold=loransac_config['threshold'],
                        confidence=loransac_config['confidence'],
                        max_iterations=loransac_config['max_iterations']
                    )
                    
                    F, inliers = loransac.fit(data)
                    
                    if F is not None and len(inliers) >= loransac_config['min_inliers']:
                        inlier_ratio = len(inliers) / len(matches)
                        
                        if inlier_ratio >= loransac_config['min_inlier_ratio']:
                            inlier_matches = matches[inliers]
                            
                            db.add_two_view_geometry(
                                image_id0, image_id1,
                                matches=inlier_matches,
                                F=F, E=np.eye(3), H=np.eye(3),
                                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                config=len(inlier_matches)
                            )
                            
                            successful_verifications += 1
                            
                            if verbose:
                                logger.info(f"LoRANSAC Fundamental: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                        else:
                            # Add empty geometry for low inlier ratio
                            db.add_two_view_geometry(
                                image_id0, image_id1,
                                matches=np.array([]).reshape(0, 2),
                                F=np.eye(3), E=np.eye(3), H=np.eye(3),
                                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                config=0
                            )
                    else:
                        # Add empty geometry for failed estimation
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=np.array([]).reshape(0, 2),
                            F=np.eye(3), E=np.eye(3), H=np.eye(3),
                            qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                            config=0
                        )
                        
            except Exception as e:
                if verbose:
                    logger.warning(f"LoRANSAC failed for {name0}-{name1}: {e}")
                # Add empty geometry for failed cases
                db.add_two_view_geometry(
                    image_id0, image_id1,
                    matches=np.array([]).reshape(0, 2),
                    F=np.eye(3), E=np.eye(3), H=np.eye(3),
                    qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                    config=0
                )
        else:
            # Use fundamental matrix only for non-pinhole cameras
            model = FundamentalMatrixModel()
            loransac = LoRANSAC(
                model=model,
                threshold=loransac_config['threshold'],
                confidence=loransac_config['confidence'],
                max_iterations=loransac_config['max_iterations']
            )
            
            try:
                F, inliers = loransac.fit(data)
                
                if F is not None and len(inliers) >= loransac_config['min_inliers']:
                    inlier_ratio = len(inliers) / len(matches)
                    
                    if inlier_ratio >= loransac_config['min_inlier_ratio']:
                        inlier_matches = matches[inliers]
                        
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=inlier_matches,
                            F=F, E=np.eye(3), H=np.eye(3),
                            qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                            config=len(inlier_matches)
                        )
                        
                        successful_verifications += 1
                        
                        if verbose:
                            logger.info(f"LoRANSAC Fundamental: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                    else:
                        # Add empty geometry for low inlier ratio
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=np.array([]).reshape(0, 2),
                            F=np.eye(3), E=np.eye(3), H=np.eye(3),
                            qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                            config=0
                        )
                else:
                    # Add empty geometry for failed estimation
                    db.add_two_view_geometry(
                        image_id0, image_id1,
                        matches=np.array([]).reshape(0, 2),
                        F=np.eye(3), E=np.eye(3), H=np.eye(3),
                        qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                        config=0
                    )
                    
            except Exception as e:
                if verbose:
                    logger.warning(f"LoRANSAC failed for {name0}-{name1}: {e}")
                # Add empty geometry for failed cases
                db.add_two_view_geometry(
                    image_id0, image_id1,
                    matches=np.array([]).reshape(0, 2),
                    F=np.eye(3), E=np.eye(3), H=np.eye(3),
                    qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                    config=0
                )
    
    db.commit()
    db.close()
    
    success_rate = successful_verifications / max(total_pairs, 1) * 100
    logger.info(f"LoRANSAC geometric verification completed: {successful_verifications}/{total_pairs} pairs verified successfully ({success_rate:.1f}%)")


def standard_ransac_verification(database_path: Path, pairs_path: Path, verbose: bool = False):
    """Fallback to standard RANSAC when PyRANSAC is not available"""
    from .utils.database import image_ids_to_pair_id, array_to_blob, blob_to_array
    
    # Standard RANSAC configuration
    ransac_config = {
        'threshold': 2.0,
        'confidence': 0.999,
        'max_num_trials': 10000,
        'min_inlier_ratio': 0.15,
        'min_inliers': 8
    }
    
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    db = COLMAPDatabase.connect(database_path)
    
    # Get camera information
    cameras = {}
    for row in db.execute("SELECT camera_id, model, width, height, params FROM cameras"):
        camera_id, model, width, height, params = row
        params = blob_to_array(params, np.float64)
        cameras[camera_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
    
    processed_pairs = set()
    successful_verifications = 0
    total_pairs = 0
    
    for name0, name1 in tqdm(pairs, desc="Standard RANSAC verification"):
        image_id0_result = db.execute("SELECT image_id, camera_id FROM images WHERE name=?", (name0,)).fetchone()
        image_id1_result = db.execute("SELECT image_id, camera_id FROM images WHERE name=?", (name1,)).fetchone()
        
        if image_id0_result is None or image_id1_result is None:
            continue
            
        image_id0, camera_id0 = image_id0_result
        image_id1, camera_id1 = image_id1_result
        
        pair_key = tuple(sorted([image_id0, image_id1]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        pair_id = image_ids_to_pair_id(image_id0, image_id1)
        
        kps0_result = db.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id0,)).fetchone()
        kps1_result = db.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id1,)).fetchone()
        matches_result = db.execute("SELECT data FROM matches WHERE pair_id=?", (pair_id,)).fetchone()
        
        if kps0_result is None or kps1_result is None or matches_result is None:
            continue
        
        kps0 = blob_to_array(kps0_result[0], np.float32).reshape(-1, 2)
        kps1 = blob_to_array(kps1_result[0], np.float32).reshape(-1, 2)
        matches = blob_to_array(matches_result[0], np.uint32).reshape(-1, 2)
        
        total_pairs += 1
        
        if len(matches) < 8:
            db.add_two_view_geometry(
                image_id0, image_id1, 
                matches=np.array([]).reshape(0, 2),
                F=np.eye(3), E=np.eye(3), H=np.eye(3),
                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                config=0
            )
            continue
        
        pts0 = kps0[matches[:, 0]]
        pts1 = kps1[matches[:, 1]]
        
        cam0 = cameras[camera_id0]
        cam1 = cameras[camera_id1]
        
        try:
            # Try essential matrix first
            if cam0['model'] == 0 and cam1['model'] == 0:
                K0 = np.array([
                    [cam0['params'][0], 0, cam0['params'][2]],
                    [0, cam0['params'][1], cam0['params'][3]],
                    [0, 0, 1]
                ])
                K1 = np.array([
                    [cam1['params'][0], 0, cam1['params'][2]],
                    [0, cam1['params'][1], cam1['params'][3]],
                    [0, 0, 1]
                ])
                
                E, mask = cv2.findEssentialMat(
                    pts0, pts1, K0, K1,
                    method=cv2.RANSAC,
                    prob=ransac_config['confidence'],
                    threshold=ransac_config['threshold'],
                    maxIters=ransac_config['max_num_trials']
                )
                
                if E is not None and mask is not None:
                    inlier_mask = mask.ravel() == 1
                    inlier_ratio = np.sum(inlier_mask) / len(matches)
                    
                    if (inlier_ratio >= ransac_config['min_inlier_ratio'] and 
                        np.sum(inlier_mask) >= ransac_config['min_inliers']):
                        
                        inlier_matches = matches[inlier_mask]
                        inlier_pts0 = pts0[inlier_mask]
                        inlier_pts1 = pts1[inlier_mask]
                        
                        _, R, t, _ = cv2.recoverPose(E, inlier_pts0, inlier_pts1, K0, K1)
                        qvec = rotation_matrix_to_quaternion(R)
                        F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
                        
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=inlier_matches,
                            F=F, E=E, H=np.eye(3),
                            qvec=qvec, tvec=t.ravel(),
                            config=len(inlier_matches)
                        )
                        
                        successful_verifications += 1
                        
                        if verbose:
                            logger.info(f"RANSAC Essential: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                    else:
                        # Try fundamental matrix as fallback
                        F, mask = cv2.findFundamentalMat(
                            pts0, pts1,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=ransac_config['threshold'],
                            confidence=ransac_config['confidence'],
                            maxIters=ransac_config['max_num_trials']
                        )
                        
                        if F is not None and mask is not None:
                            inlier_mask = mask.ravel() == 1
                            inlier_ratio = np.sum(inlier_mask) / len(matches)
                            
                            if (inlier_ratio >= ransac_config['min_inlier_ratio'] and 
                                np.sum(inlier_mask) >= ransac_config['min_inliers']):
                                
                                inlier_matches = matches[inlier_mask]
                                
                                db.add_two_view_geometry(
                                    image_id0, image_id1,
                                    matches=inlier_matches,
                                    F=F, E=np.eye(3), H=np.eye(3),
                                    qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                    config=len(inlier_matches)
                                )
                                
                                successful_verifications += 1
                                
                                if verbose:
                                    logger.info(f"RANSAC Fundamental: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                            else:
                                db.add_two_view_geometry(
                                    image_id0, image_id1,
                                    matches=np.array([]).reshape(0, 2),
                                    F=np.eye(3), E=np.eye(3), H=np.eye(3),
                                    qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                    config=0
                                )
                        else:
                            db.add_two_view_geometry(
                                image_id0, image_id1,
                                matches=np.array([]).reshape(0, 2),
                                F=np.eye(3), E=np.eye(3), H=np.eye(3),
                                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                config=0
                            )
                else:
                    # Essential matrix failed, try fundamental matrix
                    F, mask = cv2.findFundamentalMat(
                        pts0, pts1,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=ransac_config['threshold'],
                        confidence=ransac_config['confidence'],
                        maxIters=ransac_config['max_num_trials']
                    )
                    
                    if F is not None and mask is not None:
                        inlier_mask = mask.ravel() == 1
                        inlier_ratio = np.sum(inlier_mask) / len(matches)
                        
                        if (inlier_ratio >= ransac_config['min_inlier_ratio'] and 
                            np.sum(inlier_mask) >= ransac_config['min_inliers']):
                            
                            inlier_matches = matches[inlier_mask]
                            
                            db.add_two_view_geometry(
                                image_id0, image_id1,
                                matches=inlier_matches,
                                F=F, E=np.eye(3), H=np.eye(3),
                                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                config=len(inlier_matches)
                            )
                            
                            successful_verifications += 1
                            
                            if verbose:
                                logger.info(f"RANSAC Fundamental: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                        else:
                            db.add_two_view_geometry(
                                image_id0, image_id1,
                                matches=np.array([]).reshape(0, 2),
                                F=np.eye(3), E=np.eye(3), H=np.eye(3),
                                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                                config=0
                            )
                    else:
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=np.array([]).reshape(0, 2),
                            F=np.eye(3), E=np.eye(3), H=np.eye(3),
                            qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                            config=0
                        )
            else:
                # Use fundamental matrix only for non-pinhole cameras
                F, mask = cv2.findFundamentalMat(
                    pts0, pts1,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_config['threshold'],
                    confidence=ransac_config['confidence'],
                    maxIters=ransac_config['max_num_trials']
                )
                
                if F is not None and mask is not None:
                    inlier_mask = mask.ravel() == 1
                    inlier_ratio = np.sum(inlier_mask) / len(matches)
                    
                    if (inlier_ratio >= ransac_config['min_inlier_ratio'] and 
                        np.sum(inlier_mask) >= ransac_config['min_inliers']):
                        
                        inlier_matches = matches[inlier_mask]
                        
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=inlier_matches,
                            F=F, E=np.eye(3), H=np.eye(3),
                            qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                            config=len(inlier_matches)
                        )
                        
                        successful_verifications += 1
                        
                        if verbose:
                            logger.info(f"RANSAC Fundamental: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                    else:
                        db.add_two_view_geometry(
                            image_id0, image_id1,
                            matches=np.array([]).reshape(0, 2),
                            F=np.eye(3), E=np.eye(3), H=np.eye(3),
                            qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                            config=0
                        )
                else:
                    db.add_two_view_geometry(
                        image_id0, image_id1,
                        matches=np.array([]).reshape(0, 2),
                        F=np.eye(3), E=np.eye(3), H=np.eye(3),
                        qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                        config=0
                    )
                    
        except Exception as e:
            if verbose:
                logger.warning(f"RANSAC failed for {name0}-{name1}: {e}")
            db.add_two_view_geometry(
                image_id0, image_id1,
                matches=np.array([]).reshape(0, 2),
                F=np.eye(3), E=np.eye(3), H=np.eye(3),
                qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                config=0
            )
    
    db.commit()
    db.close()
    
    success_rate = successful_verifications / max(total_pairs, 1) * 100
    logger.info(f"Standard RANSAC geometric verification completed: {successful_verifications}/{total_pairs} pairs verified successfully ({success_rate:.1f}%)")


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    return np.array([w, x, y, z])


def geometric_verification(
    image_ids: Dict[str, int],
    reference: pycolmap.Reconstruction,
    database_path: Path,
    features_path: Path,
    pairs_path: Path,
    matches_path: Path,
    max_error: float = 4.0,
):
    logger.info("Performing geometric verification of the matches...")

    pairs = parse_retrieval(pairs_path)
    db = COLMAPDatabase.connect(database_path)

    inlier_ratios = []
    matched = set()
    for name0 in tqdm(pairs):
        id0 = image_ids[name0]
        image0 = reference.images[id0]
        cam0 = reference.cameras[image0.camera_id]
        kps0, noise0 = get_keypoints(features_path, name0, return_uncertainty=True)
        noise0 = 1.0 if noise0 is None else noise0
        if len(kps0) > 0:
            kps0 = np.stack(cam0.cam_from_img(kps0))
        else:
            kps0 = np.zeros((0, 2))

        for name1 in pairs[name0]:
            id1 = image_ids[name1]
            image1 = reference.images[id1]
            cam1 = reference.cameras[image1.camera_id]
            kps1, noise1 = get_keypoints(features_path, name1, return_uncertainty=True)
            noise1 = 1.0 if noise1 is None else noise1
            if len(kps1) > 0:
                kps1 = np.stack(cam1.cam_from_img(kps1))
            else:
                kps1 = np.zeros((0, 2))

            matches = get_matches(matches_path, name0, name1)[0]

            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue
            matched |= {(id0, id1), (id1, id0)}

            if matches.shape[0] == 0:
                db.add_two_view_geometry(id0, id1, matches)
                continue

            cam1_from_cam0 = image1.cam_from_world() * image0.cam_from_world().inverse()
            errors0, errors1 = compute_epipolar_errors(
                cam1_from_cam0, kps0[matches[:, 0]], kps1[matches[:, 1]]
            )
            valid_matches = np.logical_and(
                errors0 <= cam0.cam_from_img_threshold(noise0 * max_error),
                errors1 <= cam1.cam_from_img_threshold(noise1 * max_error),
            )
            # TODO: We could also add E to the database, but we need
            # to reverse the transformations if id0 > id1 in utils/database.py.
            db.add_two_view_geometry(id0, id1, matches[valid_matches, :])
            inlier_ratios.append(np.mean(valid_matches))
    logger.info(
        "mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.",
        np.mean(inlier_ratios) * 100,
        np.median(inlier_ratios) * 100,
        np.min(inlier_ratios) * 100,
        np.max(inlier_ratios) * 100,
    )

    db.commit()
    db.close()


def run_triangulation(
    model_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    model_path.mkdir(parents=True, exist_ok=True)
    logger.info("Running 3D triangulation...")
    if options is None:
        options = {}
    with OutputCapture(verbose):
        reconstruction = pycolmap.triangulate_points(
            reference_model, database_path, image_dir, model_path, options=options
        )
    return reconstruction


def main(
    sfm_dir: Path,
    reference_model: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    skip_geometric_verification: bool = False,
    estimate_two_view_geometries: bool = False,
    min_match_score: Optional[float] = None,
    verbose: bool = False,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    assert reference_model.exists(), reference_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    reference = pycolmap.Reconstruction(reference_model)

    image_ids = create_db_from_model(reference, database)
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    if not skip_geometric_verification:
        if estimate_two_view_geometries:
            estimation_and_geometric_verification(database, pairs, verbose)
        else:
            geometric_verification(
                image_ids, reference, database, features, pairs, matches
            )
    reconstruction = run_triangulation(
        sfm_dir, database, image_dir, reference, verbose, mapper_options
    )
    logger.info(
        "Finished the triangulation with statistics:\n%s", reconstruction.summary()
    )
    return reconstruction


def parse_option_args(args: List[str], default_options) -> Dict[str, Any]:
    options = {}
    for arg in args:
        idx = arg.find("=")
        if idx == -1:
            raise ValueError("Options format: key1=value1 key2=value2 etc.")
        key, value = arg[:idx], arg[idx + 1 :]
        if not hasattr(default_options, key):
            raise ValueError(
                f'Unknown option "{key}", allowed options and default values'
                f" for {default_options.summary()}"
            )
        value = eval(value)
        target_type = type(getattr(default_options, key))
        if not isinstance(value, target_type):
            raise ValueError(
                f'Incorrect type for option "{key}":' f" {type(value)} vs {target_type}"
            )
        options[key] = value
    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--reference_sfm_model", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, mapper_options=mapper_options)
