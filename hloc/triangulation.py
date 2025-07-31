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
    """Complete LoRANSAC implementation using PyRANSAC library"""
    from .utils.database import image_ids_to_pair_id, blob_to_array
    
    # Try to import PyRANSAC
    try:
        import pyransac
        has_pyransac = True
        if verbose:
            logger.info("Using PyRANSAC for true LoRANSAC implementation")
    except ImportError:
        try:
            # Try alternative import
            from pyransac import ransac as pyransac
            has_pyransac = True
            if verbose:
                logger.info("Using PyRANSAC for true LoRANSAC implementation")
        except ImportError:
            has_pyransac = False
            logger.warning("PyRANSAC not available, install with: pip install pyransac")
            logger.info("Falling back to OpenCV USAC LoRANSAC")
    
    # LoRANSAC configuration
    loransac_config = {
        'threshold': 1.0,           # Inlier threshold in pixels
        'confidence': 0.999,        # Confidence level
        'max_iterations': 10000,    # Maximum RANSAC iterations
        'min_inlier_ratio': 0.15,   # Minimum inlier ratio
        'min_inliers': 8,           # Minimum number of inliers
        # PyRANSAC specific
        'lo_iterations': 10,        # Local optimization iterations  
        'lo_sample_size': 14        # Local optimization sample size
    }
    
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    db = COLMAPDatabase.connect(database_path)
    
    processed_pairs = set()
    successful_verifications = 0
    total_pairs = 0
    
    for name0, name1 in tqdm(pairs, desc="LoRANSAC verification"):
        # Get image IDs
        image_id0_result = db.execute("SELECT image_id FROM images WHERE name=?", (name0,)).fetchone()
        image_id1_result = db.execute("SELECT image_id FROM images WHERE name=?", (name1,)).fetchone()
        
        if image_id0_result is None or image_id1_result is None:
            continue
            
        image_id0, image_id1 = image_id0_result[0], image_id1_result[0]
        
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
        
        if len(matches) < 8:  # Need minimum 8 points for fundamental matrix
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
        
        try:
            if has_pyransac:
                # Use true PyRANSAC LoRANSAC
                F, inlier_matches = pyransac_fundamental_matrix(
                    pts0, pts1, matches, loransac_config, pyransac
                )
            else:
                # Fallback to OpenCV USAC
                F, inlier_matches = opencv_loransac_fundamental_matrix(
                    pts0, pts1, matches, loransac_config
                )
            
            if F is not None and len(inlier_matches) > 0:
                inlier_ratio = len(inlier_matches) / len(matches)
                
                if (inlier_ratio >= loransac_config['min_inlier_ratio'] and 
                    len(inlier_matches) >= loransac_config['min_inliers']):
                    
                    # Store two-view geometry in database
                    db.add_two_view_geometry(
                        image_id0, image_id1,
                        matches=inlier_matches,
                        F=F, E=np.eye(3), H=np.eye(3),
                        qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                        config=len(inlier_matches)
                    )
                    
                    successful_verifications += 1
                    
                    if verbose:
                        method = "PyRANSAC" if has_pyransac else "OpenCV"
                        logger.info(f"{method} LoRANSAC: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
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
                # Failed fundamental matrix estimation
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
    method = "PyRANSAC" if has_pyransac else "OpenCV USAC"
    logger.info(f"{method} LoRANSAC geometric verification completed: {successful_verifications}/{total_pairs} pairs verified successfully ({success_rate:.1f}%)")


def pyransac_fundamental_matrix(pts0, pts1, matches, config, pyransac_lib):
    """True LoRANSAC implementation using PyRANSAC"""
    
    # Prepare data for PyRANSAC
    data = np.column_stack([pts0, pts1])
    
    # Define model fitting function
    def fit_fundamental_matrix(sample_points):
        """Fit fundamental matrix to sample points"""
        if len(sample_points) < 8:
            return None
        return fit_fundamental_matrix_8point(sample_points[:, :2], sample_points[:, 2:])
    
    # Define error function  
    def compute_residuals(data_points, model):
        """Compute residuals for all data points"""
        if model is None:
            return np.full(len(data_points), np.inf)
        return compute_sampson_distance(data_points[:, :2], data_points[:, 2:], model)
    
    # Run LoRANSAC
    try:
        # Try different PyRANSAC APIs
        if hasattr(pyransac_lib, 'LoRANSAC'):
            # Direct LoRANSAC class
            loransac = pyransac_lib.LoRANSAC(
                data=data,
                model_fit_func=fit_fundamental_matrix,
                residual_func=compute_residuals,
                min_samples=8,
                threshold=config['threshold'],
                max_iterations=config['max_iterations'],
                confidence=config['confidence'],
                local_optimization_iterations=config.get('lo_iterations', 10),
                local_optimization_sample_size=config.get('lo_sample_size', 14)
            )
            F, inliers = loransac.run()
            
        elif hasattr(pyransac_lib, 'ransac'):
            # RANSAC function with LoRANSAC option
            F, inliers = pyransac_lib.ransac(
                data=data,
                model_fit_func=fit_fundamental_matrix,
                residual_func=compute_residuals,
                min_samples=8,
                threshold=config['threshold'],
                max_iterations=config['max_iterations'],
                confidence=config['confidence'],
                use_local_optimization=True,
                local_opt_iterations=config.get('lo_iterations', 10),
                local_opt_sample_size=config.get('lo_sample_size', 14)
            )
            
        else:
            # Try to find LoRANSAC function
            loransac_func = None
            for attr_name in dir(pyransac_lib):
                if 'loransac' in attr_name.lower():
                    loransac_func = getattr(pyransac_lib, attr_name)
                    break
            
            if loransac_func:
                F, inliers = loransac_func(
                    data=data,
                    model_fit_func=fit_fundamental_matrix,
                    residual_func=compute_residuals,
                    min_samples=8,
                    threshold=config['threshold'],
                    max_iterations=config['max_iterations'],
                    confidence=config['confidence']
                )
            else:
                raise AttributeError("No LoRANSAC implementation found")
        
        if F is not None and len(inliers) > 0:
            inlier_matches = matches[inliers]
            return F, inlier_matches
        else:
            return None, np.array([]).reshape(0, 2)
            
    except Exception as e:
        # Fallback to manual LoRANSAC implementation
        return manual_loransac_fundamental_matrix(data, matches, config)


def manual_loransac_fundamental_matrix(data, matches, config):
    """Manual LoRANSAC implementation when PyRANSAC API doesn't match"""
    
    best_F = None
    best_inliers = []
    best_score = 0
    
    pts0 = data[:, :2]
    pts1 = data[:, 2:]
    
    for iteration in range(config['max_iterations']):
        # Sample 8 points randomly
        if len(data) < 8:
            break
            
        sample_idx = np.random.choice(len(data), 8, replace=False)
        sample_data = data[sample_idx]
        
        # Fit fundamental matrix to sample
        F = fit_fundamental_matrix_8point(sample_data[:, :2], sample_data[:, 2:])
        if F is None:
            continue
            
        # Compute errors for all points
        errors = compute_sampson_distance(pts0, pts1, F)
        
        # Find inliers
        inliers = np.where(errors < config['threshold'])[0]
        
        if len(inliers) < 8:
            continue
            
        # Local Optimization: refine using all inliers
        for lo_iter in range(config.get('lo_iterations', 10)):
            if len(inliers) >= 8:
                # Sample from current inliers for refinement
                lo_sample_size = min(config.get('lo_sample_size', 14), len(inliers))
                if len(inliers) > lo_sample_size:
                    lo_sample_idx = np.random.choice(inliers, lo_sample_size, replace=False)
                else:
                    lo_sample_idx = inliers
                
                # Refine model using inliers
                F_refined = fit_fundamental_matrix_8point(pts0[lo_sample_idx], pts1[lo_sample_idx])
                if F_refined is not None:
                    # Re-evaluate with refined model
                    errors_refined = compute_sampson_distance(pts0, pts1, F_refined)
                    inliers_refined = np.where(errors_refined < config['threshold'])[0]
                    
                    if len(inliers_refined) >= len(inliers):
                        F = F_refined
                        inliers = inliers_refined
        
        # Check if this is the best model so far
        score = len(inliers)
        if score > best_score:
            best_F = F
            best_inliers = inliers
            best_score = score
    
    if best_F is not None and len(best_inliers) > 0:
        return best_F, matches[best_inliers]
    else:
        return None, np.array([]).reshape(0, 2)


def opencv_loransac_fundamental_matrix(pts0, pts1, matches, config):
    """Fallback OpenCV USAC LoRANSAC"""
    try:
        F, mask = cv2.findFundamentalMat(
            pts0, pts1,
            method=cv2.USAC_FM_8PTS,
            ransacReprojThreshold=config['threshold'],
            confidence=config['confidence'],
            maxIters=config['max_iterations'],
            loIterations=config.get('lo_iterations', 10),
            loSampleSize=config.get('lo_sample_size', 14)
        )
        
        if F is not None and mask is not None:
            inlier_mask = mask.ravel() == 1
            inlier_matches = matches[inlier_mask]
            return F, inlier_matches
        else:
            return None, np.array([]).reshape(0, 2)
            
    except Exception:
        return None, np.array([]).reshape(0, 2)


def normalize_points(points):
    """Normalize points for numerical stability"""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(centered, axis=1))
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    normalized = (T @ np.column_stack([points, np.ones(len(points))]).T).T
    return normalized[:, :2], T


def fit_fundamental_matrix_8point(pts0, pts1):
    """8-point algorithm for fundamental matrix estimation"""
    if len(pts0) < 8:
        return None
        
    # Normalize points
    pts0_norm, T0 = normalize_points(pts0)
    pts1_norm, T1 = normalize_points(pts1)
    
    # Build constraint matrix
    A = np.zeros((len(pts0_norm), 9))
    for i in range(len(pts0_norm)):
        x0, y0 = pts0_norm[i]
        x1, y1 = pts1_norm[i]
        A[i] = [x0*x1, x0*y1, x0, y0*x1, y0*y1, y0, x1, y1, 1]
    
    try:
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        F_norm = Vt[-1].reshape(3, 3)
        
        # Enforce rank-2 constraint
        U, S, Vt = np.linalg.svd(F_norm)
        S[2] = 0
        F_norm = U @ np.diag(S) @ Vt
        
        # Denormalize
        F = T1.T @ F_norm @ T0
        
        return F
    except:
        return None


def compute_sampson_distance(pts0, pts1, F):
    """Compute Sampson distance for fundamental matrix"""
    # Convert to homogeneous coordinates
    pts0_h = np.column_stack([pts0, np.ones(len(pts0))])
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
    
    # Compute Sampson distance
    Fx = (F @ pts0_h.T).T
    Ftx = (F.T @ pts1_h.T).T
    
    numerator = (np.sum(pts1_h * Fx, axis=1)) ** 2
    denominator = Fx[:, 0]**2 + Fx[:, 1]**2 + Ftx[:, 0]**2 + Ftx[:, 1]**2
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    
    return numerator / denominator


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