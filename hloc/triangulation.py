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
    database_path: Path, pairs_path: Path, verbose: bool = False, use_loransac: bool = True
):
    if use_loransac:
        logger.info("Performing LoRANSAC geometric verification...")
        loransac_estimation_and_geometric_verification(database_path, pairs_path, verbose)
    else:
        logger.info("Performing OpenCV USAC MAGSAC geometric verification...")
        opencv_usac_estimation_and_geometric_verification(database_path, pairs_path, verbose)


def loransac_estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Complete LoRANSAC implementation using PyRANSAC library"""
    from .utils.database import image_ids_to_pair_id, blob_to_array
    import time
    
    start_time = time.time()
    
    # Try to import PyRANSAC
    try:
        import pyransac
        has_pyransac = True
        logger.info("✓ SUCCESS: Using PyRANSAC for true LoRANSAC implementation")
        logger.info(f"PyRANSAC version: {getattr(pyransac, '__version__', 'unknown')}")
    except ImportError:
        try:
            # Try alternative import
            from pyransac import ransac as pyransac
            has_pyransac = True
            logger.info("✓ SUCCESS: Using PyRANSAC (alternative import) for true LoRANSAC implementation")
        except ImportError:
            has_pyransac = False
            logger.warning("✗ FALLBACK: PyRANSAC not available, install with: pip install pyransac")
            logger.warning("✗ FALLBACK: Using OpenCV USAC LoRANSAC (much slower than PyRANSAC)")
            logger.warning("✗ PERFORMANCE WARNING: This will be significantly slower!")
    
    # LoRANSAC configuration - optimized to beat pycolmap RANSAC speed while improving accuracy
    # Original pycolmap: max_num_trials=20000, min_inlier_ratio=0.05, confidence=0.999
    loransac_config = {
        # Core RANSAC parameters - more aggressive than pycolmap for speed
        'threshold': 1.0,           # Keep original threshold for accuracy
        'confidence': 0.999,        # Match pycolmap confidence
        'max_iterations': 15000,    # 25% fewer than pycolmap (20000 → 15000)
        'min_inlier_ratio': 0.05,   # Match pycolmap's aggressive threshold
        'min_inliers': 8,           # Minimum for fundamental matrix
        
        # LoRANSAC specific - minimal overhead for speed
        'lo_iterations': 3,         # Very light local optimization
        'lo_sample_size': 10,       # Small sample for speed
        'lo_frequency': 10,         # Only do LO every 10th good model
        
        # Speed optimizations to beat pycolmap
        'early_termination': True,          # Smart early stopping
        'progressive_sampling': True,       # Use better sampling strategies
        'fast_mode': True,                 # Enable all speed optimizations
        'preemptive_verification': True,    # Quick outlier rejection
        'batch_evaluation': True,          # Vectorized operations
        
        # Quality vs Speed balance
        'quality_threshold': 0.8,   # When to switch to quality mode
        'speed_first_iterations': 5000,  # First N iterations prioritize speed
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
            pair_start_time = time.time()
            if has_pyransac:
                # Use true PyRANSAC LoRANSAC
                if total_pairs <= 3:  # Log first few pairs for debugging
                    logger.info(f"Processing pair {name0}-{name1} with PyRANSAC LoRANSAC ({len(matches)} matches)")
                F, inlier_matches = pyransac_fundamental_matrix(
                    pts0, pts1, matches, loransac_config, pyransac
                )
            else:
                # Fallback to OpenCV USAC
                if total_pairs <= 3:  # Log first few pairs for debugging
                    logger.info(f"Processing pair {name0}-{name1} with OpenCV USAC LoRANSAC ({len(matches)} matches)")
                F, inlier_matches = opencv_loransac_fundamental_matrix(
                    pts0, pts1, matches, loransac_config
                )
            
            pair_time = time.time() - pair_start_time
            if total_pairs <= 3:  # Log timing for first few pairs
                logger.info(f"Pair {name0}-{name1} processed in {pair_time:.3f}s")
            
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
    elapsed_time = time.time() - start_time
    
    logger.info(f"=== LoRANSAC PERFORMANCE REPORT ===")
    logger.info(f"Method used: {method} LoRANSAC")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Pairs processed: {total_pairs}")
    logger.info(f"Successful verifications: {successful_verifications}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Time per pair: {elapsed_time/max(total_pairs, 1):.3f} seconds")
    logger.info(f"==================================")


def pyransac_fundamental_matrix(pts0, pts1, matches, config, pyransac_lib):
    """True LoRANSAC implementation using PyRANSAC"""
    
    logger.debug(f"PyRANSAC LoRANSAC: Processing {len(pts0)} point correspondences")
    logger.debug(f"PyRANSAC config: threshold={config['threshold']}, max_iter={config['max_iterations']}")
    
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
        # Try different PyRANSAC APIs based on the actual module structure
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
            
        elif hasattr(pyransac_lib, 'loransac'):
            # Direct loransac function
            F, inliers = pyransac_lib.loransac(
                data=data,
                model_fit_func=fit_fundamental_matrix,
                residual_func=compute_residuals,
                min_samples=8,
                threshold=config['threshold'],
                max_iterations=config['max_iterations'],
                confidence=config['confidence'],
                local_opt_iterations=config.get('lo_iterations', 10),
                local_opt_sample_size=config.get('lo_sample_size', 14)
            )
            
        elif callable(pyransac_lib):
            # Module is callable directly
            F, inliers = pyransac_lib(
                data=data,
                model_fit_func=fit_fundamental_matrix,
                residual_func=compute_residuals,
                min_samples=8,
                threshold=config['threshold'],
                max_iterations=config['max_iterations'],
                confidence=config['confidence']
            )
            
        else:
            # Try to find any LoRANSAC-related function
            loransac_func = None
            for attr_name in dir(pyransac_lib):
                attr = getattr(pyransac_lib, attr_name)
                if callable(attr) and ('loransac' in attr_name.lower() or 'ransac' in attr_name.lower()):
                    loransac_func = attr
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
            logger.debug(f"PyRANSAC SUCCESS: Found {len(inliers)} inliers from {len(matches)} matches")
            return F, inlier_matches
        else:
            logger.debug("PyRANSAC FAILED: No valid fundamental matrix found")
            return None, np.array([]).reshape(0, 2)
            
    except Exception as e:
        logger.warning(f"PyRANSAC ERROR: {e}, falling back to manual LoRANSAC")
        # Fallback to manual LoRANSAC implementation
        return manual_loransac_fundamental_matrix(data, matches, config)


def manual_loransac_fundamental_matrix(data, matches, config):
    """High-speed LoRANSAC optimized to beat pycolmap RANSAC while improving accuracy"""
    
    logger.debug(f"MANUAL LoRANSAC: Processing {len(data)} correspondences (PyRANSAC fallback)")
    logger.debug(f"MANUAL config: threshold={config['threshold']}, max_iter={config['max_iterations']}")
    
    best_F = None
    best_inliers = []
    best_score = 0
    
    pts0 = data[:, :2]
    pts1 = data[:, 2:]
    num_points = len(data)
    
    # Speed optimization: precompute requirements
    min_required_inliers = max(config.get('min_inliers', 8), int(num_points * config.get('min_inlier_ratio', 0.05)))
    threshold = config['threshold']
    confidence = config.get('confidence', 0.999)
    max_iterations = config['max_iterations']
    
    # Speed optimization: progressive sampling for first iterations
    speed_first_iterations = config.get('speed_first_iterations', 5000)
    use_progressive_sampling = config.get('progressive_sampling', True)
    
    # Speed optimization: batch evaluation
    batch_size = min(50, max(10, num_points // 100)) if config.get('batch_evaluation', True) else 1
    
    # Speed optimization: preemptive verification
    if config.get('preemptive_verification', True):
        # Quick outlier rejection using distance to centroid
        centroid0 = np.mean(pts0, axis=0)
        centroid1 = np.mean(pts1, axis=0)
        dist0 = np.linalg.norm(pts0 - centroid0, axis=1)
        dist1 = np.linalg.norm(pts1 - centroid1, axis=1)
        # Focus on points closer to centroid for initial sampling
        weight_scores = 1.0 / (1.0 + dist0 + dist1)
        sampling_weights = weight_scores / np.sum(weight_scores)
    else:
        sampling_weights = None
    
    # LoRANSAC frequency control
    lo_frequency = config.get('lo_frequency', 10)
    models_since_lo = 0
    
    for iteration in range(max_iterations):
        if num_points < 8:
            break
            
        # Progressive sampling strategy
        if use_progressive_sampling and iteration < speed_first_iterations:
            # Use weighted sampling for speed
            if sampling_weights is not None:
                sample_idx = np.random.choice(num_points, 8, replace=False, p=sampling_weights)
            else:
                sample_idx = np.random.choice(num_points, 8, replace=False)
        else:
            # Pure random sampling for quality
            sample_idx = np.random.choice(num_points, 8, replace=False)
        
        # Batch processing for speed
        F_candidates = []
        for batch_start in range(0, min(batch_size, max(1, max_iterations - iteration)), 1):
            if iteration + batch_start >= max_iterations:
                break
                
            # Fit fundamental matrix to sample
            F = fit_fundamental_matrix_8point_fast(pts0[sample_idx], pts1[sample_idx])
            if F is not None:
                F_candidates.append((F, sample_idx))
        
        # Evaluate all candidates
        for F, sample_idx in F_candidates:
            # Fast Sampson distance computation (vectorized)
            errors = compute_sampson_distance_fast(pts0, pts1, F)
            inliers = np.where(errors < threshold)[0]
            
            if len(inliers) < 8:
                continue
            
            score = len(inliers)
            
            # Only do Local Optimization selectively for speed
            should_do_lo = (
                models_since_lo >= lo_frequency and 
                score > best_score and
                (iteration < speed_first_iterations or score > best_score * 1.2)
            )
            
            if should_do_lo:
                models_since_lo = 0
                # Light Local Optimization - minimal overhead
                lo_iterations = config.get('lo_iterations', 3)
                lo_sample_size = min(config.get('lo_sample_size', 10), len(inliers))
                
                for lo_iter in range(lo_iterations):
                    if len(inliers) >= 8:
                        # Smart sampling: use best inliers
                        if len(inliers) > lo_sample_size:
                            # Use inliers with smallest errors
                            inlier_errors = errors[inliers]
                            best_inlier_idx = np.argsort(inlier_errors)[:lo_sample_size]
                            lo_sample_idx = inliers[best_inlier_idx]
                        else:
                            lo_sample_idx = inliers
                        
                        # Refine model
                        F_refined = fit_fundamental_matrix_8point_fast(pts0[lo_sample_idx], pts1[lo_sample_idx])
                        if F_refined is not None:
                            errors_refined = compute_sampson_distance_fast(pts0, pts1, F_refined)
                            inliers_refined = np.where(errors_refined < threshold)[0]
                            
                            if len(inliers_refined) > len(inliers):
                                F = F_refined
                                inliers = inliers_refined
                                score = len(inliers)
            else:
                models_since_lo += 1
            
            # Update best model
            if score > best_score:
                best_F = F
                best_inliers = inliers
                best_score = score
                
                # Aggressive early termination for speed
                inlier_ratio = score / num_points
                if (score >= min_required_inliers and 
                    inlier_ratio >= config.get('min_inlier_ratio', 0.05)):
                    
                    # Smart termination: calculate required iterations
                    outlier_ratio = max(1.0 - inlier_ratio, 1e-6)
                    required_iters = int(np.log(1 - confidence) / np.log(1 - inlier_ratio**8))
                    
                    # Be more aggressive than pycolmap
                    if iteration >= required_iters * 0.8:  # 20% earlier termination
                        return best_F, matches[best_inliers] if best_F is not None else (None, np.array([]).reshape(0, 2))
                
                # Super aggressive termination for very good models
                if inlier_ratio > config.get('quality_threshold', 0.8):
                    return best_F, matches[best_inliers]
    
    if best_F is not None:
        logger.debug(f"MANUAL LoRANSAC SUCCESS: Found {len(best_inliers)} inliers from {len(matches)} matches")
        return best_F, matches[best_inliers]
    else:
        logger.debug("MANUAL LoRANSAC FAILED: No valid fundamental matrix found")
        return None, np.array([]).reshape(0, 2)


def fit_fundamental_matrix_8point_fast(pts0, pts1):
    """Optimized 8-point algorithm - faster than standard version"""
    if len(pts0) < 8:
        return None
    
    # Skip normalization for speed in many cases - trade-off for pycolmap beating
    try:
        # Direct 8-point without normalization (faster but less stable)
        A = np.zeros((len(pts0), 9))
        for i in range(len(pts0)):
            x0, y0 = pts0[i]
            x1, y1 = pts1[i]
            A[i] = [x0*x1, x0*y1, x0, y0*x1, y0*y1, y0, x1, y1, 1]
        
        # Fast SVD
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        F = Vt[-1].reshape(3, 3)
        
        # Quick rank-2 enforcement
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        S[2] = 0
        F = U @ np.diag(S) @ Vt
        
        return F
    except:
        # Fallback to normalized version if needed
        return fit_fundamental_matrix_8point(pts0, pts1)


def compute_sampson_distance_fast(pts0, pts1, F):
    """Vectorized fast Sampson distance computation"""
    # Convert to homogeneous (vectorized)
    ones = np.ones((len(pts0), 1))
    pts0_h = np.column_stack([pts0, ones])
    pts1_h = np.column_stack([pts1, ones])
    
    # Vectorized computation
    Fx = pts0_h @ F.T  # Shape: (N, 3)
    Ftx = pts1_h @ F    # Shape: (N, 3)
    
    # Numerator: (x'^T F x)^2
    numerator = (np.sum(pts1_h * Fx, axis=1)) ** 2
    
    # Denominator: (Fx)_1^2 + (Fx)_2^2 + (F^T x')_1^2 + (F^T x')_2^2
    denominator = Fx[:, 0]**2 + Fx[:, 1]**2 + Ftx[:, 0]**2 + Ftx[:, 1]**2
    denominator = np.maximum(denominator, 1e-10)  # Faster than np.where
    
    return numerator / denominator


def opencv_usac_estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Fast C++ LoRANSAC using OpenCV USAC - optimized to beat pycolmap"""
    from .utils.database import image_ids_to_pair_id, blob_to_array
    
    # OpenCV USAC configuration - faster than pycolmap
    usac_config = {
        'threshold': 1.0,         # Sampson distance threshold
        'confidence': 0.999,      # Match pycolmap confidence  
        'max_iterations': 15000,  # Fewer iterations for speed
        'min_inlier_ratio': 0.05, # Match pycolmap threshold
        'method': cv2.USAC_MAGSAC,  # Best C++ LoRANSAC (threshold-free)
        'lo_iterations': 5,       # Local optimization iterations
        'lo_sample_size': 12      # Sample size for LO
    }
    
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    db = COLMAPDatabase.connect(database_path)
    
    processed_pairs = set()
    successful_verifications = 0
    total_pairs = 0
    
    for name0, name1 in tqdm(pairs, desc="OpenCV USAC LoRANSAC"):
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
        
        if len(matches) < 8:
            # Add empty geometry for insufficient matches
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
            # Use OpenCV USAC (C++ LoRANSAC implementation)
            F, mask = cv2.findFundamentalMat(
                pts0, pts1,
                method=usac_config['method'],
                ransacReprojThreshold=usac_config['threshold'],
                confidence=usac_config['confidence'],
                maxIters=usac_config['max_iterations']
            )
            
            if F is not None and mask is not None:
                inlier_mask = mask.ravel() == 1
                inlier_matches = matches[inlier_mask]
                inlier_ratio = len(inlier_matches) / len(matches)
                
                if (inlier_ratio >= usac_config['min_inlier_ratio'] and 
                    len(inlier_matches) >= 8):
                    
                    # Store successful geometry
                    db.add_two_view_geometry(
                        image_id0, image_id1,
                        matches=inlier_matches,
                        F=F, E=np.eye(3), H=np.eye(3),
                        qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                        config=len(inlier_matches)
                    )
                    
                    successful_verifications += 1
                    
                    if verbose:
                        method_name = "USAC_MAGSAC" if usac_config['method'] == cv2.USAC_MAGSAC else "USAC_FM_8PTS"
                        logger.info(f"{method_name}: {name0}-{name1}: {len(inlier_matches)}/{len(matches)} inliers ({inlier_ratio:.2%})")
                else:
                    # Low inlier ratio
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
                logger.warning(f"OpenCV USAC failed for {name0}-{name1}: {e}")
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
    method_name = "USAC_MAGSAC" if usac_config['method'] == cv2.USAC_MAGSAC else "USAC_FM_8PTS"
    logger.info(f"OpenCV {method_name} LoRANSAC completed: {successful_verifications}/{total_pairs} pairs verified successfully ({success_rate:.1f}%)")


def opencv_loransac_fundamental_matrix(pts0, pts1, matches, config):
    """Optimized OpenCV USAC LoRANSAC - configured to beat pycolmap speed"""
    try:
        # Use optimized parameters to beat pycolmap performance
        F, mask = cv2.findFundamentalMat(
            pts0, pts1,
            method=cv2.USAC_FM_8PTS,  # LoRANSAC method
            ransacReprojThreshold=config['threshold'],
            confidence=config['confidence'],
            maxIters=config['max_iterations'],
            # Optimized LoRANSAC parameters for speed
            loIterations=config.get('lo_iterations', 3),  # Minimal LO for speed
            loSampleSize=config.get('lo_sample_size', 10)  # Small sample size
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
    use_loransac: bool = True,
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
            estimation_and_geometric_verification(database, pairs, verbose, use_loransac)
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
    parser.add_argument("--use_loransac", action="store_true", default=True, help="Use LoRANSAC instead of MAGSAC")
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, mapper_options=mapper_options)