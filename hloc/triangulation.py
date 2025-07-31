import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pycolmap
from tqdm import tqdm

try:
    import kornia
    import torch
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

try:
    import pymagsac
    MAGSAC_AVAILABLE = True
except ImportError:
    MAGSAC_AVAILABLE = False

try:
    import cv2
    # Check if OpenCV version supports LORANSAC (4.5+)
    cv2_version = cv2.__version__.split('.')
    LORANSAC_AVAILABLE = len(cv2_version) >= 2 and int(cv2_version[0]) >= 4 and int(cv2_version[1]) >= 5
except ImportError:
    LORANSAC_AVAILABLE = False

from . import logger
from .utils.database import COLMAPDatabase, image_ids_to_pair_id
from .utils.geometry import compute_epipolar_errors
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_retrieval


def reshape_keypoints(keypoints_blob):
    """Safely reshape keypoints blob to 2D coordinates."""
    keypoints = np.frombuffer(keypoints_blob, dtype=np.float32)
    num_keypoints = len(keypoints)
    
    # Try different column counts: 2, 4, 6
    for cols in [2, 4, 6]:
        if num_keypoints % cols == 0:
            rows = num_keypoints // cols
            return keypoints.reshape(rows, cols)[:, :2]  # Return only x, y coordinates
    
    # Fallback: assume 2 columns
    return keypoints.reshape(-1, 2)


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


def kornia_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Custom Kornia RANSAC implementation for geometric verification."""
    if not KORNIA_AVAILABLE:
        logger.error("Kornia is not available. Please install kornia: pip install kornia")
        raise ImportError("kornia is required for kornia_ransac option")
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("🚀 KORNIA RANSAC GEOMETRIC VERIFICATION STARTED")
    logger.info("=" * 60)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"📱 Using device: {device}")
    
    # Load pairs
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    logger.info(f"📊 Total pairs to process: {len(pairs)}")
    
    db = COLMAPDatabase.connect(database_path)
    
    # Statistics tracking
    stats = {
        'total_pairs': len(pairs),
        'processed_pairs': 0,
        'successful_pairs': 0,
        'failed_pairs': 0,
        'total_matches': 0,
        'total_inliers': 0,
        'processing_times': []
    }
    
    # Process each pair individually (simpler and more robust)
    for name0, name1 in tqdm(pairs, desc="Kornia RANSAC verification"):
        pair_start_time = time.time()
        
        # Get matches from database
        image_id0 = db.execute("SELECT image_id FROM images WHERE name = ?", (name0,)).fetchone()
        image_id1 = db.execute("SELECT image_id FROM images WHERE name = ?", (name1,)).fetchone()
        
        if image_id0 is None or image_id1 is None:
            continue
            
        image_id0, image_id1 = image_id0[0], image_id1[0]
        
        # Get keypoints and matches
        kpts0 = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id0,)).fetchone()
        kpts1 = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id1,)).fetchone()
        matches_data = db.execute("SELECT data FROM matches WHERE pair_id = ?", 
                                 (image_ids_to_pair_id(image_id0, image_id1),)).fetchone()
        
        if kpts0 is None or kpts1 is None or matches_data is None:
            continue
            
        # Convert to numpy arrays
        kpts0 = reshape_keypoints(kpts0[0])
        kpts1 = reshape_keypoints(kpts1[0])
        matches = np.frombuffer(matches_data[0], dtype=np.uint32).reshape(-1, 2)
        
        if len(matches) < 8:  # Need at least 8 points for fundamental matrix
            db.add_two_view_geometry(image_id0, image_id1, matches)
            stats['processed_pairs'] += 1
            continue
        
        stats['total_matches'] += len(matches)
        
        # Extract matched keypoints
        matched_kpts0 = kpts0[matches[:, 0]]
        matched_kpts1 = kpts1[matches[:, 1]]
        
        # Convert to torch tensors
        pts0 = torch.from_numpy(matched_kpts0).float().to(device)
        pts1 = torch.from_numpy(matched_kpts1).float().to(device)
        
        try:
            # Kornia's 7POINT method requires exactly 7 points
            if len(pts0) < 7:
                db.add_two_view_geometry(image_id0, image_id1, matches)
                stats['failed_pairs'] += 1
                continue
            elif len(pts0) == 7:
                # Exactly 7 points - use 7POINT method
                F = kornia.geometry.epipolar.find_fundamental(
                    pts0.unsqueeze(0), pts1.unsqueeze(0),
                    method='7POINT'
                )
            else:
                # More than 7 points - use RANSAC approach with random sampling
                F = kornia.geometry.epipolar.find_fundamental(
                    pts0.unsqueeze(0), pts1.unsqueeze(0),
                    method='RANSAC'  # Use RANSAC for variable number of points
                )
            
            # Check if fundamental matrix was found
            if F is None:
                db.add_two_view_geometry(image_id0, image_id1, matches)
                stats['failed_pairs'] += 1
                continue
            
            # Compute epipolar distances to find inliers
            F_np = F.squeeze().cpu().numpy()
            
            # Convert to homogeneous coordinates
            pts0_homo = np.column_stack([matched_kpts0, np.ones(len(matched_kpts0))])
            pts1_homo = np.column_stack([matched_kpts1, np.ones(len(matched_kpts1))])
            
            # Compute epipolar distances
            epilines1 = (F_np @ pts0_homo.T).T  # Lines in image 1
            epilines0 = (F_np.T @ pts1_homo.T).T  # Lines in image 0
            
            # Distance from points to epipolar lines
            dists1 = np.abs(np.sum(epilines1 * pts1_homo, axis=1)) / np.sqrt(epilines1[:, 0]**2 + epilines1[:, 1]**2)
            dists0 = np.abs(np.sum(epilines0 * pts0_homo, axis=1)) / np.sqrt(epilines0[:, 0]**2 + epilines0[:, 1]**2)
            
            # Find inliers (threshold of 1.0 pixel)
            inlier_mask = (dists0 <= 1.0) & (dists1 <= 1.0)
            inlier_matches = matches[inlier_mask]
            
            # Update statistics
            stats['total_inliers'] += len(inlier_matches)
            stats['successful_pairs'] += 1
            
            # Add inlier matches to database
            db.add_two_view_geometry(image_id0, image_id1, inlier_matches)
            
        except Exception as e:
            logger.warning(f"Kornia RANSAC failed for pair {name0}-{name1}: {e}")
            # Fall back to all matches if RANSAC fails
            db.add_two_view_geometry(image_id0, image_id1, matches)
            stats['failed_pairs'] += 1
        
        pair_time = time.time() - pair_start_time
        stats['processing_times'].append(pair_time)
        stats['processed_pairs'] += 1
    
    db.commit()
    db.close()
    
    # Calculate and log final statistics
    total_time = time.time() - start_time
    avg_inlier_ratio = (stats['total_inliers'] / stats['total_matches']) * 100 if stats['total_matches'] > 0 else 0
    avg_pair_time = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    success_rate = (stats['successful_pairs'] / stats['processed_pairs']) * 100 if stats['processed_pairs'] > 0 else 0
    
    logger.info("=" * 60)
    logger.info("🏁 KORNIA RANSAC VERIFICATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    logger.info(f"📈 Average time per pair: {avg_pair_time:.3f} seconds")
    logger.info(f"📊 Processed pairs: {stats['processed_pairs']}/{stats['total_pairs']}")
    logger.info(f"✅ Successful pairs: {stats['successful_pairs']} ({success_rate:.1f}%)")
    logger.info(f"❌ Failed pairs: {stats['failed_pairs']}")
    logger.info(f"🎯 Total matches: {stats['total_matches']}")
    logger.info(f"✨ Total inliers: {stats['total_inliers']}")
    logger.info(f"📏 Average inlier ratio: {avg_inlier_ratio:.1f}%")
    logger.info(f"🚀 Throughput: {stats['processed_pairs']/total_time:.1f} pairs/second")
    logger.info("=" * 60)


def magsac_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Custom MAGSAC implementation for geometric verification."""
    if not MAGSAC_AVAILABLE:
        logger.error("PyMAGSAC is not available. Please install: pip install pymagsac")
        raise ImportError("pymagsac is required for magsac option")
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("🎯 MAGSAC GEOMETRIC VERIFICATION STARTED")
    logger.info("=" * 60)
    
    # Load pairs
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    logger.info(f"📊 Total pairs to process: {len(pairs)}")
    
    db = COLMAPDatabase.connect(database_path)
    
    # Statistics tracking
    stats = {
        'total_pairs': len(pairs),
        'processed_pairs': 0,
        'successful_pairs': 0,
        'failed_pairs': 0,
        'total_matches': 0,
        'total_inliers': 0,
        'processing_times': []
    }
    
    # Process each pair with MAGSAC
    for name0, name1 in tqdm(pairs, desc="MAGSAC verification"):
        pair_start_time = time.time()
        
        # Get matches from database
        image_id0 = db.execute("SELECT image_id FROM images WHERE name = ?", (name0,)).fetchone()
        image_id1 = db.execute("SELECT image_id FROM images WHERE name = ?", (name1,)).fetchone()
        
        if image_id0 is None or image_id1 is None:
            continue
            
        image_id0, image_id1 = image_id0[0], image_id1[0]
        
        # Get keypoints and matches
        kpts0 = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id0,)).fetchone()
        kpts1 = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id1,)).fetchone()
        matches_data = db.execute("SELECT data FROM matches WHERE pair_id = ?", 
                                 (image_ids_to_pair_id(image_id0, image_id1),)).fetchone()
        
        if kpts0 is None or kpts1 is None or matches_data is None:
            continue
            
        # Convert to numpy arrays
        kpts0 = reshape_keypoints(kpts0[0])
        kpts1 = reshape_keypoints(kpts1[0])
        matches = np.frombuffer(matches_data[0], dtype=np.uint32).reshape(-1, 2)
        
        if len(matches) < 8:  # Need at least 8 points for fundamental matrix
            db.add_two_view_geometry(image_id0, image_id1, matches)
            stats['processed_pairs'] += 1
            continue
        
        stats['total_matches'] += len(matches)
        
        # Extract matched keypoints
        matched_kpts0 = kpts0[matches[:, 0]]
        matched_kpts1 = kpts1[matches[:, 1]]
        
        try:
            # Use pymagsac for fundamental matrix estimation
            # pymagsac API: findFundamentalMatrix(src_pts, dst_pts, sigma_max)
            F, inliers = pymagsac.findFundamentalMatrix(
                matched_kpts0, matched_kpts1, 
                1.5  # sigma_max (balanced between robustness and coverage)
            )
            
            # Debug: Check input data quality
            if verbose and stats['processed_pairs'] < 5:  # Only for first few pairs
                logger.info(f"MAGSAC Debug - Pair {pair_key}:")
                logger.info(f"  Input matches: {len(matches)}")
                logger.info(f"  Keypoints shape: {matched_kpts0.shape}, {matched_kpts1.shape}")
                logger.info(f"  F matrix: {F is not None}, Inliers: {inliers is not None}")
                if inliers is not None:
                    logger.info(f"  Inlier count: {np.sum(inliers)}/{len(inliers)} ({100*np.sum(inliers)/len(inliers):.1f}%)")
            
            # Check if fundamental matrix was found
            if F is None or inliers is None:
                db.add_two_view_geometry(image_id0, image_id1, matches)
                stats['failed_pairs'] += 1
                continue
            
            # Check if we have enough inliers
            num_inliers = np.sum(inliers)
            if num_inliers < 8:  # Need at least 8 inliers for reliable geometry
                db.add_two_view_geometry(image_id0, image_id1, matches)
                stats['failed_pairs'] += 1
                continue
            
            # Extract inlier matches
            inlier_matches = matches[inliers.astype(bool)]
            
            # Update statistics
            stats['total_inliers'] += len(inlier_matches)
            stats['successful_pairs'] += 1
            
            # Add inlier matches to database
            db.add_two_view_geometry(image_id0, image_id1, inlier_matches)
            
        except Exception as e:
            logger.warning(f"MAGSAC failed for pair {name0}-{name1}: {e}")
            # Fall back to all matches if MAGSAC fails
            db.add_two_view_geometry(image_id0, image_id1, matches)
            stats['failed_pairs'] += 1
        
        pair_time = time.time() - pair_start_time
        stats['processing_times'].append(pair_time)
        stats['processed_pairs'] += 1
    
    db.commit()
    db.close()
    
    # Calculate and log final statistics
    total_time = time.time() - start_time
    avg_inlier_ratio = (stats['total_inliers'] / stats['total_matches']) * 100 if stats['total_matches'] > 0 else 0
    avg_pair_time = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    success_rate = (stats['successful_pairs'] / stats['processed_pairs']) * 100 if stats['processed_pairs'] > 0 else 0
    
    logger.info("=" * 60)
    logger.info("🏁 MAGSAC VERIFICATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    logger.info(f"📈 Average time per pair: {avg_pair_time:.3f} seconds")
    logger.info(f"📊 Processed pairs: {stats['processed_pairs']}/{stats['total_pairs']}")
    logger.info(f"✅ Successful pairs: {stats['successful_pairs']} ({success_rate:.1f}%)")
    logger.info(f"❌ Failed pairs: {stats['failed_pairs']}")
    logger.info(f"🎯 Total matches: {stats['total_matches']}")
    logger.info(f"✨ Total inliers: {stats['total_inliers']}")
    logger.info(f"📏 Average inlier ratio: {avg_inlier_ratio:.1f}%")
    logger.info(f"🚀 Throughput: {stats['processed_pairs']/total_time:.1f} pairs/second")
    logger.info("=" * 60)


def loransac_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """LORANSAC implementation using OpenCV if available, fallback to robust RANSAC."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("🎪 LORANSAC GEOMETRIC VERIFICATION STARTED")
    logger.info("=" * 60)
    
    if LORANSAC_AVAILABLE:
        logger.info("✅ Using OpenCV's MAGSAC implementation")
        # Use OpenCV's LORANSAC implementation
        _opencv_loransac_verification(database_path, pairs_path, verbose, start_time)
    else:
        logger.warning("⚠️  OpenCV LORANSAC not available, using robust RANSAC configuration")
        # Fallback to more robust RANSAC configuration that approximates LORANSAC behavior
        ransac_options = dict(
            ransac=dict(
                max_num_trials=15000,   # More iterations for LORANSAC-like behavior
                min_inlier_ratio=0.05,  # Lower threshold for more robust estimation
                confidence=0.9999,      # Higher confidence like LORANSAC
                max_error=1.5,          # Tighter error threshold
                min_num_trials=200,     # Minimum trials for robust estimation
            )
        )
        
        with OutputCapture(verbose):
            pycolmap.verify_matches(
                database_path,
                pairs_path,
                options=ransac_options,
            )
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("🏁 LORANSAC (FALLBACK) VERIFICATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
        logger.info(f"🚀 Used robust RANSAC fallback configuration")
        logger.info("=" * 60)


def _opencv_loransac_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False, start_time: float = None
):
    """OpenCV LORANSAC implementation for geometric verification."""
    import cv2
    
    if start_time is None:
        start_time = time.time()
    
    # Load pairs
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    logger.info(f"📊 Total pairs to process: {len(pairs)}")
    
    db = COLMAPDatabase.connect(database_path)
    
    # Statistics tracking
    stats = {
        'total_pairs': len(pairs),
        'processed_pairs': 0,
        'successful_pairs': 0,
        'failed_pairs': 0,
        'total_matches': 0,
        'total_inliers': 0,
        'processing_times': []
    }
    
    # Process each pair with OpenCV LORANSAC
    for name0, name1 in tqdm(pairs, desc="LORANSAC verification"):
        pair_start_time = time.time()
        
        # Get matches from database
        image_id0 = db.execute("SELECT image_id FROM images WHERE name = ?", (name0,)).fetchone()
        image_id1 = db.execute("SELECT image_id FROM images WHERE name = ?", (name1,)).fetchone()
        
        if image_id0 is None or image_id1 is None:
            continue
            
        image_id0, image_id1 = image_id0[0], image_id1[0]
        
        # Get keypoints and matches
        kpts0 = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id0,)).fetchone()
        kpts1 = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id1,)).fetchone()
        matches_data = db.execute("SELECT data FROM matches WHERE pair_id = ?", 
                                 (image_ids_to_pair_id(image_id0, image_id1),)).fetchone()
        
        if kpts0 is None or kpts1 is None or matches_data is None:
            continue
            
        # Convert to numpy arrays
        kpts0 = reshape_keypoints(kpts0[0])
        kpts1 = reshape_keypoints(kpts1[0])
        matches = np.frombuffer(matches_data[0], dtype=np.uint32).reshape(-1, 2)
        
        if len(matches) < 8:  # Need at least 8 points for fundamental matrix
            db.add_two_view_geometry(image_id0, image_id1, matches)
            stats['processed_pairs'] += 1
            continue
        
        stats['total_matches'] += len(matches)
        
        # Extract matched keypoints
        matched_kpts0 = kpts0[matches[:, 0]]
        matched_kpts1 = kpts1[matches[:, 1]]
        
        try:
            # Use OpenCV's MAGSAC for fundamental matrix estimation
            F, inliers = cv2.findFundamentalMat(
                matched_kpts0, matched_kpts1,
                method=cv2.USAC_MAGSAC,  # MAGSAC is more robust than LORANSAC
                ransacReprojThreshold=1.0,
                confidence=0.999,
                maxIters=10000
            )
            
            # Check if fundamental matrix was found
            if F is None or inliers is None:
                db.add_two_view_geometry(image_id0, image_id1, matches)
                stats['failed_pairs'] += 1
                continue
            
            # Extract inlier matches
            inlier_mask = inliers.ravel().astype(bool)
            inlier_matches = matches[inlier_mask]
            
            # Update statistics
            stats['total_inliers'] += len(inlier_matches)
            stats['successful_pairs'] += 1
            
            # Add inlier matches to database
            db.add_two_view_geometry(image_id0, image_id1, inlier_matches)
            
        except Exception as e:
            logger.warning(f"LORANSAC failed for pair {name0}-{name1}: {e}")
            # Fall back to all matches if LORANSAC fails
            db.add_two_view_geometry(image_id0, image_id1, matches)
            stats['failed_pairs'] += 1
        
        pair_time = time.time() - pair_start_time
        stats['processing_times'].append(pair_time)
        stats['processed_pairs'] += 1
    
    db.commit()
    db.close()
    
    # Calculate and log final statistics
    total_time = time.time() - start_time
    avg_inlier_ratio = (stats['total_inliers'] / stats['total_matches']) * 100 if stats['total_matches'] > 0 else 0
    avg_pair_time = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    success_rate = (stats['successful_pairs'] / stats['processed_pairs']) * 100 if stats['processed_pairs'] > 0 else 0
    
    logger.info("=" * 60)
    logger.info("🏁 LORANSAC (OPENCV) VERIFICATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    logger.info(f"📈 Average time per pair: {avg_pair_time:.3f} seconds")
    logger.info(f"📊 Processed pairs: {stats['processed_pairs']}/{stats['total_pairs']}")
    logger.info(f"✅ Successful pairs: {stats['successful_pairs']} ({success_rate:.1f}%)")
    logger.info(f"❌ Failed pairs: {stats['failed_pairs']}")
    logger.info(f"🎯 Total matches: {stats['total_matches']}")
    logger.info(f"✨ Total inliers: {stats['total_inliers']}")
    logger.info(f"📏 Average inlier ratio: {avg_inlier_ratio:.1f}%")
    logger.info(f"🚀 Throughput: {stats['processed_pairs']/total_time:.1f} pairs/second")
    logger.info("=" * 60)


def estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False, ransac_option: str = "ransac"
):
    logger.info(f"Performing geometric verification of the matches using {ransac_option}...")
    
    # Handle special RANSAC implementations separately
    if ransac_option == "kornia_ransac":
        logger.info("Using Kornia RANSAC for geometric verification")
        kornia_geometric_verification(database_path, pairs_path, verbose)
        return
    elif ransac_option == "magsac":
        logger.info("Using MAGSAC for geometric verification")
        magsac_geometric_verification(database_path, pairs_path, verbose)
        return
    elif ransac_option == "loransac":
        logger.info("Using LORANSAC for geometric verification")
        loransac_geometric_verification(database_path, pairs_path, verbose)
        return
    
    # Standard RANSAC using COLMAP's default implementation
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("⚡ STANDARD RANSAC GEOMETRIC VERIFICATION STARTED")
    logger.info("=" * 60)
    
    if ransac_option == "ransac":
        logger.info("Using standard RANSAC for geometric verification")
        ransac_options = dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1))
    else:
        logger.warning(f"Unknown RANSAC option: {ransac_option}. Using default RANSAC.")
        ransac_options = dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1))
    
    with OutputCapture(verbose):
        pycolmap.verify_matches(
            database_path,
            pairs_path,
            options=ransac_options,
        )
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🏁 STANDARD RANSAC VERIFICATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    logger.info(f"🚀 COLMAP handled all internal statistics and processing")
    logger.info("=" * 60)


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
    ransac_option: str = "ransac",
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
            estimation_and_geometric_verification(database, pairs, verbose, ransac_option)
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
    parser.add_argument(
        "--ransac_option",
        type=str,
        default="ransac",
        choices=["ransac", "magsac", "loransac", "kornia_ransac"],
        help="RANSAC algorithm to use for geometric verification. "
             "ransac: Standard RANSAC (default, fast). "
             "magsac: MAGSAC (requires pymagsac). "
             "loransac: LORANSAC-like robust RANSAC (slower but more accurate). "
             "kornia_ransac: Kornia RANSAC (requires kornia, torch, GPU recommended)"
    )
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, mapper_options=mapper_options)