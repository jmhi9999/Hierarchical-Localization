import argparse
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


def kornia_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Custom Kornia RANSAC implementation for geometric verification."""
    if not KORNIA_AVAILABLE:
        logger.error("Kornia is not available. Please install kornia: pip install kornia")
        raise ImportError("kornia is required for kornia_ransac option")
    
    logger.info("Performing geometric verification using Kornia RANSAC...")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load pairs
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    db = COLMAPDatabase.connect(database_path)
    
    # Process pairs in batches for better GPU utilization
    batch_size = 32 if device.type == 'cuda' else 1  # Larger batches for GPU
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        
        # Prepare batch data
        batch_pts0 = []
        batch_pts1 = []
        batch_matches = []
        batch_image_ids = []
        valid_pairs = []
        
        for name0, name1 in batch_pairs:
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
                                     (db.pair_id_from_image_ids(image_id0, image_id1),)).fetchone()
            
            if kpts0 is None or kpts1 is None or matches_data is None:
                continue
                
            # Convert to numpy arrays
            kpts0 = np.frombuffer(kpts0[0], dtype=np.float32).reshape(-1, 6)[:, :2]
            kpts1 = np.frombuffer(kpts1[0], dtype=np.float32).reshape(-1, 6)[:, :2]
            matches = np.frombuffer(matches_data[0], dtype=np.uint32).reshape(-1, 2)
            
            if len(matches) < 8:
                db.add_two_view_geometry(image_id0, image_id1, matches)
                continue
            
            # Extract matched keypoints
            matched_kpts0 = kpts0[matches[:, 0]]
            matched_kpts1 = kpts1[matches[:, 1]]
            
            batch_pts0.append(matched_kpts0)
            batch_pts1.append(matched_kpts1)
            batch_matches.append(matches)
            batch_image_ids.append((image_id0, image_id1))
            valid_pairs.append((name0, name1))
        
        if not batch_pts0:
            continue
        
        try:
            # Pad sequences to same length for batch processing
            max_len = max(len(pts) for pts in batch_pts0)
            
            # Create padded tensors
            pts0_tensor = torch.zeros(len(batch_pts0), max_len, 2, device=device)
            pts1_tensor = torch.zeros(len(batch_pts0), max_len, 2, device=device)
            
            for j, (pts0, pts1) in enumerate(zip(batch_pts0, batch_pts1)):
                pts0_tensor[j, :len(pts0)] = torch.from_numpy(pts0).float()
                pts1_tensor[j, :len(pts1)] = torch.from_numpy(pts1).float()
            
            # Use Kornia's RANSAC for fundamental matrix estimation (batch processing)
            F, inliers = kornia.geometry.epipolar.find_fundamental(
                pts0_tensor, pts1_tensor,
                method=kornia.geometry.epipolar.SolverType.SEVEN_POINT,
                ransac_reproj_threshold=1.0,
                ransac_max_iter=5000,
                ransac_confidence=0.99
            )
            
            # Process results for each pair in batch
            for j, ((name0, name1), (image_id0, image_id1), matches) in enumerate(zip(valid_pairs, batch_image_ids, batch_matches)):
                if F is None or inliers is None:
                    logger.warning(f"Kornia RANSAC failed to find fundamental matrix for pair {name0}-{name1}")
                    db.add_two_view_geometry(image_id0, image_id1, matches)
                    continue
                
                # Extract inlier matches for this pair
                pair_inliers = inliers[j].cpu().numpy().astype(bool)
                # Only consider inliers up to the actual number of matches
                actual_len = len(matches)
                pair_inliers = pair_inliers[:actual_len]
                inlier_matches = matches[pair_inliers]
                
                # Add inlier matches to database
                db.add_two_view_geometry(image_id0, image_id1, inlier_matches)
                
        except Exception as e:
            logger.warning(f"Kornia RANSAC batch failed: {e}")
            # Fall back to individual processing for this batch
            for (name0, name1), (image_id0, image_id1), matches in zip(valid_pairs, batch_image_ids, batch_matches):
                db.add_two_view_geometry(image_id0, image_id1, matches)
    
    db.commit()
    db.close()


def magsac_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """Custom MAGSAC implementation for geometric verification."""
    if not MAGSAC_AVAILABLE:
        logger.error("PyMAGSAC is not available. Please install: pip install pymagsac")
        raise ImportError("pymagsac is required for magsac option")
    
    logger.info("Performing geometric verification using MAGSAC...")
    
    # Load pairs
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    
    db = COLMAPDatabase.connect(database_path)
    
    # Process each pair with MAGSAC
    for name0, name1 in tqdm(pairs, desc="MAGSAC verification"):
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
                                 (db.pair_id_from_image_ids(image_id0, image_id1),)).fetchone()
        
        if kpts0 is None or kpts1 is None or matches_data is None:
            continue
            
        # Convert to numpy arrays
        kpts0 = np.frombuffer(kpts0[0], dtype=np.float32).reshape(-1, 6)[:, :2]  # x, y coords only
        kpts1 = np.frombuffer(kpts1[0], dtype=np.float32).reshape(-1, 6)[:, :2]
        matches = np.frombuffer(matches_data[0], dtype=np.uint32).reshape(-1, 2)
        
        if len(matches) < 8:  # Need at least 8 points for fundamental matrix
            db.add_two_view_geometry(image_id0, image_id1, matches)
            continue
        
        # Extract matched keypoints
        matched_kpts0 = kpts0[matches[:, 0]]
        matched_kpts1 = kpts1[matches[:, 1]]
        
        try:
            # Use pymagsac for fundamental matrix estimation
            F, inliers = pymagsac.findFundamentalMatrix(
                matched_kpts0, matched_kpts1,
                threshold=1.0,
                conf=0.99,
                maxIters=10000,
                # MAGSAC specific parameters
                sampler=0,  # 0: PROSAC, 1: P-NAPSAC, 2: NG_RANSAC
                scorer=1,   # 0: RANSAC, 1: MSAC, 2: MLESAC, 3: MAGSAC
                neighborhood_size=20
            )
            
            # Check if fundamental matrix was found
            if F is None or inliers is None:
                logger.warning(f"MAGSAC failed to find fundamental matrix for pair {name0}-{name1}")
                db.add_two_view_geometry(image_id0, image_id1, matches)
                continue
            
            # Extract inlier matches
            inlier_matches = matches[inliers.astype(bool)]
            
            # Add inlier matches to database
            db.add_two_view_geometry(image_id0, image_id1, inlier_matches)
            
        except Exception as e:
            logger.warning(f"MAGSAC failed for pair {name0}-{name1}: {e}")
            # Fall back to all matches if MAGSAC fails
            db.add_two_view_geometry(image_id0, image_id1, matches)
    
    db.commit()
    db.close()


def loransac_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    """LORANSAC implementation using COLMAP's built-in support."""
    logger.info("Performing geometric verification using LORANSAC...")
    
    # COLMAP's LORANSAC implementation
    # Note: COLMAP may not have built-in LORANSAC, so we use a more robust RANSAC configuration
    # that approximates LORANSAC behavior with higher confidence and more iterations
    ransac_options = dict(
        two_view_geometry=dict(
            compute_relative_pose=True,
            ransac=dict(
                max_num_trials=15000,  # More iterations for LORANSAC-like behavior
                min_inlier_ratio=0.05,  # Lower threshold for more robust estimation
                confidence=0.9999,      # Higher confidence like LORANSAC
                max_error=1.5,          # Tighter error threshold
                min_num_trials=200,     # Minimum trials for robust estimation
            )
        )
    )
    
    with OutputCapture(verbose):
        pycolmap.verify_matches(
            database_path,
            pairs_path,
            options=ransac_options,
        )


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