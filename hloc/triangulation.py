import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pycolmap
import torch
import kornia as K
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
    logger.info("Performing geometric verification of the matches...")
    with OutputCapture(verbose):
        pycolmap.verify_matches(
            database_path,
            pairs_path,
            options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)),
        )


def kornia_ransac_verification_no_reference(
    image_ids: Dict[str, int],
    database_path: Path,
    features_path: Path,
    pairs_path: Path,
    matches_path: Path,
    max_error: float = 4.0,
    ransac_confidence: float = 0.99,
    max_iter: int = 10000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Geometric verification using Kornia GPU-based RANSAC without reference model."""
    logger.info(f"Performing Kornia GPU-based geometric verification (no reference) on {device}...")

    pairs = parse_retrieval(pairs_path)
    db = COLMAPDatabase.connect(database_path)

    inlier_ratios = []
    matched = set()
    
    for name0 in tqdm(pairs):
        id0 = image_ids[name0]
        kps0 = get_keypoints(features_path, name0)
        
        for name1 in pairs[name0]:
            id1 = image_ids[name1]
            kps1 = get_keypoints(features_path, name1)
            matches = get_matches(matches_path, name0, name1)[0]

            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue
            matched |= {(id0, id1), (id1, id0)}

            if matches.shape[0] == 0:
                db.add_two_view_geometry(id0, id1, matches)
                continue

            # Use Kornia RANSAC for fundamental matrix estimation
            if matches.shape[0] >= 8:  # Need at least 8 points for fundamental matrix
                try:
                    # Convert to torch tensors and add batch dimension
                    pts0 = torch.from_numpy(kps0[matches[:, 0]]).float().to(device).unsqueeze(0)
                    pts1 = torch.from_numpy(kps1[matches[:, 1]]).float().to(device).unsqueeze(0)
                    
                    # Estimate fundamental matrix using Kornia RANSAC
                    F, inliers = K.geometry.ransac.find_fundamental(
                        pts0, pts1,
                        threshold=max_error,
                        confidence=ransac_confidence,
                        max_iter=max_iter
                    )
                    
                    # Get inlier mask
                    inlier_mask = inliers.squeeze(0).cpu().numpy().astype(bool)
                    valid_matches = matches[inlier_mask]
                    
                except Exception as e:
                    logger.debug(f"Kornia RANSAC failed for {name0}-{name1}: {e}, using all matches")
                    valid_matches = matches
            else:
                # Not enough points for fundamental matrix, use all matches
                valid_matches = matches

            db.add_two_view_geometry(id0, id1, valid_matches)
            inlier_ratios.append(len(valid_matches) / len(matches) if len(matches) > 0 else 0)

    logger.info(
        "mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.",
        np.mean(inlier_ratios) * 100,
        np.median(inlier_ratios) * 100,
        np.min(inlier_ratios) * 100,
        np.max(inlier_ratios) * 100,
    )

    db.commit()
    db.close()


def kornia_ransac_geometric_verification(
    image_ids: Dict[str, int],
    reference: pycolmap.Reconstruction,
    database_path: Path,
    features_path: Path,
    pairs_path: Path,
    matches_path: Path,
    max_error: float = 4.0,
    ransac_confidence: float = 0.99,
    max_iter: int = 10000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Geometric verification using Kornia GPU-based RANSAC for fundamental matrix estimation."""
    logger.info(f"Performing Kornia GPU-based geometric verification on {device}...")

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

            # Use Kornia RANSAC for fundamental matrix estimation
            if matches.shape[0] >= 8:  # Need at least 8 points for fundamental matrix
                try:
                    # Convert to torch tensors
                    pts0 = torch.from_numpy(kps0[matches[:, 0]]).float().to(device)
                    pts1 = torch.from_numpy(kps1[matches[:, 1]]).float().to(device)
                    
                    # Add homogeneous coordinates
                    pts0_h = torch.cat([pts0, torch.ones(pts0.shape[0], 1, device=device)], dim=1)
                    pts1_h = torch.cat([pts1, torch.ones(pts1.shape[0], 1, device=device)], dim=1)
                    
                    # Estimate fundamental matrix using Kornia RANSAC
                    F, inliers = K.geometry.ransac.find_fundamental(
                        pts0_h.unsqueeze(0), pts1_h.unsqueeze(0),
                        threshold=max_error,
                        confidence=ransac_confidence,
                        max_iter=max_iter
                    )
                    
                    # Get inlier mask
                    inlier_mask = inliers.squeeze(0).cpu().numpy().astype(bool)
                    valid_matches = matches[inlier_mask]
                    
                    # Fallback to epipolar error computation if Kornia RANSAC fails
                    if valid_matches.shape[0] == 0:
                        raise RuntimeError("No inliers found")
                        
                except Exception as e:
                    logger.debug(f"Kornia RANSAC failed for {name0}-{name1}: {e}, falling back to epipolar error")
                    # Fallback to original epipolar error method
                    cam1_from_cam0 = image1.cam_from_world() * image0.cam_from_world().inverse()
                    errors0, errors1 = compute_epipolar_errors(
                        cam1_from_cam0, kps0[matches[:, 0]], kps1[matches[:, 1]]
                    )
                    valid_mask = np.logical_and(
                        errors0 <= cam0.cam_from_img_threshold(noise0 * max_error),
                        errors1 <= cam1.cam_from_img_threshold(noise1 * max_error),
                    )
                    valid_matches = matches[valid_mask]
            else:
                # Not enough points for fundamental matrix, use original method
                cam1_from_cam0 = image1.cam_from_world() * image0.cam_from_world().inverse()
                errors0, errors1 = compute_epipolar_errors(
                    cam1_from_cam0, kps0[matches[:, 0]], kps1[matches[:, 1]]
                )
                valid_mask = np.logical_and(
                    errors0 <= cam0.cam_from_img_threshold(noise0 * max_error),
                    errors1 <= cam1.cam_from_img_threshold(noise1 * max_error),
                )
                valid_matches = matches[valid_mask]

            db.add_two_view_geometry(id0, id1, valid_matches)
            inlier_ratios.append(len(valid_matches) / len(matches) if len(matches) > 0 else 0)

    logger.info(
        "mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.",
        np.mean(inlier_ratios) * 100,
        np.median(inlier_ratios) * 100,
        np.min(inlier_ratios) * 100,
        np.max(inlier_ratios) * 100,
    )

    db.commit()
    db.close()


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
    use_kornia_ransac: bool = False,
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
        elif use_kornia_ransac:
            kornia_ransac_geometric_verification(
                image_ids, reference, database, features, pairs, matches
            )
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
    parser.add_argument("--use_kornia_ransac", action="store_true", 
                       help="Use Kornia GPU-based RANSAC for geometric verification")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, mapper_options=mapper_options)
