import argparse
import collections.abc as collections
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import logger
from .utils.base_model import dynamic_load
from .utils.io import get_keypoints, list_h5_names
from .utils.parsers import names_to_pair, parse_image_lists


class ImprovedMatcher:
    """Improved feature matcher using LightGlue with inlier mask extraction"""
    
    def __init__(self, conf: Dict):
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load LightGlue matcher
        try:
            from .matchers import lightglue
            self.matcher = lightglue.LightGlue(conf["matcher"]).eval().to(self.device)
            logger.info("Successfully loaded LightGlue matcher")
        except ImportError:
            # Fallback to basic matcher if LightGlue not available
            logger.warning("LightGlue not available, using basic matcher")
            try:
                from .matchers import nearest_neighbor
                self.matcher = nearest_neighbor.NearestNeighbor(conf["matcher"])
            except ImportError:
                logger.error("No compatible matcher found")
                raise
        
    @torch.no_grad()
    def match_pair(self, data0: Dict, data1: Dict) -> Dict:
        """Match features between two images using LightGlue"""
        
        # Prepare data for LightGlue
        data = {
            "image0": data0["image"].to(self.device, non_blocking=True),
            "keypoints0": data0["keypoints"].to(self.device, non_blocking=True),
            "descriptors0": data0["descriptors"].to(self.device, non_blocking=True),
            "image1": data1["image"].to(self.device, non_blocking=True), 
            "keypoints1": data1["keypoints"].to(self.device, non_blocking=True),
            "descriptors1": data1["descriptors"].to(self.device, non_blocking=True),
        }
        
        # Run LightGlue matching
        pred = self.matcher(data)
        
        # Extract matches and confidence scores
        try:
            matches0 = pred["matches0"][0].cpu().numpy()
            matching_scores0 = pred["matching_scores0"][0].cpu().numpy()
        except (KeyError, IndexError) as e:
            logger.warning(f"Unexpected prediction format: {e}")
            # Fallback: create empty matches
            matches0 = np.array([]).reshape(0, 2)
            matching_scores0 = np.array([])
        
        # Extract inlier mask if available (this is LightGlue's geometric verification)
        inlier_mask = None
        try:
            if "inlier_mask" in pred:
                inlier_mask = pred["inlier_mask"][0].cpu().numpy()
            elif "confidence" in pred:
                # Use confidence as a proxy for inliers
                confidence_threshold = self.conf.get("confidence_threshold", 0.8)
                inlier_mask = matching_scores0 > confidence_threshold
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not extract inlier mask: {e}")
            inlier_mask = None
            
        return {
            "matches0": matches0,
            "matching_scores0": matching_scores0,
            "inlier_mask": inlier_mask,
        }


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    matches: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    
    logger.info(f"Extracting and matching local features with configuration:\n{conf}")
    
    # Load image pairs
    if isinstance(pairs, (Path, str)):
        # Parse pairs file directly
        pairs_name = []
        with open(pairs, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == "#":
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    pairs_name.append((parts[0], parts[1]))
    elif isinstance(pairs, collections.Iterable):
        pairs_name = list(pairs)
    else:
        raise ValueError(f"Unknown format for pairs argument: {pairs}")
        
    if matches is None:
        matches = Path(str(features).replace("feats", "matches") + ".h5")
    matches.parent.mkdir(exist_ok=True, parents=True)
    
    # Skip existing matches if not overwriting
    skip_pairs = set()
    if matches.exists() and not overwrite:
        with h5py.File(str(matches), "r", libver="latest") as fd:
            skip_pairs = set([k for k in fd.keys()])
    pairs_name = [p for p in pairs_name if names_to_pair(*p) not in skip_pairs]
    
    if len(pairs_name) == 0:
        logger.info("Skipping the matching.")
        return matches
        
    # Initialize matcher
    matcher = ImprovedMatcher(conf)
    
    # Load features
    feature_file = h5py.File(str(features), "r", libver="latest")
    
    # Process pairs
    for name0, name1 in tqdm(pairs_name):
        pair_name = names_to_pair(name0, name1)
        
        # Load features for both images
        try:
            # Load keypoints and descriptors
            kpts0 = feature_file[name0]["keypoints"].__array__()
            desc0 = feature_file[name0]["descriptors"].__array__()
            kpts1 = feature_file[name1]["keypoints"].__array__()
            desc1 = feature_file[name1]["descriptors"].__array__()
            
            # Add confidence scores (set to 1.0 for SuperPoint features)
            if kpts0.shape[1] == 2:  # Only x, y coordinates
                confidence0 = np.ones((kpts0.shape[0], 1))
                kpts0 = np.hstack([kpts0, confidence0])
            if kpts1.shape[1] == 2:  # Only x, y coordinates
                confidence1 = np.ones((kpts1.shape[0], 1))
                kpts1 = np.hstack([kpts1, confidence1])
            
            data0 = {
                "keypoints": torch.from_numpy(kpts0),
                "descriptors": torch.from_numpy(desc0),
                "image": torch.zeros((1, 1, 100, 100)),  # Dummy image for LightGlue
            }
            data1 = {
                "keypoints": torch.from_numpy(kpts1),
                "descriptors": torch.from_numpy(desc1),
                "image": torch.zeros((1, 1, 100, 100)),  # Dummy image for LightGlue
            }
        except KeyError as e:
            logger.warning(f"Could not find features for pair {pair_name}: {e}")
            continue
            
        # Match features
        try:
            pred = matcher.match_pair(data0, data1)
            
            # Save matches
            with h5py.File(str(matches), "a", libver="latest") as fd:
                if pair_name in fd:
                    del fd[pair_name]
                grp = fd.create_group(pair_name)
                
                grp.create_dataset("matches0", data=pred["matches0"])
                grp.create_dataset("matching_scores0", data=pred["matching_scores0"])
                
                if pred["inlier_mask"] is not None:
                    grp.create_dataset("inlier_mask", data=pred["inlier_mask"])
                    
        except Exception as e:
            logger.warning(f"Failed to match pair {pair_name}: {e}")
            continue
                
    feature_file.close()
    logger.info("Finished matching features.")
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)  
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--conf", type=str, default="superpoint+lightglue")
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()
    
    # Configuration for SuperPoint + LightGlue
    confs = {
        "superpoint+lightglue": {
            "matcher": {
                "features": "superpoint",
                "depth_confidence": 0.95,
                "width_confidence": 0.99,
            },
            "confidence_threshold": 0.8,
        }
    }
    
    main(confs[args.conf], args.pairs, args.features, args.matches, args.overwrite)