"""
COLMAP mapper options optimized for different RANSAC methods.
When MAGSAC produces fewer inliers, we need more lenient reconstruction parameters.
"""

def get_robust_mapper_options(ransac_option: str = "ransac"):
    """Get optimized COLMAP mapper options based on RANSAC method."""
    
    base_options = {
        "min_num_matches": 15,
        "ignore_watermarks": False,
        "multiple_models": True,
        "max_num_models": 50,
        "max_model_overlap": 20,
        "min_model_size": 10,
    }
    
    if ransac_option == "magsac":
        # MAGSAC is more conservative, so relax reconstruction parameters
        return {
            **base_options,
            "min_num_matches": 10,  # Lower threshold (default 15)
            "min_model_size": 8,    # Allow smaller models (default 10)
            "max_model_overlap": 30, # Allow more overlap (default 20)
            "abs_pose_min_num_inliers": 15,  # Lower inlier requirement (default 30)
            "abs_pose_min_inlier_ratio": 0.15,  # Lower ratio (default 0.25)
            "filter_max_reproj_error": 6.0,  # More lenient reprojection error (default 4.0)
            "filter_min_tri_angle": 1.0,  # Lower triangulation angle (default 1.5)
            "init_min_num_inliers": 80,  # Lower initialization requirement (default 100)
            "init_min_inlier_ratio": 0.2,  # Lower initialization ratio (default 0.25) 
        }
    elif ransac_option == "loransac":
        # LORANSAC is more accurate, can use stricter parameters
        return {
            **base_options,
            "min_num_matches": 20,  # Higher threshold for accuracy
            "abs_pose_min_num_inliers": 35,  # Higher inlier requirement
            "abs_pose_min_inlier_ratio": 0.3,  # Higher ratio
            "filter_max_reproj_error": 3.0,  # Stricter reprojection error
            "filter_min_tri_angle": 2.0,  # Higher triangulation angle
        }
    elif ransac_option == "kornia_ransac":
        # GPU-accelerated, can handle more matches
        return {
            **base_options,
            "min_num_matches": 12,
            "max_num_models": 100,  # Can handle more models efficiently
        }
    else:  # vanilla ransac
        return base_options