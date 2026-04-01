import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]


def _get_camera_name(env, camera_name=None):
    """Get the appropriate camera name for the environment."""
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError(f"Unknown robot type: {env.robot_uid}")
    return camera_name


def get_env_masks_from_obs(env, obs, camera_name=None):
    """
    Extract robot and target object segmentation masks from ManiSkill2 environment observation.
    This function is used to test the mask quality from segmentation model.
    This function uses the environment's built-in segmentation to reliably extract:
    - Robot arm mask: pixels belonging to the robot links
    - Object mask: pixels belonging to target manipulation objects
    
    Args:
        env: ManiSkill2 environment instance
        obs: Observation dictionary from env.step() or env.reset()
        camera_name: Camera name to use. If None, auto-detected based on robot type.
        
    Returns:
        tuple: (robot_mask, object_mask)
            - robot_mask: np.ndarray (H, W), uint8, 1=robot, 0=other
            - object_mask: np.ndarray (H, W), uint8, 1=object, 0=other
            Returns (None, None) if segmentation is not available.
    """
    camera_name = _get_camera_name(env, camera_name)
    
    # Check if segmentation is available in observations
    if "image" not in obs:
        logger.warning("No 'image' key in observation, cannot extract env masks")
        return None, None
    
    if camera_name not in obs["image"]:
        logger.warning(f"Camera '{camera_name}' not found in observation")
        return None, None
    
    cam_obs = obs["image"][camera_name]
    if "Segmentation" not in cam_obs:
        logger.warning(f"No 'Segmentation' in camera '{camera_name}' observation. "
                      "Make sure camera_cfgs={'add_segmentation': True} is set.")
        return None, None
    
    # Get segmentation data
    # Segmentation shape: (H, W, 4)
    # [..., 0]: mesh-level segmentation
    # [..., 1]: actor-level segmentation (what we need)
    # [..., 2:]: unused (zeros)
    seg = cam_obs["Segmentation"]
    actor_seg = seg[..., 1]  # Actor-level segmentation
    
    # === Get Robot Link IDs ===
    robot_mask = None
    try:
        # Access the unwrapped environment to get robot links
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Try to get robot_link_ids directly (may be cached by wrapper)
        if hasattr(unwrapped_env, 'robot_link_ids'):
            robot_link_ids = unwrapped_env.robot_link_ids
        elif hasattr(unwrapped_env, 'agent') and hasattr(unwrapped_env.agent, 'robot'):
            # Get robot links from agent
            robot_links = unwrapped_env.agent.robot.get_links()
            robot_link_ids = np.array([link.id for link in robot_links], dtype=np.int32)
        else:
            logger.warning("Cannot access robot links from environment")
            robot_link_ids = np.array([], dtype=np.int32)
        
        # Generate robot mask
        if len(robot_link_ids) > 0:
            robot_mask = np.isin(actor_seg, robot_link_ids).astype(np.uint8)
            logger.debug(f"Robot mask: {np.count_nonzero(robot_mask)}/{robot_mask.size} pixels, "
                        f"robot_link_ids={robot_link_ids[:5]}..." if len(robot_link_ids) > 5 else f"robot_link_ids={robot_link_ids}")
        else:
            robot_mask = np.zeros_like(actor_seg, dtype=np.uint8)
            logger.warning("No robot link IDs found, robot_mask is empty")
            
    except Exception as e:
        logger.error(f"Failed to get robot mask: {e}")
        robot_mask = np.zeros_like(actor_seg, dtype=np.uint8)
    
    # === Get Target Object Actor IDs ===
    object_mask = None
    try:
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Get all actors and filter to target objects
        # Exclude: ground, goal_site, empty names, arena, and always-overlay objects
        exclude_names = ['ground', 'goal_site', '', 'arena']
        
        # Check for rgb_always_overlay_objects attribute
        if hasattr(unwrapped_env, 'rgb_always_overlay_objects'):
            exclude_names.extend(unwrapped_env.rgb_always_overlay_objects)
        
        if hasattr(unwrapped_env, 'get_actors'):
            actors = unwrapped_env.get_actors()
            target_object_actor_ids = np.array(
                [actor.id for actor in actors if actor.name not in exclude_names],
                dtype=np.int32
            )
        else:
            logger.warning("Environment does not have get_actors() method")
            target_object_actor_ids = np.array([], dtype=np.int32)
        
        # Also get link IDs of articulated objects (like drawers, cabinets)
        other_link_ids = []
        if hasattr(unwrapped_env, '_scene') and hasattr(unwrapped_env._scene, 'get_all_articulations'):
            for art_obj in unwrapped_env._scene.get_all_articulations():
                # Skip the robot itself
                if hasattr(unwrapped_env, 'agent') and art_obj is unwrapped_env.agent.robot:
                    continue
                # Skip always-overlay objects
                if hasattr(unwrapped_env, 'rgb_always_overlay_objects') and art_obj.name in unwrapped_env.rgb_always_overlay_objects:
                    continue
                for link in art_obj.get_links():
                    other_link_ids.append(link.id)
        other_link_ids = np.array(other_link_ids, dtype=np.int32)
        
        # Combine target objects and articulated object links
        all_object_ids = np.concatenate([target_object_actor_ids, other_link_ids]) if len(other_link_ids) > 0 else target_object_actor_ids
        
        # Generate object mask
        if len(all_object_ids) > 0:
            object_mask = np.isin(actor_seg, all_object_ids).astype(np.uint8)
            logger.debug(f"Object mask: {np.count_nonzero(object_mask)}/{object_mask.size} pixels, "
                        f"num_objects={len(target_object_actor_ids)}, num_articulated_links={len(other_link_ids)}")
        else:
            object_mask = np.zeros_like(actor_seg, dtype=np.uint8)
            logger.warning("No target object IDs found, object_mask is empty")
            
    except Exception as e:
        logger.error(f"Failed to get object mask: {e}")
        object_mask = np.zeros_like(actor_seg, dtype=np.uint8)
    
    return robot_mask, object_mask
