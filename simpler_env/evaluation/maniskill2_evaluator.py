"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import logging

import numpy as np
from transforms3d.euler import quat2euler

# Configure logger
logger = logging.getLogger(__name__)

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
    get_env_masks_from_obs,
)
from simpler_env.utils.visualization import write_interval_video, write_video

# Import image simplification module
try:
    from simpler_env.utils.image_simplification import create_image_simplifier
    IMAGE_SIMPLIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Image simplification not available: {e}")
    IMAGE_SIMPLIFICATION_AVAILABLE = False


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
    image_simplifier=None,  # Added: image simplification module
    enable_simplification=False,  # Added: flag to enable/disable
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    # __import__('ipdb').set_trace()
    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    
    # === Image Simplification: Reset for new episode ===
    # reset() automatically marks for re-detection on next frame
    if enable_simplification and image_simplifier is not None:
        image_simplifier.reset()
    
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"
    # action_ensemble = model.action_ensemble_temp  if hasattr(model, "action_ensemble") else "none"

    # Step the environment
    task_descriptions = []
    # Exit loop if: model predicts termination, episode is truncated (max steps), or task succeeds (and is final subtask for multi-subtask envs)
    while not (predicted_terminated or truncated or (done and is_final_subtask)):
        # === Image Simplification: Dual Mask Strategy ===
        # 1. Get environment-provided masks (reliable: robot arm + manipulation objects)
        # 2. Get GroundingSAM2 detected masks (semantic detection)
        # 3. Combine both for complete task-relevant mask
        task_mask = None
        robot_mask = None
        object_mask = None
        
        if enable_simplification and image_simplifier is not None:
            try:
                # Step 1: Get environment segmentation masks (100% reliable)
                try:
                    robot_mask, object_mask = get_env_masks_from_obs(env, obs, camera_name=obs_camera_name)
                    if robot_mask is not None:
                        logger.debug(f"[ENV MASK] Robot mask: {np.count_nonzero(robot_mask)}/{robot_mask.size} pixels")
                    if object_mask is not None:
                        logger.debug(f"[ENV MASK] Object mask: {np.count_nonzero(object_mask)}/{object_mask.size} pixels")
                except Exception as e:
                    logger.warning(f"[ENV MASK] Failed to get environment masks: {e}")
                    robot_mask, object_mask = None, None
                
                # Step 2 & 3: Call ImageSimplifier with environment masks
                # ImageSimplifier will combine env masks with GroundingSAM2 detected masks
                logger.info(f"[DEBUG] About to call simplify_frame, frame shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
                result = image_simplifier.simplify_frame(
                    image, 
                    task_description,
                    robot_mask=robot_mask,
                    object_mask=object_mask
                )
                if isinstance(result, tuple):
                    image, task_mask = result
                    logger.info(f"[DEBUG] simplify_frame returned: image shape={image.shape}, "
                              f"mask shape={task_mask.shape}, mask dtype={task_mask.dtype}, "
                              f"mask range=[{task_mask.min()}, {task_mask.max()}], "
                              f"mask non-zero pixels={np.count_nonzero(task_mask)}/{task_mask.size}")
                else:
                    image = result
                    logger.info(f"[DEBUG] simplify_frame returned single value (no mask)")
                logger.info(f"[DEBUG] simplify_frame returned successfully")
            except Exception as e:
                logger.error(f"[DEBUG] Image simplification failed: {e}")
                print(f"Warning: Image simplification failed: {e}, using original image")
        
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(
            image, 
            task_description, 
            task_mask=task_mask,
            eef_pos=obs["agent"]["eef_pos"]
        )
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            ),
        )

        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
            # === Image Simplification: Mark new task for re-detection ===
            if enable_simplification and image_simplifier is not None:
                image_simplifier.mark_new_task()
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=obs_camera_name
        )
        images.append(image)
        task_descriptions.append(task_description)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name

    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)
    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    
    # === Initialize Image Simplifier (once for all episodes) ===
    image_simplifier = None
    
    # Check environment variables first, then args
    env_var = os.environ.get('ENABLE_IMAGE_SIMPLIFICATION', '0')
    logger.info(f"[DEBUG] ENABLE_IMAGE_SIMPLIFICATION env var: '{env_var}'")
    enable_simplification = env_var == '1' or getattr(args, 'enable_image_simplification', False)
    logger.info(f"[DEBUG] enable_simplification final value: {enable_simplification}")
    
    logger.info(f"[DEBUG] IMAGE_SIMPLIFICATION_AVAILABLE: {IMAGE_SIMPLIFICATION_AVAILABLE}")
    
    if enable_simplification and IMAGE_SIMPLIFICATION_AVAILABLE:
        try:
            # Create visualization directory with env_name to avoid conflicts between different tasks
            # Each task/environment will have its own subdirectory
            simplification_save_dir = os.path.join(args.logging_dir, "image_simplification", args.env_name)
            logger.info(f"[DEBUG] Creating ImageSimplifier with save_dir: {simplification_save_dir}")
            print(f"Initializing image simplification module...")
            print(f"Visualization will be saved to: {simplification_save_dir}")
            
            # Get blur_sigma from environment or args
            blur_sigma = float(os.environ.get('IMAGE_SIMPLIFICATION_BLUR_SIGMA', 
                              str(getattr(args, 'blur_sigma', 10.0))))
            
            # Get detection_interval from environment or args
            detection_interval = int(os.environ.get('IMAGE_SIMPLIFICATION_DETECTION_INTERVAL',
                                    str(getattr(args, 'detection_interval', 1))))
          
            image_simplifier = create_image_simplifier(
                device="cuda" if args.policy_model else "cuda",
                save_dir=None,
                blur_sigma=blur_sigma,
                detection_interval=detection_interval
            )
            print(f"Image simplification initialized successfully! (blur_sigma={blur_sigma})")
        except Exception as e:
            print(f"Warning: Failed to initialize image simplification: {e}")
            print("Continuing without image simplification...")
            enable_simplification = False

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                    image_simplifier=image_simplifier,  # Added
                    enable_simplification=enable_simplification,  # Added
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(
                        args.obj_episode_range[0], args.obj_episode_range[1]
                    ):
                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                obj_episode_id=obj_episode_id, **kwargs
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr
