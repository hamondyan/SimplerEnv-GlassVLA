"""
Real-time Mask Detection Module for SimplerEnv-OpenVLA-GS
Integrates GroundingDINO and SAM2 for semantic object detection and mask generation.

This module:
- Detects objects using GroundingDINO based on task descriptions
- Generates precise masks using SAM2 Image Predictor
- Tracks objects across frames using SAM2 Video Predictor
- Returns masks for model's internal adaptive blur processing

Note: Image blurring is now handled by the model's adaptive_blur module, not here.
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import logging
import tempfile

# Add Grounded-SAM-2 to path using absolute paths
grounded_sam2_root = "/home/futuremm/workplace/Grounded-SAM-2/"
grounding_dino_path = os.path.join(grounded_sam2_root, "grounding_dino")

for path in [grounded_sam2_root, grounding_dino_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import from Grounded-SAM-2
from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert

logger = logging.getLogger(__name__)


class TorchLoadPatch:
    """Context manager to patch torch.load for PyTorch 2.6 compatibility."""
    def __enter__(self):
        self.original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return self.original_load(*args, **kwargs)
        torch.load = patched_load
        return self
    
    def __exit__(self, *args):
        torch.load = self.original_load


def parse_task_description(task_description: str, object_list: Optional[List[str]] = None) -> str:
    """
    Parse task description to GroundingDINO prompt.
    Based on video_simplification/utils/prompt_parser.py logic.
    
    Args:
        task_description: Original task (e.g., "open top drawer")
        object_list: List of valid objects. Uses default if None.
        
    Returns:
        Simplified detection prompt (e.g., "top drawer. cabinet. robotic arm.")
    """
    # Default object list based on SimplerEnv tasks
    if object_list is None:
        object_list = [
            "7up can", 
            "apple", 
            "blue plastic bottle", 
            "bottom drawer", 
            "bowl", 
            "cabinet", 
            "coke can", 
            "middle drawer", 
            "orange", 
            "orange can", 
            "pepsi can", 
            "red bull",
            "redbull", 
            "sponge", 
            "top drawer"
        ]
    
    # Fixed suffixes to append (base: always include robotic arm)
    fixed_suffixes = ["robotic arm"]
    
    # Convert to lowercase for case-insensitive matching
    task_lower = task_description.lower()
    
    # Extract matching objects
    found_objects = []
    
    # Sort object list by length (descending) to match longer phrases first
    # This prevents "can" from matching before "coke can"
    sorted_objects = sorted(object_list, key=len, reverse=True)
    
    # Track positions to avoid overlapping matches
    matched_positions = set()
    
    for obj in sorted_objects:
        obj_lower = obj.lower()
        start = 0
        
        while True:
            # Find next occurrence
            pos = task_lower.find(obj_lower, start)
            
            if pos == -1:
                break
            
            # Check if this position overlaps with already matched positions
            obj_positions = set(range(pos, pos + len(obj_lower)))
            
            if not obj_positions & matched_positions:
                # No overlap, this is a valid match
                found_objects.append(obj)
                matched_positions.update(obj_positions)
                logger.debug(f"Found object: '{obj}' at position {pos}")
            
            start = pos + 1
    
    # Remove duplicates while preserving order
    unique_objects = []
    seen = set()
    for obj in found_objects:
        obj_lower = obj.lower()
        if obj_lower not in seen:
            unique_objects.append(obj)
            seen.add(obj_lower)
    
    # Special rules based on keywords in task description
    has_drawer = "drawer" in task_lower
    if has_drawer:
        fixed_suffixes.append("cabinet")
        logger.debug("Detected 'drawer' in prompt, adding 'cabinet' to suffixes")
    if "redbull" in task_lower:
        fixed_suffixes.append("red bull")
        logger.debug("Detected 'redbull' in prompt, adding 'red bull' to suffixes")
    
    # Build final prompt
    result_parts = unique_objects + fixed_suffixes
    
    # Format with periods between items
    prompt = ". ".join(result_parts) + "."
    
    logger.debug(f"Task '{task_description}' -> Prompt '{prompt}'")
    return prompt




class ImageSimplifier:
    """
    Real-time mask detection using GroundingDINO + SAM2.
    
    Models are loaded once and reused across all episodes for efficiency.
    Provides mask detection and tracking, with blur handled by the model internally.
    """
    
    def __init__(
        self,
        grounding_dino_config: str,
        grounding_dino_checkpoint: str,
        sam2_checkpoint: str,
        sam2_config: str,  # NEW: SAM2 config path
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        blur_sigma: float = 30.0,  # Deprecated: no longer used, kept for backward compatibility
        save_dir: Optional[str] = None,
        detection_interval: int = 1,
    ):
        """
        Initialize the image simplifier with model paths.
        
        Args:
            grounding_dino_config: Path to GroundingDINO config
            grounding_dino_checkpoint: Path to GroundingDINO weights
            sam2_checkpoint: Path to SAM2 checkpoint
            sam2_config: Path to SAM2 config file (Hydra module path)
            device: Device to use ('cuda' or 'cpu')
            box_threshold: Confidence threshold for object detection
            text_threshold: Text matching threshold
            blur_sigma: Deprecated, no longer used (blur is done by model internally)
            save_dir: Optional directory to save debug images
            detection_interval: Frame interval for re-detection (1 = every frame)
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.save_dir = save_dir
        self.detection_interval = detection_interval
        
        # Initialize state variables
        self.inference_state = None
        self.total_frames = 0  # Global counter, never reset
        self.episode_frame_count = 0  # Frame counter within current episode
        self.current_prompt = None
        self.object_masks = {}
        self.force_reinit_next_frame = True
        self.frame_cache_limit = max(1, detection_interval - 1)  # Frame cache limit
        
        # SAM2 models: image predictor for detection, video predictor for tracking
        self.image_predictor = None
        self.video_predictor = None
        
        # Setup output directory
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            # Create log file for task descriptions and prompts
            self.log_file = os.path.join(self.save_dir, "prompts_log.txt")
            with open(self.log_file, 'w') as f:
                f.write("Frame\tTask Description\tGroundingDINO Prompt\n")
                f.write("="*80 + "\n")
        else:
            self.log_file = None
        
        # Load models once with PyTorch 2.6 compatibility
        logger.info("Loading GroundingDINO model...")
        with TorchLoadPatch():
            self.grounding_model = load_model(
                model_config_path=grounding_dino_config,
                model_checkpoint_path=grounding_dino_checkpoint,
                device=device
            )
        
        logger.info(f"Loading SAM2 models (config: {sam2_config})...")
        # SAM2 uses Hydra config system: config paths are relative to sam2 module
        # e.g., "configs/sam2.1/sam2.1_hiera_l.yaml"
        with TorchLoadPatch():
            # Load SAM2 base model once (shared by both predictors)
            sam2_model = build_sam2(sam2_config, sam2_checkpoint)
            
            # Create video predictor for tracking (may load model internally, but we use shared model for image predictor)
            self.video_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)
            
            # Create image predictor using shared model for initial detection
            self.image_predictor = SAM2ImagePredictor(sam2_model)
        
        logger.info("ImageSimplifier initialized successfully")
    
    def _cleanup_inference_state(self):
        """Helper method to clean up inference_state and free GPU memory."""
        if self.inference_state is None:
            return
        
        try:
            # Clear cached features
            if "cached_features" in self.inference_state:
                self.inference_state["cached_features"].clear()
            
            # Clear image tensors
            if "images" in self.inference_state:
                if isinstance(self.inference_state["images"], torch.Tensor):
                    self.inference_state["images"] = self.inference_state["images"].cpu()
                del self.inference_state["images"]
            
            # Clear tracking dictionaries
            for key in ["output_dict_per_obj", "temp_output_dict_per_obj", 
                       "point_inputs_per_obj", "mask_inputs_per_obj"]:
                if key in self.inference_state and isinstance(self.inference_state[key], dict):
                    self.inference_state[key].clear()
        except Exception as e:
            logger.warning(f"Error during inference_state cleanup: {e}")
        
        self.inference_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset(self):
        """Reset state for new episode with strict memory cleanup."""
        self._cleanup_inference_state()
        self.episode_frame_count = 0
        self.current_prompt = None
        self.object_masks = {}
        self.force_reinit_next_frame = True
        logger.info("ImageSimplifier reset - ready for new episode")
    
    def mark_new_task(self):
        """Mark that a new task/episode has started, requiring re-detection."""
        self.force_reinit_next_frame = True
        logger.debug("Marked for re-initialization on next frame")

    
    def _detect_objects(self, image_rgb: np.ndarray, prompt: str) -> Tuple[np.ndarray, List[str]]:
        """
        Step 1: Detect objects using GroundingDINO.
        
        Args:
            image_rgb: RGB image (H, W, 3)
            prompt: Text prompt for detection
            
        Returns:
            boxes: (N, 4) array in xyxy format
            labels: List of detected labels
        """
        # Convert to PIL and save to temp file (required by load_image)
        image_pil = Image.fromarray(image_rgb)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            image_pil.save(temp_path)
        
        try:
            _, processed_image = load_image(temp_path)
            
            # Detect objects with threshold filtering
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=processed_image,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
        finally:
            os.unlink(temp_path)
        
        if len(boxes) == 0:
            return np.array([]), []
        
        # Convert boxes from cxcywh to xyxy
        h, w = image_rgb.shape[:2]
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        logger.debug(f"Detected {len(boxes_xyxy)} objects: {labels}")
        return boxes_xyxy, labels
    
    def _generate_masks_from_boxes(self, image_rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Generate precise segmentation masks using SAM2 Image Predictor.
        
        Args:
            image_rgb: RGB image (H, W, 3)
            boxes: Bounding boxes (N, 4) in xyxy format
            
        Returns:
            masks: (N, H, W) binary masks, dtype=bool
        """
        if len(boxes) == 0:
            return np.array([])
        
        # Set image for SAM2 Image Predictor
        self.image_predictor.set_image(image_rgb)
        
        # Generate masks from boxes
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        
        # Normalize mask shape to (N, H, W)
        # SAM2 returns masks in various formats depending on multimask_output
        if masks.ndim == 4:  # (B, C, H, W)
            if masks.shape[1] == 1:
                masks = masks.squeeze(1)  # -> (B, H, W)
            else:
                masks = masks[:, 0, :, :]  # Take first mask per box
        elif masks.ndim == 3:
            if masks.shape[-1] <= 10:  # Probably (H, W, N)
                masks = masks.transpose(2, 0, 1)  # -> (N, H, W)
        elif masks.ndim == 2:  # Single mask (H, W)
            masks = masks[None]  # -> (1, H, W)
        
        logger.debug(f"Generated {len(masks)} masks with shape {masks.shape}")
        return masks
    
    def _init_video_tracking(self, image_rgb: np.ndarray, prompt: str):
        """
        Initialize SAM2 video tracking with first frame.
        
        Pipeline:
        1. Detect objects with GroundingDINO
        2. Generate precise masks with SAM2 Image Predictor
        3. Initialize video tracking with masks
        
        Called at episode start or when detection_interval is reached.
        """
        # Check if we need to reset due to frame cache limit
        if self.inference_state is not None and "images" in self.inference_state:
            if isinstance(self.inference_state["images"], torch.Tensor):
                cache_size = self.inference_state["images"].shape[0]
                if cache_size > self.frame_cache_limit:
                    logger.info(
                        f"[Memory Control] Resetting inference state after {cache_size} cached frames "
                        f"(limit: {self.frame_cache_limit}) to free GPU memory"
                    )
                    self._cleanup_inference_state()
        
        # Step 1: Detect objects using GroundingDINO
        boxes, labels = self._detect_objects(image_rgb, prompt)
        
        if len(boxes) == 0:
            logger.warning("No objects detected, returning empty mask")
            self.inference_state = None
            return
        
        logger.info(f"Detected {len(boxes)} objects: {labels}")
        
        # Step 2: Generate precise masks using SAM2 Image Predictor
        masks = self._generate_masks_from_boxes(image_rgb, boxes)
        
        if len(masks) == 0:
            logger.warning("Failed to generate masks, returning empty mask")
            self.inference_state = None
            return
        
        logger.info(f"Generated {len(masks)} precise masks")
        
        # Step 3: Initialize SAM2 video tracking with generated masks
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["video_height"] = image_rgb.shape[0]
        self.inference_state["video_width"] = image_rgb.shape[1]
        
        # Add first frame and reset state
        frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_rgb)
        self.video_predictor.reset_state(self.inference_state)
        
        # Add masks to video predictor
        for obj_id, (mask, label) in enumerate(zip(masks, labels), start=1):
            self.video_predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
            )
            logger.debug(f"Added object {obj_id}: {label}")
        
        # Run inference to get initial masks
        _, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
        )
        
        # Store masks
        self.object_masks = {}
        for i, obj_id in enumerate(obj_ids):
            mask = (video_res_masks[i] > 0.0).cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            self.object_masks[obj_id] = mask
        
        logger.info(f"Initialized video tracking with {len(self.object_masks)} objects")
    
    def _track_frame(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Track objects in current frame using SAM2 video predictor.
        
        Args:
            image_rgb: RGB image (H, W, 3)
            
        Returns:
            Combined mask for all tracked objects (H, W), uint8
        """
        if self.inference_state is None:
            # No tracking initialized, return empty mask
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        
        # Add new frame to SAM2 video predictor
        # Note: With offload_state_to_cpu=True, intermediate states are on CPU
        frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_rgb)
        
        # Propagate masks to current frame
        _, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
        )
        
        # Update and combine object masks
        self.object_masks = {}
        for i, obj_id in enumerate(obj_ids):
            mask = (video_res_masks[i] > 0.0).cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            self.object_masks[obj_id] = mask
        
        # Combine all masks into single binary mask
        if self.object_masks:
            combined_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
            for mask in self.object_masks.values():
                combined_mask |= mask.astype(bool)
            return combined_mask.astype(np.uint8)
        else:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    
    def simplify_frame(
        self, 
        image_rgb: np.ndarray, 
        task_description: str,
        robot_mask: Optional[np.ndarray] = None,
        object_mask: Optional[np.ndarray] = None,
        force_redetect: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate task-relevant mask using dual strategy: environment segmentation + GroundingSAM2.
        
        The final mask is a combination of:
        1. Environment-provided masks (robot_mask, object_mask) - 100% reliable
        2. GroundingSAM2 detected masks - semantic detection as supplement
        
        Args:
            image_rgb: Input RGB image (H, W, 3), uint8
            task_description: Task description for object detection
            robot_mask: Optional environment-provided robot arm mask (H, W), uint8, 1=robot
            object_mask: Optional environment-provided object mask (H, W), uint8, 1=object
            force_redetect: Force re-detection even if not at interval
            
        Returns:
            Tuple[image_rgb, combined_mask]: Original image and combined mask (H, W), uint8
        """
        logger.info(f"simplify_frame called: episode_frame={self.episode_frame_count}, "
                   f"total_frames={self.total_frames}, task='{task_description}'")
        
        # Parse task description to prompt
        prompt = parse_task_description(task_description)
        
        # Check if we need to initialize/re-initialize tracking
        should_reinit = (
            self.inference_state is None or 
            prompt != self.current_prompt or 
            force_redetect or
            self.force_reinit_next_frame or
            self.episode_frame_count % self.detection_interval == 0
        )
        
        if should_reinit:
            if self.episode_frame_count % self.detection_interval == 0 and self.episode_frame_count > 0:
                logger.info(
                    f"[Periodic Re-detection] Frame {self.episode_frame_count}: "
                    f"Re-initializing tracking (detection_interval={self.detection_interval})"
                )
            
            logger.debug(f"Initializing tracking with prompt: {prompt}")
            self.current_prompt = prompt
            self._init_video_tracking(image_rgb, prompt)
            self.force_reinit_next_frame = False
        
        # Get GroundingSAM2 detected mask for current frame
        gsam_mask = self._track_frame(image_rgb)
        
        # === Dual Mask Strategy: Combine environment masks with GroundingSAM2 mask ===
        combined_mask = gsam_mask.astype(bool)
        
        # Log GroundingSAM2 mask stats
        gsam_nonzero = np.count_nonzero(gsam_mask)
        logger.info(f"GroundingSAM2 mask: {gsam_nonzero}/{gsam_mask.size} pixels ({gsam_nonzero*100/gsam_mask.size:.2f}%)")
        
        # Merge robot mask from environment (if provided)
        if robot_mask is not None:
            # Ensure robot_mask matches image shape
            if robot_mask.shape[:2] != image_rgb.shape[:2]:
                robot_mask = cv2.resize(
                    robot_mask.astype(np.uint8), 
                    (image_rgb.shape[1], image_rgb.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            robot_mask_bool = robot_mask.astype(bool)
            combined_mask |= robot_mask_bool
            
            robot_nonzero = np.count_nonzero(robot_mask_bool)
            logger.info(f"[ENV MASK] Robot mask: {robot_nonzero}/{robot_mask_bool.size} pixels ({robot_nonzero*100/robot_mask_bool.size:.2f}%)")
        else:
            logger.debug("No robot_mask provided from environment")
        
        # Merge object mask from environment (if provided)
        if object_mask is not None:
            # Ensure object_mask matches image shape
            if object_mask.shape[:2] != image_rgb.shape[:2]:
                object_mask = cv2.resize(
                    object_mask.astype(np.uint8), 
                    (image_rgb.shape[1], image_rgb.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            object_mask_bool = object_mask.astype(bool)
            combined_mask |= object_mask_bool
            
            object_nonzero = np.count_nonzero(object_mask_bool)
            logger.info(f"[ENV MASK] Object mask: {object_nonzero}/{object_mask_bool.size} pixels ({object_nonzero*100/object_mask_bool.size:.2f}%)")
        else:
            logger.debug("No object_mask provided from environment")
        
        # Convert to uint8 for output
        combined_mask = combined_mask.astype(np.uint8)
        
        # Log final combined mask stats
        final_nonzero = np.count_nonzero(combined_mask)
        logger.info(f"Final combined mask: {final_nonzero}/{combined_mask.size} pixels ({final_nonzero*100/combined_mask.size:.2f}%)")
        
        # Save visualization if needed (always save when save_dir is set)
        if self.save_dir:
            logger.info(f"Saving visualization for frame {self.total_frames}")
            self._save_visualization(
                image_rgb, combined_mask, 
                task_description=task_description, prompt=prompt,
                gsam_mask=gsam_mask,
                robot_mask=robot_mask,
                object_mask=object_mask
            )
        
        self.episode_frame_count += 1  # Increment episode frame counter
        self.total_frames += 1
        
        return image_rgb, combined_mask
    
    def _save_visualization(
        self, 
        original: np.ndarray, 
        mask: np.ndarray, 
        task_description: str = "", 
        prompt: str = "",
        gsam_mask: Optional[np.ndarray] = None,
        robot_mask: Optional[np.ndarray] = None,
        object_mask: Optional[np.ndarray] = None
    ):
        """
        Save visualization images with text annotations and multi-source mask overlay.
        
        Uses different colors for different mask sources:
        - Green: GroundingSAM2 detected mask
        - Blue: Environment robot mask
        - Red: Environment object mask
        - Final overlay shows combined mask
        
        Args:
            original: Original RGB image
            mask: Final combined mask
            task_description: Task description text
            prompt: GroundingDINO prompt
            gsam_mask: GroundingSAM2 detected mask (optional)
            robot_mask: Environment robot mask (optional)
            object_mask: Environment object mask (optional)
        """
        frame_name = f"frame_{self.total_frames:05d}.jpg"
        
        # Log task and prompt to file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{self.total_frames:05d}\t{task_description}\t{prompt}\n")
        
        # Calculate mask statistics
        mask_nonzero = np.count_nonzero(mask)
        mask_total = mask.size
        mask_ratio = mask_nonzero / mask_total if mask_total > 0 else 0.0
        logger.info(f"[DEBUG] Frame {self.total_frames}: final mask stats - "
                   f"non-zero={mask_nonzero}/{mask_total} ({mask_ratio*100:.2f}%), "
                   f"dtype={mask.dtype}, range=[{mask.min()}, {mask.max()}]")
        
        # Convert original to BGR for OpenCV
        original_bgr = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2BGR)
        h, w = original.shape[:2]
        
        # Create overlay with multi-source mask coloring
        overlay = original_bgr.copy().astype(np.float32)
        
        # Helper function to normalize and resize mask
        def prepare_mask(m):
            if m is None:
                return None
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            if m.dtype == np.uint8:
                return (m.astype(np.float32) / 255.0) if m.max() > 1 else m.astype(np.float32)
            return m.astype(np.float32)
        
        # Prepare individual masks
        gsam_norm = prepare_mask(gsam_mask)
        robot_norm = prepare_mask(robot_mask)
        object_norm = prepare_mask(object_mask)
        
        # Apply color overlays for each mask source
        # Color scheme (BGR): Green=GSAM, Blue=Robot, Red=Object
        alpha = 0.3  # Overlay transparency
        
        if gsam_norm is not None:
            # Green for GroundingSAM2 detected regions
            green_overlay = np.zeros_like(overlay)
            green_overlay[:, :, 1] = 255  # Green channel
            mask_3d = np.stack([gsam_norm] * 3, axis=2)
            overlay = overlay * (1 - mask_3d * alpha) + green_overlay * (mask_3d * alpha)
        
        if robot_norm is not None:
            # Blue for robot mask from environment
            blue_overlay = np.zeros_like(overlay)
            blue_overlay[:, :, 0] = 255  # Blue channel
            mask_3d = np.stack([robot_norm] * 3, axis=2)
            overlay = overlay * (1 - mask_3d * alpha) + blue_overlay * (mask_3d * alpha)
        
        if object_norm is not None:
            # Red for object mask from environment
            red_overlay = np.zeros_like(overlay)
            red_overlay[:, :, 2] = 255  # Red channel
            mask_3d = np.stack([object_norm] * 3, axis=2)
            overlay = overlay * (1 - mask_3d * alpha) + red_overlay * (mask_3d * alpha)
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Add text annotations
        y_offset = 25
        line_height = 25
        
        if prompt:
            cv2.putText(overlay, f"Prompt: {prompt}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height
        
        if task_description:
            cv2.putText(overlay, f"Task: {task_description}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Add mask statistics
        cv2.putText(overlay, f"Combined: {mask_nonzero}/{mask_total} ({mask_ratio*100:.1f}%)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        
        # Add individual mask stats
        if gsam_norm is not None:
            gsam_nonzero = np.count_nonzero(gsam_mask)
            cv2.putText(overlay, f"GSAM (green): {gsam_nonzero} px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20
        
        if robot_norm is not None:
            robot_nonzero = np.count_nonzero(robot_mask)
            cv2.putText(overlay, f"Robot (blue): {robot_nonzero} px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            y_offset += 20
        
        if object_norm is not None:
            obj_nonzero = np.count_nonzero(object_mask)
            cv2.putText(overlay, f"Object (red): {obj_nonzero} px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
        cv2.imwrite(os.path.join(self.save_dir, frame_name), overlay)
        
        logger.debug(f"Saved visualization for frame {self.total_frames}")


def create_image_simplifier(
    device: str = "cuda",
    save_dir: Optional[str] = None,
    blur_sigma: float = 10.0,
    detection_interval: int = 1,
    box_threshold: Optional[float] = None,
    text_threshold: Optional[float] = None
) -> ImageSimplifier:
    """
    Factory function to create ImageSimplifier with default paths.
    
    Args:
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save visualizations
        blur_sigma: Deprecated, no longer used (kept for backward compatibility)
        detection_interval: Frame interval for re-detection (1 = every frame)
        box_threshold: Confidence threshold for object detection (default: 0.35)
        text_threshold: Text matching threshold (default: 0.25)
        
    Returns:
        Initialized ImageSimplifier instance
    """
    # Model paths (absolute)
    grounding_dino_config = "/home/futuremm/workplace/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    grounding_dino_checkpoint = "/home/futuremm/workplace/Grounded-SAM-2/gdino_checkpoints/checkpoint_best_regular.pth"
    sam2_checkpoint = "/home/futuremm/workplace/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    # SAM2 config: Hydra module path (relative to sam2 module, not file system path)
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # Get thresholds from environment variables or use provided/default values
    if box_threshold is None:
        box_threshold = float(os.environ.get('IMAGE_SIMPLIFICATION_BOX_THRESHOLD', '0.35'))
    if text_threshold is None:
        text_threshold = float(os.environ.get('IMAGE_SIMPLIFICATION_TEXT_THRESHOLD', '0.25'))
    
    logger.info("=" * 80)
    logger.info("Creating ImageSimplifier with absolute paths:")
    logger.info(f"  GroundingDINO config: {grounding_dino_config}")
    logger.info(f"  GroundingDINO checkpoint: {grounding_dino_checkpoint}")
    logger.info(f"  SAM2 checkpoint: {sam2_checkpoint}")
    logger.info(f"  SAM2 config: {sam2_config} (Hydra module path)")
    logger.info(f"  Box threshold: {box_threshold}")
    logger.info(f"  Text threshold: {text_threshold}")
    logger.info("=" * 80)
    
    return ImageSimplifier(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        blur_sigma=blur_sigma,
        save_dir=save_dir,
        detection_interval=detection_interval
    )
