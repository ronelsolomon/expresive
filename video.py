import os
import torch
import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler, AutoencoderKL
from diffusers.utils import export_to_gif, export_to_video
import hashlib
import pickle
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameCache:
    """Memory cache for processed frames and intermediate results"""
    
    def __init__(self, max_size_mb: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        
    def _get_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        if isinstance(obj, Image.Image):
            return obj.size[0] * obj.size[1] * 3  # RGB
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, list):
            return sum(self._estimate_size(item) for item in obj)
        else:
            return len(pickle.dumps(obj))
    
    def _evict_lru(self, needed_bytes: int) -> None:
        """Evict least recently used items to free memory"""
        if not self.cache:
            return
            
        # Simple FIFO eviction (in production, use proper LRU)
        keys_to_remove = []
        freed_bytes = 0
        
        for key in list(self.cache.keys()):
            if freed_bytes >= needed_bytes:
                break
            item_size = self._estimate_size(self.cache[key])
            keys_to_remove.append(key)
            freed_bytes += item_size
        
        for key in keys_to_remove:
            self.current_size_bytes -= self._estimate_size(self.cache[key])
            del self.cache[key]
            logger.debug(f"Evicted cache entry: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache"""
        if key in self.cache:
            self.hit_count += 1
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        self.miss_count += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Store item in cache with memory management"""
        item_size = self._estimate_size(value)
        
        # Check if item is too large for cache
        if item_size > self.max_size_bytes:
            logger.warning(f"Item too large for cache: {item_size} bytes")
            return
        
        # Evict if needed
        if self.current_size_bytes + item_size > self.max_size_bytes:
            self._evict_lru(item_size)
        
        self.cache[key] = value
        self.current_size_bytes += item_size
        logger.debug(f"Cached item {key}: {item_size} bytes")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'entries': len(self.cache),
            'size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': f"{hit_rate:.2%}"
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.current_size_bytes = 0
        logger.info("Cache cleared")

class Config:
    """Configuration parameters for video processing"""
    # Model configuration
    MODEL_ID = "./Counterfeit-V3.0.safetensors"
    MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"
    VAE_PATH = ""
    USE_VAE = False
    
    # I/O configuration
    INPUT_VIDEO = "four.mov"
    OUTPUT_DIR = "output_frames"
    INPUT_FRAMES_DIR = "input_frames"
    
    # Processing parameters - IMPROVED DEFAULTS
    TARGET_FPS = 8
    BATCH_SIZE = 16
    OVERLAP = 12  # Increased for smoother transitions
    STRENGTH = 0.5  # Reduced for better quality preservation
    GUIDANCE_SCALE = 9  # Reduced from 10 for more natural results
    NUM_INFERENCE_STEPS = 40  # Increased for better quality
    SEED = 42
    
    # Temporal smoothing - IMPROVED DEFAULTS
    TEMPORAL_ALPHA = 0.2  # Increased for stronger smoothing
    TEMPORAL_SIGMA = .1  # Increased for more blur
    
    # Output settings
    OUTPUT_WIDTH = 512
    OUTPUT_HEIGHT = 384
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_SIZE_MB = 1000
    CACHE_PROCESSED_FRAMES = True
    
    # Frame output settings
    SAVE_INPUT_FRAMES = True
    
    # NEW: Advanced settings for better quality
    USE_BILATERAL_FILTER = True  # Preserve edges while smoothing
    USE_ADAPTIVE_STRENGTH = True  # Adjust strength based on motion
    APPLY_COLOR_CORRECTION = True  # Maintain color consistency
    USE_GUIDED_FILTER = True  # Better edge-aware smoothing
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
        if self.SAVE_INPUT_FRAMES:
            Path(self.INPUT_FRAMES_DIR).mkdir(exist_ok=True, parents=True)
        logger.info(f"Using device: {self.device} (dtype: {self.dtype})")
        
        self.generator = torch.Generator(self.device).manual_seed(self.SEED)
        self.pipeline = None
        self.motion_adapter = None

def extract_frames(video_path: str, max_frames: Optional[int] = None, 
                   target_fps: int = 8, save_frames: bool = False,
                   output_dir: str = None) -> List[Image.Image]:
    """Extract frames from video with specified FPS and optionally save them."""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
        
    frames = []
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / target_fps))
    
    logger.info(f"Extracting frames from {video_path}")
    logger.info(f"Original FPS: {original_fps}, extracting every {frame_skip} frames")
    
    if save_frames and output_dir:
        logger.info(f"Saving input frames to: {output_dir}")
    
    frame_count = 0
    extracted_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # IMPROVED: Use LANCZOS for better downsampling
                pil_frame = Image.fromarray(frame_rgb).resize(
                    (Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT), 
                    Image.Resampling.LANCZOS
                )
                frames.append(pil_frame)
                
                if save_frames and output_dir:
                    frame_path = Path(output_dir) / f"input_frame_{extracted_count:04d}.png"
                    pil_frame.save(frame_path)
                
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
            
    finally:
        cap.release()
    
    if not frames:
        raise ValueError("No frames were extracted from the video")
        
    logger.info(f"Successfully extracted {len(frames)} frames")
    return frames


def apply_bilateral_filter(frame: np.ndarray, d: int = 9, 
                          sigma_color: float = 75, 
                          sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter to preserve edges while smoothing"""
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

def apply_guided_filter(frame: np.ndarray, guide: np.ndarray, 
                       radius: int = 8, eps: float = 0.01) -> np.ndarray:
    """Apply guided filter for edge-aware smoothing"""
    try:
        # Simple guided filter implementation
        mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(frame, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * frame, cv2.CV_32F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        return mean_a * guide + mean_b
    except:
        return frame

def match_color_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match color histogram of source to reference for consistency"""
    result = np.zeros_like(source)
    
    for channel in range(3):
        src_hist, _ = np.histogram(source[:, :, channel].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference[:, :, channel].flatten(), 256, [0, 256])
        
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        lookup_table = np.interp(src_cdf, ref_cdf, np.arange(256))
        result[:, :, channel] = lookup_table[source[:, :, channel]]
    
    return result.astype(np.uint8)

def apply_temporal_smoothing(frames: List[Image.Image], 
                            alpha: float = 0.5, 
                            sigma: float = 1.0,
                            cache: Optional[FrameCache] = None,
                            use_bilateral: bool = True,
                            use_guided: bool = True,
                            use_color_correction: bool = True) -> List[Image.Image]:
    """Apply temporal smoothing with advanced filtering"""
    if not frames or len(frames) < 2:
        return frames.copy()
    
    logger.info(f"Applying temporal smoothing (alpha={alpha}, sigma={sigma})")
    logger.info(f"Advanced filters: bilateral={use_bilateral}, guided={use_guided}, color_correction={use_color_correction}")
    
    np_frames = [np.array(frame) for frame in frames]
    smoothed_frames = [np_frames[0]]
    
    try:
        for i in range(1, len(np_frames)):
            # Simple frame blending without optical flow
            blended = cv2.addWeighted(
                np_frames[i].astype(np.float32), 1 - alpha, 
                np_frames[i-1].astype(np.float32), alpha, 
                0
            ).astype(np.uint8)
            
            # Apply bilateral filter for edge preservation
            if use_bilateral:
                blended = apply_bilateral_filter(blended)
            
            # Apply guided filter using current frame as guide
            if use_guided:
                blended_float = blended.astype(np.float32) / 255.0
                guide_float = np_frames[i].astype(np.float32) / 255.0
                blended_float = apply_guided_filter(blended_float, guide_float)
                blended = (blended_float * 255).clip(0, 255).astype(np.uint8)
            
            # Color correction
            if use_color_correction and i > 0:
                blended = match_color_histogram(blended, np_frames[i])
            
            # Apply Gaussian smoothing
            if sigma > 0:
                blended = cv2.GaussianBlur(blended, (0, 0), sigma)
            
            smoothed_frames.append(blended)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Smoothed {i+1}/{len(np_frames)} frames")
            
    except Exception as e:
        logger.error(f"Error in temporal smoothing: {str(e)}")
        return frames
    
    return [Image.fromarray(frame) for frame in smoothed_frames]


class VideoProcessor:
    """Main class for video processing pipeline with caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = None
        self.motion_adapter = None
        
        if config.CACHE_ENABLED:
            self.cache = FrameCache(max_size_mb=config.CACHE_SIZE_MB)
            logger.info(f"Memory cache enabled: {config.CACHE_SIZE_MB}MB")
        else:
            self.cache = None
            logger.info("Memory cache disabled")
    
    def initialize_models(self) -> None:
        """Initialize all required models and pipelines"""
        try:
            logger.info("Loading Motion Adapter...")
            self.motion_adapter = MotionAdapter.from_pretrained(
                self.config.MOTION_ADAPTER_ID,
                torch_dtype=self.config.dtype
            )
            
            logger.info("Loading AnimateDiff Video-to-Video Pipeline...")
            if os.path.isfile(self.config.MODEL_ID):
                # Load from local file
                self.pipeline = AnimateDiffVideoToVideoPipeline.from_single_file(
                    self.config.MODEL_ID,
                    motion_adapter=self.motion_adapter,
                    torch_dtype=self.config.dtype
                )
            else:
                # Fallback to loading from Hugging Face Hub
                self.pipeline = AnimateDiffVideoToVideoPipeline.from_pretrained(
                    self.config.MODEL_ID,
                    motion_adapter=self.motion_adapter,
                    torch_dtype=self.config.dtype,
                    variant="fp16" if self.config.device == "cuda" else None
                )
            
            if self.config.USE_VAE and Path(self.config.VAE_PATH).exists():
                try:
                    logger.info(f"Loading VAE from {self.config.VAE_PATH}")
                    vae = AutoencoderKL.from_single_file(
                        self.config.VAE_PATH,
                        torch_dtype=self.config.dtype
                    )
                    self.pipeline.vae = vae.to(self.config.device)
                    logger.info("VAE loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load VAE: {str(e)}")
            else:
                logger.info("Using default VAE")
            
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config,
                beta_schedule="linear",
                timestep_spacing="leading"
            )
            self.pipeline.enable_vae_slicing()
            
            if self.config.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                # IMPROVED: Enable memory efficient attention
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except:
                    logger.info("xformers not available, using default attention")
            
            self.pipeline = self.pipeline.to(self.config.device)
            logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    def process_video(self, video_path: str, output_path: str = None) -> List[Image.Image]:
        """Process a video file through the pipeline with caching"""
        if not self.pipeline:
            self.initialize_models()
        
        try:
            logger.info(f"Extracting frames from {video_path}...")
            input_frames = extract_frames(
                video_path, 
                target_fps=self.config.TARGET_FPS,
                save_frames=self.config.SAVE_INPUT_FRAMES,
                output_dir=self.config.INPUT_FRAMES_DIR if self.config.SAVE_INPUT_FRAMES else None
            )
            
            # Process frames in batches
            processed_frames = self._process_frames(input_frames)
            
            # Apply temporal smoothing with advanced filters
            logger.info("Applying temporal smoothing with advanced filters...")
            processed_frames = apply_temporal_smoothing(
                processed_frames,
                alpha=self.config.TEMPORAL_ALPHA,
                sigma=self.config.TEMPORAL_SIGMA,
                cache=self.cache,
                use_bilateral=self.config.USE_BILATERAL_FILTER,
                use_guided=self.config.USE_GUIDED_FILTER,
                use_color_correction=self.config.APPLY_COLOR_CORRECTION
            )
            
            if self.cache:
                stats = self.cache.get_stats()
                logger.info(f"Cache stats: {stats}")
            
            if output_path:
                self._save_outputs(processed_frames, output_path, input_frames)
            
            return processed_frames
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise
    
    def _calculate_adaptive_strength(self, batch: List[Image.Image]) -> float:
        """Calculate adaptive strength based on motion in the batch"""
        if not self.config.USE_ADAPTIVE_STRENGTH or len(batch) < 2:
            return self.config.STRENGTH
        
        try:
            np_frames = [np.array(frame) for frame in batch[:2]]
            flow = compute_optical_flow_cached(np_frames[0], np_frames[1], cache=self.cache)
            motion = calculate_motion_magnitude(flow)
            
            # Adjust strength: less motion = more strength (more stylization)
            # More motion = less strength (preserve motion)
            base_strength = self.config.STRENGTH
            motion_threshold = 5.0
            
            if motion < motion_threshold:
                adaptive_strength = base_strength * 1.2
            else:
                adaptive_strength = base_strength * 0.8
            
            adaptive_strength = np.clip(adaptive_strength, 0.05, 0.4)
            logger.info(f"Adaptive strength: {adaptive_strength:.3f} (motion: {motion:.2f})")
            
            return adaptive_strength
            
        except:
            return self.config.STRENGTH
    
    def _match_style_to_reference(self, frame: Image.Image, reference: Image.Image) -> Image.Image:
        """Match the color style of frame to reference using histogram matching"""
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')
        if reference.mode != 'RGB':
            reference = reference.convert('RGB')
            
        frame_np = np.array(frame)
        reference_np = np.array(reference)
        
        # Convert to float32 for calculations
        frame_np = frame_np.astype('float32')
        reference_np = reference_np.astype('float32')
        
        # Calculate mean and std for each channel
        frame_mean = frame_np.mean(axis=(0, 1))
        frame_std = frame_np.std(axis=(0, 1)) + 1e-8  # Avoid division by zero
        
        ref_mean = reference_np.mean(axis=(0, 1))
        ref_std = reference_np.std(axis=(0, 1)) + 1e-8
        
        # Apply color matching
        result = (frame_np - frame_mean) * (ref_std / frame_std) + ref_mean
        
        # Clip values to valid range and convert back to uint8
        result = np.clip(result, 0, 255).astype('uint8')
        
        return Image.fromarray(result)
    
    def _process_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Process frames with strong temporal consistency"""
        if not frames:
            return []
            
        processed_frames = []
        
        # Process first batch to establish style
        first_batch = frames[:self.config.BATCH_SIZE]
        
        # First pass to establish style
        first_styled = self._process_batch(first_batch, strength=self.config.STRENGTH * 1.2)
        style_reference = first_styled[0].copy()
        
        # Second pass with its own style as reference for consistency
        first_styled = self._process_batch(
            first_batch,
            style_reference=style_reference,
            strength=self.config.STRENGTH * 0.8
        )
        processed_frames = first_styled.copy()
        style_reference = first_styled[0].copy()  # Update style reference with refined style
        
        logger.info("Processed first batch twice for better style consistency")
        
        for batch_idx in range(self.config.BATCH_SIZE - self.config.OVERLAP, len(frames), 
                            self.config.BATCH_SIZE - self.config.OVERLAP):
            batch_end = min(batch_idx + self.config.BATCH_SIZE, len(frames))
            current_batch = frames[batch_idx:batch_end]
            
            if not current_batch:
                continue
            
            # CRITICAL: Use last processed frames as initialization
            init_frames = processed_frames[-(self.config.OVERLAP):] + current_batch
            
            # Process with lower strength after first batch
            current_styled = self._process_batch(
                init_frames,
                style_reference=style_reference,
                strength=self.config.STRENGTH * 0.8  # Even lower for subsequent batches
            )
            
            # Remove overlap frames from output
            current_styled = current_styled[self.config.OVERLAP:]
            
            # Smoother blending with more overlap frames
            overlap_size = min(self.config.OVERLAP, len(processed_frames))
            for i in range(overlap_size):
                weight = (i + 1) / (overlap_size + 1)
                # Stronger blending near boundaries
                weight = 0.5 + 0.5 * np.sin((weight - 0.5) * np.pi)
                
                blended = Image.blend(
                    processed_frames[-(overlap_size - i)],
                    current_styled[i] if i < len(current_styled) else processed_frames[-(overlap_size - i)],
                    weight
                )
                processed_frames[-(overlap_size - i)] = blended
            
            # Add new frames
            if len(current_styled) > overlap_size:
                processed_frames.extend(current_styled[overlap_size:])
            
            logger.info(f"Processed batch {batch_idx // (self.config.BATCH_SIZE - self.config.OVERLAP) + 1}")
        
        return processed_frames
    
    def _process_batch(self, batch: List[Image.Image], style_reference: Image.Image = None, strength: float = None) -> List[Image.Image]:
        """Process a single batch through the pipeline with style reference"""
        if not batch:
            return []
            
        if strength is None:
            strength = self.config.STRENGTH
        
        # If we have a style reference, prepend it to the batch to guide the model
        if style_reference is not None:
            # Add the style reference as the first frame(s) to establish context
            batch_with_reference = [style_reference] + batch
            process_batch = batch_with_reference
        else:
            process_batch = batch
        
        try:
            output = self.pipeline(
                prompt="anime, cartoon, high quality, detailed, consistent style",  # Add "consistent style"
                negative_prompt=(
                    "low quality, worst quality, blurry, distorted, watermark, "
                    "text, jpeg artifacts, ugly, deformed, photorealistic, "
                    "3d render, dull colors, washed out, style change, inconsistent"  # Add style-related negatives
                ),
                video=process_batch,
                strength=strength,
                guidance_scale=self.config.GUIDANCE_SCALE,
                num_inference_steps=self.config.NUM_INFERENCE_STEPS,
                generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(self.config.SEED),
                output_type='pil'
            )
            
            if hasattr(output, 'frames') and output.frames:
                result = output.frames[0] if isinstance(output.frames[0], list) else output.frames
                
                # If we added a style reference, remove it from the output
                if style_reference is not None and len(result) > len(batch):
                    result = result[1:]  # Skip the first frame (style reference)
                
                # Apply color/style matching to each frame
                if style_reference is not None:
                    result = [self._match_style_to_reference(frame, style_reference) for frame in result]
                
                return result[:len(batch)]  # Ensure correct length
                
            return [Image.new('RGB', (100, 100), 'black')] * len(batch)
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return [Image.new('RGB', (100, 100), 'black')] * len(batch)
    
    def _save_outputs(self, processed_frames: List[Image.Image], 
                     output_path: str, original_frames: List[Image.Image] = None) -> None:
        """Save processed frames and create output videos"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(processed_frames)} frames to {output_path}")
        for i, frame in enumerate(processed_frames):
            frame_path = output_path / f"styled_frame_{i:04d}.png"
            frame.save(frame_path, optimize=True)  # IMPROVED: Enable PNG optimization
        
        gif_path = output_path / "styled_video.gif"
        export_to_gif(processed_frames, str(gif_path), fps=self.config.TARGET_FPS)
        logger.info(f"Saved GIF: {gif_path}")
        
        mp4_path = output_path / "styled_video.mp4"
        export_to_video(processed_frames, str(mp4_path), fps=self.config.TARGET_FPS)
        logger.info(f"Saved MP4: {mp4_path}")
        
        if original_frames:
            self._create_comparison_video(original_frames, processed_frames, output_path)
    
    def _create_comparison_video(self, original_frames: List[Image.Image],
                               processed_frames: List[Image.Image],
                               output_path: Path) -> None:
        """Create side-by-side comparison video"""
        comparison_frames = []
        min_frames = min(len(original_frames), len(processed_frames))
        
        for i in range(min_frames):
            orig_resized = original_frames[i].resize(
                (self.config.OUTPUT_WIDTH, self.config.OUTPUT_HEIGHT),
                Image.Resampling.LANCZOS
            )
            
            combined = Image.new('RGB', 
                               (self.config.OUTPUT_WIDTH * 2, self.config.OUTPUT_HEIGHT))
            combined.paste(orig_resized, (0, 0))
            combined.paste(processed_frames[i], (self.config.OUTPUT_WIDTH, 0))
            
            comparison_frames.append(combined)
        
        comparison_path = output_path / "comparison.mp4"
        export_to_video(comparison_frames, str(comparison_path), fps=self.config.TARGET_FPS)
        logger.info(f"Saved comparison video: {comparison_path}")

def main():
    """Main function to run the video processing pipeline"""
    try:
        config = Config()
        processor = VideoProcessor(config)
        
        processor.process_video(
            video_path=config.INPUT_VIDEO,
            output_path=config.OUTPUT_DIR
        )
        
        logger.info("\n" + "="*50)
        logger.info("✅ PROCESSING COMPLETE!")
        logger.info(f"   📁 Output directory: {config.OUTPUT_DIR}")
        if config.SAVE_INPUT_FRAMES:
            logger.info(f"   📥 Input frames: {config.INPUT_FRAMES_DIR}")
        logger.info("   🎬 styled_video.mp4 - Final styled video")
        logger.info("   🎨 styled_video.gif - Animated GIF")
        logger.info("   📊 comparison.mp4 - Side-by-side comparison")
        
        if config.CACHE_ENABLED and processor.cache:
            stats = processor.cache.get_stats()
            logger.info(f"\n   💾 Cache Performance:")
            logger.info(f"      - Entries: {stats['entries']}")
            logger.info(f"      - Memory: {stats['size_mb']:.2f}MB / {stats['max_size_mb']:.0f}MB")
            logger.info(f"      - Hit rate: {stats['hit_rate']}")
        
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())