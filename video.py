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
    MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
    MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"
    VAE_PATH = "vae.safetensors"
    USE_VAE = True
    
    # I/O configuration
    INPUT_VIDEO = "four.mov"
    OUTPUT_DIR = "output_frames"
    INPUT_FRAMES_DIR = "input_frames"  # NEW: Directory for input frames
    
    # Processing parameters
    TARGET_FPS = 8
    BATCH_SIZE = 16
    OVERLAP = 8
    STRENGTH = 0.2
    GUIDANCE_SCALE = 10
    NUM_INFERENCE_STEPS = 25
    SEED = 42
    
    # Temporal smoothing
    TEMPORAL_ALPHA = 0.3
    TEMPORAL_SIGMA = .5
    
    # Output settings
    OUTPUT_WIDTH = 512
    OUTPUT_HEIGHT = 512
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_SIZE_MB = 1000
    CACHE_OPTICAL_FLOW = True
    CACHE_PROCESSED_FRAMES = True
    
    # NEW: Frame output settings
    SAVE_INPUT_FRAMES = True  # Enable saving of loaded frames
    
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
                pil_frame = Image.fromarray(frame_rgb).resize(
                    (Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT), 
                    Image.LANCZOS
                )
                frames.append(pil_frame)
                
                # NEW: Save the frame immediately after loading
                if save_frames and output_dir:
                    frame_path = Path(output_dir) / f"input_frame_{extracted_count:04d}.png"
                    pil_frame.save(frame_path)
                    logger.info(f"Saved input frame {extracted_count}: {frame_path}")
                
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

def compute_optical_flow_cached(prev_frame: np.ndarray, next_frame: np.ndarray, 
                               cache: Optional[FrameCache] = None) -> np.ndarray:
    """Compute optical flow with caching support"""
    # Generate cache key from frame data
    if cache:
        key_data = hashlib.md5(prev_frame.tobytes() + next_frame.tobytes()).hexdigest()
        cache_key = f"flow_{key_data}"
        
        # Check cache
        cached_flow = cache.get(cache_key)
        if cached_flow is not None:
            return cached_flow
    
    # Compute flow
    if prev_frame.shape != next_frame.shape:
        raise ValueError("Frame dimensions must match")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Cache result
    if cache:
        cache.put(cache_key, flow)
    
    return flow

def apply_temporal_smoothing(frames: List[Image.Image], 
                            alpha: float = 0.5, 
                            sigma: float = 1.0,
                            cache: Optional[FrameCache] = None) -> List[Image.Image]:
    """Apply temporal smoothing with caching"""
    if not frames or len(frames) < 2:
        return frames.copy()
    
    logger.info(f"Applying temporal smoothing (alpha={alpha}, sigma={sigma})")
    
    np_frames = [np.array(frame) for frame in frames]
    smoothed_frames = [np_frames[0]]
    
    try:
        for i in range(1, len(np_frames)):
            # Compute optical flow (with caching)
            flow = compute_optical_flow_cached(
                np_frames[i-1], 
                np_frames[i],
                cache=cache
            )
            
            # Warp previous frame
            h, w = flow.shape[:2]
            flow_map = np.indices((h, w)).transpose(1, 2, 0) + flow
            
            warped = cv2.remap(
                np_frames[i-1], 
                flow_map.astype(np.float32), 
                None, 
                cv2.INTER_LINEAR
            )
            
            # Blend frames
            blended = cv2.addWeighted(
                np_frames[i], 1 - alpha, 
                warped, alpha, 
                0
            )
            
            # Apply Gaussian smoothing
            if sigma > 0:
                from scipy.ndimage import gaussian_filter
                blended = gaussian_filter(blended, sigma=(0, sigma, sigma, 0))
            
            smoothed_frames.append(blended)
            
    except Exception as e:
        logger.error(f"Error in temporal smoothing: {str(e)}")
        return frames
    
    return [Image.fromarray(np.uint8(frame)) for frame in smoothed_frames]

class VideoProcessor:
    """Main class for video processing pipeline with caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = None
        self.motion_adapter = None
        
        # Initialize cache
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
            self.pipeline = AnimateDiffVideoToVideoPipeline.from_pretrained(
                self.config.MODEL_ID,
                motion_adapter=self.motion_adapter,
                torch_dtype=self.config.dtype,
                variant="fp16" if self.config.device == "cuda" else None
            )
            
            # Load VAE if enabled and file exists
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
            
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.enable_vae_slicing()
            
            if self.config.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            
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
            # NEW: Pass save_frames and output_dir parameters
            input_frames = extract_frames(
                video_path, 
                target_fps=self.config.TARGET_FPS,
                save_frames=self.config.SAVE_INPUT_FRAMES,
                output_dir=self.config.INPUT_FRAMES_DIR if self.config.SAVE_INPUT_FRAMES else None
            )
            
            # Process frames in batches
            processed_frames = self._process_frames(input_frames)
            
            # Apply temporal smoothing with cache
            logger.info("Applying temporal smoothing...")
            processed_frames = apply_temporal_smoothing(
                processed_frames,
                alpha=self.config.TEMPORAL_ALPHA,
                sigma=self.config.TEMPORAL_SIGMA,
                cache=self.cache if self.config.CACHE_OPTICAL_FLOW else None
            )
            
            # Log cache statistics
            if self.cache:
                stats = self.cache.get_stats()
                logger.info(f"Cache stats: {stats}")
            
            if output_path:
                self._save_outputs(processed_frames, output_path, input_frames)
            
            return processed_frames
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise
    
    def _process_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Process frames in batches with overlap and caching"""
        processed_frames = []
        total_batches = (len(frames) - 1) // (self.config.BATCH_SIZE - self.config.OVERLAP) + 1
        
        for batch_idx in range(0, len(frames), self.config.BATCH_SIZE - self.config.OVERLAP):
            batch_end = min(batch_idx + self.config.BATCH_SIZE, len(frames))
            current_batch = frames[batch_idx:batch_end]
            
            logger.info(f"Processing batch {batch_idx//(self.config.BATCH_SIZE-self.config.OVERLAP) + 1}/{total_batches}")
            
            # Check cache for processed batch
            cache_key = None
            if self.cache and self.config.CACHE_PROCESSED_FRAMES:
                batch_hash = hashlib.md5(
                    b''.join(f.tobytes() for f in current_batch)
                ).hexdigest()
                cache_key = f"batch_{batch_hash}_{self.config.STRENGTH}_{self.config.SEED}"
                
                cached_batch = self.cache.get(cache_key)
                if cached_batch is not None:
                    logger.info("Using cached batch result")
                    current_styled = cached_batch
                else:
                    # Process batch
                    current_styled = self._process_batch(current_batch)
                    self.cache.put(cache_key, current_styled)
            else:
                current_styled = self._process_batch(current_batch)
            
            # Handle first batch
            if batch_idx == 0:
                processed_frames.extend(current_styled)
            else:
                # Blend overlapping region
                for i in range(min(self.config.OVERLAP, len(current_styled))):
                    if (self.config.OVERLAP - i) <= len(processed_frames):
                        weight = (i + 1) / (self.config.OVERLAP + 1)
                        prev_frame = processed_frames[-(self.config.OVERLAP - i)]
                        blended = Image.blend(prev_frame, current_styled[i], weight)
                        processed_frames[-(self.config.OVERLAP - i)] = blended
                
                processed_frames.extend(current_styled[self.config.OVERLAP:])
            
            logger.info(f"Processed {len(current_styled)} frames")
        
        return processed_frames
    
    def _process_batch(self, batch: List[Image.Image]) -> List[Image.Image]:
        """Process a single batch through the pipeline"""
        output = self.pipeline(
            prompt="Pixel art style, retro 8-bit aesthetic, game art",
            negative_prompt="low quality, worst quality, blurry, distorted, watermark, text, jpeg artifacts, ugly, deformed, photorealistic, 3d render",
            video=batch,
            strength=self.config.STRENGTH,
            guidance_scale=self.config.GUIDANCE_SCALE,
            num_inference_steps=self.config.NUM_INFERENCE_STEPS,
            generator=self.config.generator,
        )
        return output.frames[0]
    
    def _save_outputs(self, processed_frames: List[Image.Image], 
                     output_path: str, original_frames: List[Image.Image] = None) -> None:
        """Save processed frames and create output videos"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(processed_frames)} frames to {output_path}")
        for i, frame in enumerate(processed_frames):
            frame_path = output_path / f"styled_frame_{i:04d}.png"
            frame.save(frame_path)
        
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
                (self.config.OUTPUT_WIDTH, self.config.OUTPUT_HEIGHT)
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