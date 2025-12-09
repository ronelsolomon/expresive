"""
video_analyzer.py - Core Video Analysis Engine
Handles video processing, scene detection, and frame extraction
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json


class VideoAnalyzer:
    """Core video analysis and scene segmentation"""
    
    def __init__(self, video_path: str, object_detector=None):
        """
        Initialize video analyzer
        
        Args:
            video_path: Path to video file
            object_detector: Optional object detection model
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.object_detector = object_detector
        
        print(f"Video loaded: {self.width}x{self.height}, {self.fps:.2f} fps, "
              f"{self.duration:.2f}s ({self.frame_count} frames)")
        
    def __del__(self):
        """Clean up video capture"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def extract_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame from video
        
        Args:
            frame_number: Frame index to extract
            
        Returns:
            Frame as numpy array or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract frame at specific timestamp
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array or None if failed
        """
        frame_number = int(timestamp * self.fps)
        return self.extract_frame(frame_number)
    
    def calculate_frame_difference(self, frame1: np.ndarray, 
                                 frame2: np.ndarray) -> float:
        """
        Calculate difference between two frames using both histogram and edge detection
        
        Args:
            frame1, frame2: Input frames
            
        Returns:
            Combined difference score (0-1, higher means more different)
        """
        if frame1 is None or frame2 is None:
            return 0.0
            
        # Convert to grayscale for edge detection and histogram comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 1. Calculate histogram difference (using grayscale)
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms
        hist_diff = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 2. Calculate edge difference
        def get_edge_energy(image):
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            # Detect edges using Canny
            edges = cv2.Canny(blurred, 50, 150)
            # Calculate edge energy (sum of edge pixels)
            return np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255.0)
            
        # Get edge energies for both frames
        edge_energy1 = get_edge_energy(gray1)
        edge_energy2 = get_edge_energy(gray2)
        
        # Calculate edge difference (normalized to 0-1)
        edge_diff = abs(edge_energy1 - edge_energy2) / max(edge_energy1, edge_energy2, 1e-6)
        
        # 3. Combine both metrics (weighted average)
        # Give more weight to edge detection for B&W videos
        combined_diff = 0.3 * hist_diff + 0.7 * edge_diff
        
        return float(combined_diff)
    
    def detect_scene_boundaries(self, threshold: float = 0.15, 
                               sample_rate: int = 3) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries using combined histogram and edge detection
        
        Args:
            threshold: Difference threshold for scene change (0-1)
            sample_rate: Sample every Nth frame for efficiency
        
        Returns:
            List of (start_time, end_time) tuples for each scene
        """
        boundaries = [0.0]  # Start with first frame
        prev_frame = None
        
        print(f"\nDetecting scene boundaries...")
        print(f"Using combined edge detection + histogram comparison")
        print(f"Threshold: {threshold}, Sample rate: every {sample_rate} frames")
        
        # First pass: detect potential scene changes
        diffs = []
        timestamps = []
        
        for frame_num in range(0, self.frame_count, sample_rate):
            frame = self.extract_frame(frame_num)
            
            if frame is None:
                continue
                
            if prev_frame is not None:
                # Use the new combined difference metric
                diff = self.calculate_frame_difference(prev_frame, frame)
                timestamp = frame_num / self.fps
                
                diffs.append(diff)
                timestamps.append(timestamp)
                
                if diff > threshold:
                    # Avoid boundaries too close together (min 1 second apart)
                    if not boundaries or timestamp - boundaries[-1] > 1.0:
                        boundaries.append(timestamp)
                        print(f"  Potential scene boundary at {timestamp:.2f}s (diff: {diff:.3f})")
            
            prev_frame = frame
        
        # If no boundaries found, try with a lower threshold
        if len(boundaries) <= 1 and threshold > 0.05:
            print("No scenes detected, trying with lower threshold...")
            return self.detect_scene_boundaries(threshold * 0.7, sample_rate)
        
        boundaries.append(self.duration)  # End with last frame
        
        # Second pass: refine boundaries by looking for local maxima in differences
        if len(diffs) > 1:
            refined_boundaries = [0.0]
            window_size = max(1, int(1.0 * self.fps / sample_rate))  # 1 second window
            
            for i in range(window_size, len(diffs) - window_size):
                if (diffs[i] > threshold and 
                    diffs[i] == max(diffs[i-window_size:i+window_size+1]) and 
                    (not refined_boundaries or timestamps[i] - refined_boundaries[-1] > 1.0)):
                    refined_boundaries.append(timestamps[i])
            
            if len(refined_boundaries) > 1:  # If we found any refined boundaries
                boundaries = refined_boundaries + [self.duration]
        
        # Create scene intervals
        scenes = [(boundaries[i], boundaries[i+1]) 
                  for i in range(len(boundaries)-1)]
        
        print(f"\nDetected {len(scenes)} scenes:")
        for i, (start, end) in enumerate(scenes, 1):
            print(f"  Scene {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        print()
        
        return scenes
    
    def calculate_motion_intensity(self, start_time: float, 
                                   end_time: float,
                                   sample_frames: int = 10) -> float:
        """
        Calculate average motion intensity in a time range using optical flow
        
        Args:
            start_time, end_time: Time range in seconds
            sample_frames: Number of frames to sample
            
        Returns:
            Average motion intensity score
        """
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        frame_step = max(1, (end_frame - start_frame) // sample_frames)
        
        motion_scores = []
        prev_gray = None
        
        for frame_num in range(start_frame, min(end_frame, self.frame_count), frame_step):
            frame = self.extract_frame(frame_num)
            if frame is None:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate optical flow magnitude
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_scores.append(np.mean(magnitude))
            
            prev_gray = gray
        
        return float(np.mean(motion_scores)) if motion_scores else 0.0
    
    def extract_color_palette(self, frame: np.ndarray, 
                              n_colors: int = 5) -> List[str]:
        """
        Extract dominant color palette from frame
        
        Args:
            frame: Input frame
            n_colors: Number of dominant colors to extract
            
        Returns:
            List of hex color strings
        """
        if frame is None:
            return []
        
        # Reshape frame to list of pixels
        pixels = frame.reshape(-1, 3).astype(np.float32)
        
        # Use k-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        
        # Convert BGR to RGB and then to hex
        colors = []
        for center in centers:
            b, g, r = center.astype(int)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colors.append(hex_color)
        
        return colors
    
    def analyze_scene(self, start_time: float, end_time: float) -> Dict:
        """
        Comprehensive scene analysis
        
        Args:
            start_time, end_time: Scene time range
            
        Returns:
            Dictionary with scene analysis results
        """
        # Get middle frame for analysis
        mid_time = (start_time + end_time) / 2
        frame = self.extract_frame_at_time(mid_time)
        
        if frame is None:
            return {
                'shot_type': 'Wide',
                'camera_move': 'Still',
                'time_of_day': 'unknown',
                'mood': 'neutral'
            }
        
        # Calculate motion intensity
        motion = self.calculate_motion_intensity(start_time, end_time)
        
        # Extract color palette
        colors = self.extract_color_palette(frame)
        
        # Analyze brightness for time of day
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 50:
            time_of_day = "night"
        elif avg_brightness < 100:
            time_of_day = "dusk"
        elif avg_brightness < 200:
            time_of_day = "day"
        else:
            time_of_day = "dawn"
        
        # Infer shot type based on motion and frame content
        if motion < 2.0:
            camera_move = "Still"
        elif motion < 5.0:
            camera_move = "Push"
        else:
            camera_move = "Pan"
        
        # Infer mood from colors and motion
        if motion > 10.0:
            mood = "energetic"
        elif avg_brightness < 80:
            mood = "mysterious"
        elif motion < 2.0:
            mood = "calm"
        else:
            mood = "neutral"
        
        return {
            'shot_type': 'Wide',  # Default, can be improved with AI
            'camera_move': camera_move,
            'time_of_day': time_of_day,
            'mood': mood,
            'motion_intensity': motion,
            'color_palette': colors,
            'avg_brightness': float(avg_brightness)
        }
    
    def export_frames(self, output_dir: str, scenes: List[Tuple[float, float]]):
        """
        Export representative frames for each scene
        
        Args:
            output_dir: Directory to save frames
            scenes: List of scene time ranges
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, (start, end) in enumerate(scenes):
            mid_time = (start + end) / 2
            frame = self.extract_frame_at_time(mid_time)
            
            if frame is not None:
                filename = f"scene_{i+1:03d}_{mid_time:.2f}s.jpg"
                cv2.imwrite(str(output_path / filename), frame)
        
        print(f"Exported {len(scenes)} frames to {output_dir}")
