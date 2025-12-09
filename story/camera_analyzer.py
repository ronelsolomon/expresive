"""
Camera movement and shot type analysis using OpenCV
"""
import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional

class CameraMovement(Enum):
    """Types of camera movements"""
    STATIONARY = "stationary"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRUCK_LEFT = "truck_left"
    TRUCK_RIGHT = "truck_right"
    PEDESTAL_UP = "pedestal_up"
    PEDESTAL_DOWN = "pedestal_down"

class ShotType(Enum):
    """Types of camera shots"""
    BIRDS_EYE = "birds_eye"       # Directly overhead view (like a bird's eye)
    AERIAL = "aerial"             # High-angle shot from aircraft/drone
    ENVIRONMENTAL = "environmental" # Shows natural or architectural environment
    EXTREME_WIDE = "extreme_wide" # EWS (shows vast area, tiny subjects)
    WIDE = "wide"                # WS (shows full scene with context)
    MEDIUM = "medium"            # MS (shows subject from waist up)
    MEDIUM_CLOSE = "medium_close" # MCU (shows subject from chest up)
    CLOSE_UP = "close_up"        # CU (shows face and shoulders)
    EXTREME_CLOSE_UP = "extreme_close_up"  # ECU (shows part of face or detail)

class CameraAnalyzer:
    """Analyze camera movements and shot types in video"""
    
    def __init__(self, frame_width: int, frame_height: int, focal_length: float = 1.0):
        """
        Initialize camera analyzer
        
        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames
            focal_length: Focal length in pixels (default: 1.0)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length = focal_length
        self.prev_gray = None
        self.prev_keypoints = None
        self.optical_flow = None
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def analyze_frame(self, frame: np.ndarray, frame_time: float = 0.0) -> Dict:
        """
        Analyze a single frame for camera movement and shot type
        
        Args:
            frame: Input BGR image
            frame_time: Timestamp of the frame in seconds
            
        Returns:
            Dictionary containing analysis results
        """
        if frame is None:
            return {"movement": CameraMovement.STATIONARY.value}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize keypoints if needed
        if self.prev_keypoints is None or len(self.prev_keypoints) < 10:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            self.prev_gray = gray
            return {
                "movement": CameraMovement.STATIONARY.value,
                "shot_type": self._classify_shot_type(frame)
            }
        
        # Calculate optical flow
        keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_keypoints, None, **self.lk_params
        )
        
        # Filter valid points
        if status is None:
            return {
                "movement": CameraMovement.STATIONARY.value,
                "shot_type": self._classify_shot_type(frame)
            }
            
        good_new = keypoints[status == 1]
        good_old = self.prev_keypoints[status == 1]
        
        if len(good_new) < 5:  # Not enough points for analysis
            return {
                "movement": CameraMovement.STATIONARY.value,
                "shot_type": self._classify_shot_type(frame),
                "num_tracked_points": len(good_new)
            }
        
        # Calculate motion vectors
        motion_vectors = good_new - good_old
        
        # Analyze motion patterns
        movement = self._classify_movement(motion_vectors, good_old)
        
        # Update for next frame
        self.prev_gray = gray.copy()
        self.prev_keypoints = good_new.reshape(-1, 1, 2)
        
        # Classify shot type
        shot_analysis = self._classify_shot_type(frame)
        
        # Get shot type from analysis or use default
        shot_type = shot_analysis.get('shot_type', 'unknown')
        if hasattr(shot_type, 'value'):  # If it's an enum, get its value
            shot_type = shot_type.value
            
        return {
            "movement": movement.value,
            "shot_type": shot_type,
            "framing": shot_analysis.get('framing', 'unknown'),
            "visual_focus": shot_analysis.get('visual_focus', 'unknown'),
            "notes": shot_analysis.get('notes', ''),
            "num_tracked_points": len(good_new),
            "motion_vectors": motion_vectors.tolist()
        }
    
    def _classify_movement(self, motion_vectors: np.ndarray, points: np.ndarray) -> CameraMovement:
        """Classify camera movement based on optical flow"""
        mean_flow = np.mean(motion_vectors, axis=0)
        mean_x, mean_y = mean_flow
        
        # Calculate divergence (zoom) by comparing center and edge movements
        center = np.array([[self.frame_width/2, self.frame_height/2]])
        center_dist = np.linalg.norm(points - center, axis=1)
        is_center = center_dist < min(self.frame_width, self.frame_height) / 4
        
        if any(is_center) and any(~is_center):
            center_flow = np.mean(motion_vectors[is_center], axis=0)
            edge_flow = np.mean(motion_vectors[~is_center], axis=0)
            zoom_factor = np.linalg.norm(center_flow - edge_flow)
        else:
            zoom_factor = 0
        
        # Thresholds (may need adjustment)
        zoom_threshold = 0.5
        pan_tilt_threshold = 1.0
        
        if zoom_factor > zoom_threshold:
            if np.linalg.norm(center_flow) < np.linalg.norm(edge_flow):
                return CameraMovement.ZOOM_IN
            else:
                return CameraMovement.ZOOM_OUT
        elif abs(mean_x) > pan_tilt_threshold:
            return CameraMovement.PAN_LEFT if mean_x < 0 else CameraMovement.PAN_RIGHT
        elif abs(mean_y) > pan_tilt_threshold:
            return CameraMovement.TILT_UP if mean_y < 0 else CameraMovement.TILT_DOWN
        else:
            return CameraMovement.STATIONARY
    
    def _is_aerial_view(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """Detect if frame shows an aerial or bird's eye view
        
        Returns:
            Tuple of (is_aerial, is_birds_eye)
        """
        if frame is None or len(frame.shape) != 3:
            return False, False
            
        height, width = frame.shape[:2]
        
        # Check for high-angle perspective
        # 1. Edge detection to find horizon line and perspective lines
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Count lines that suggest perspective (converging lines)
            perspective_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Look for near-vertical lines (common in aerial views of buildings)
                if abs(angle) > 60 and abs(angle) < 120:
                    perspective_lines += 1
            
            # If we have multiple perspective lines, it's likely an aerial view
            if perspective_lines > 3:
                # Check if it's directly overhead (bird's eye)
                # Look for grid-like patterns common in top-down views
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
                
                # Check for grid pattern in frequency domain
                h, w = gray.shape
                cy, cx = h//2, w//2
                # Look for cross pattern in frequency domain
                cross_intensity = (np.sum(magnitude_spectrum[cy-5:cy+5, :]) + 
                                 np.sum(magnitude_spectrum[:, cx-5:cx+5])) / (10*w + 10*h)
                
                if cross_intensity > 50:  # Threshold for grid-like patterns
                    return True, True  # Bird's eye view
                return True, False  # Regular aerial view
                
        return False, False

    def _classify_shot_type(self, frame: np.ndarray) -> Dict:
        """
        Classify shot type and framing based on image analysis
        
        Returns:
            Dictionary containing shot type and framing information
        """
        result = {
            'shot_type': ShotType.WIDE,
            'framing': 'standard',
            'visual_focus': 'center',
            'notes': ''
        }
        
        if frame is None:
            return result
            
        # Convert to grayscale for some operations
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        height, width = frame.shape[:2]
        frame_area = height * width
        
        # First check for aerial/bird's eye view
        is_aerial, is_birds_eye = self._is_aerial_view(frame)
        if is_aerial:
            if is_birds_eye:
                result['shot_type'] = ShotType.BIRDS_EYE
                result['framing'] = 'top_down'
                result['visual_focus'] = 'layout'
                result['notes'] = 'Direct overhead view showing spatial relationships'
            else:
                result['shot_type'] = ShotType.AERIAL
                result['framing'] = 'high_angle'
                result['visual_focus'] = 'landscape'
                result['notes'] = 'Aerial perspective showing large area from elevation'
            return result

        # 1. Detect faces for human subjects
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Calculate largest face area
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            face_area = w * h
            face_ratio = face_area / frame_area
            
            # Determine shot type based on face size
            if face_ratio > 0.25:  # Face takes up >25% of frame
                result['shot_type'] = ShotType.EXTREME_CLOSE_UP
                result['framing'] = 'tight' if w > h * 0.8 else 'portrait_tight'
            elif face_ratio > 0.15:  # 10-25%
                result['shot_type'] = ShotType.CLOSE_UP
                result['framing'] = 'medium_tight' if w > h * 0.7 else 'portrait_medium'
            elif face_ratio > 0.08:  # 5-15%
                result['shot_type'] = ShotType.MEDIUM_CLOSE
                result['framing'] = 'medium' if w > h * 0.6 else 'portrait_loose'
            elif face_ratio > 0.04:  # 2-8%
                result['shot_type'] = ShotType.MEDIUM
                result['framing'] = 'medium_wide'
            else:
                result['shot_type'] = ShotType.WIDE
                result['framing'] = 'wide'
                
            # Check face position for rule of thirds
            face_center_x = x + w/2
            face_center_y = y + h/2
            
            # Rule of thirds positioning
            third_w = width / 3
            third_h = height / 3
            
            if abs(face_center_x - width/2) < third_w/2 and abs(face_center_y - height/2) < third_h/2:
                result['visual_focus'] = 'center'
            else:
                # Determine which third the face is in
                if face_center_x < third_w:
                    result['visual_focus'] = 'left_third'
                elif face_center_x > 2*third_w:
                    result['visual_focus'] = 'right_third'
                else:
                    result['visual_focus'] = 'center_vertical'
                    
                if face_center_y < third_h:
                    result['visual_focus'] += '_upper'
                elif face_center_y > 2*third_h:
                    result['visual_focus'] += '_lower'
        else:
            # No faces detected, use edge detection and other features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / frame_area
            
            # Calculate average brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Check for environmental/landscape shot characteristics
            is_landscape = width > height * 1.5
            has_high_detail = edge_density > 0.2
            has_architecture = edge_density > 0.15 and contrast > 40
            
            # Determine if this is an environmental shot (showing setting/environment)
            is_environmental_shot = is_landscape and (has_high_detail or has_architecture)
            if is_environmental_shot:
                is_panoramic = width > height * 2.5
                result.update({
                    'shot_type': ShotType.ENVIRONMENTAL,
                    'framing': 'panoramic' if is_panoramic else 'wide',
                    'visual_focus': 'landscape_environment' if has_architecture else 'natural_landscape'
                })
            # Otherwise use standard classification
            elif edge_density > 0.15 and contrast > 40:
                result['shot_type'] = ShotType.MEDIUM
                result['framing'] = 'medium_wide'
                result['visual_focus'] = 'center_high_detail'
            elif edge_density > 0.25 and contrast > 50:
                result['shot_type'] = ShotType.MEDIUM_CLOSE
                result['framing'] = 'medium_tight'
                result['visual_focus'] = 'center_very_high_detail'
            else:
                result['shot_type'] = ShotType.WIDE
                result['framing'] = 'wide'
                result['visual_focus'] = 'landscape' if is_landscape else 'portrait_wide'
        
        # Add composition notes based on shot type
        shot_notes = {
            ShotType.BIRDS_EYE: 'Direct overhead view, shows layout and spatial relationships from above',
            ShotType.AERIAL: 'High-angle view from aircraft/drone, shows large area from elevation',
            ShotType.ENVIRONMENTAL: 'Shows natural or architectural environment, emphasizes setting and atmosphere',
            ShotType.EXTREME_WIDE: 'Shows vast area with tiny subjects, emphasizes environment',
            ShotType.WIDE: 'Shows full scene with context, good for establishing action',
            ShotType.MEDIUM: 'Shows subject from waist up, good for interactions',
            ShotType.MEDIUM_CLOSE: 'Good for dialogue scenes, shows face and upper body',
            ShotType.CLOSE_UP: 'Intimate framing, focuses on facial features and expressions',
            ShotType.EXTREME_CLOSE_UP: 'Extreme close-up, focuses on specific details or emotions'
        }
        result['notes'] = shot_notes.get(result['shot_type'], 'Standard shot composition')
            
        return result

    def estimate_focal_length(self, frame: np.ndarray, known_width: float, known_distance: float) -> float:
        """
        Estimate focal length based on a known object
        
        Args:
            frame: Frame containing the object
            known_width: Known width of the object in real-world units
            known_distance: Distance to the object in real-world units
            
        Returns:
            Estimated focal length in pixels
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self.focal_length  # Return current if no contours found
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate focal length: (pixel width * known distance) / known width
        if w > 0 and known_width > 0:
            self.focal_length = (w * known_distance) / known_width
            
    def _classify_shot_type(self, frame: np.ndarray) -> Dict:
        """Classify shot type and composition based on object detection and image analysis"""
        # Initialize result with defaults
        result = {
            "shot_type": ShotType.WIDE.value,
            "camera_move": CameraMovement.STATIONARY.value,
            "framing": "unknown",
            "visual_focus": "unknown",
            "notes": "",
            "detected_objects": []
        }
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            gray = frame
            hsv = None
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_center = (x + w//2, y + h//2)
            frame_center = (frame.shape[1]//2, frame.shape[0]//2)
            
            # Calculate face position relative to frame
            x_ratio = face_center[0] / frame.shape[1]
            y_ratio = face_center[1] / frame.shape[0]
            
            # Determine framing based on face position
            if x_ratio < 0.4:
                result["framing"] = "Subject on left"
            elif x_ratio > 0.6:
                result["framing"] = "Subject on right"
            else:
                result["framing"] = "Subject centered"
                
            # Add vertical position info
            if y_ratio < 0.4:
                result["framing"] += ", high in frame"
            elif y_ratio > 0.6:
                result["framing"] += ", low in frame"
            else:
                result["framing"] += ", eye-level"
            
            # Determine shot type based on face size
            face_area = w * h
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area
            
            if face_ratio > 0.3:
                result["shot_type"] = ShotType.EXTREME_CLOSE_UP.value
                result["notes"] = "Extreme close-up on face"
            elif face_ratio > 0.2:
                result["shot_type"] = ShotType.CLOSE_UP.value
                result["notes"] = "Close-up on face"
            elif face_ratio > 0.1:
                result["shot_type"] = ShotType.MEDIUM_CLOSE.value
                result["notes"] = "Medium close-up"
            elif face_ratio > 0.05:
                result["shot_type"] = ShotType.MEDIUM.value
                result["notes"] = "Medium shot"
                
            # Analyze eye region for visual focus
            eye_region = gray[y:y+h//2, x:x+w]
            eye_brightness = np.mean(eye_region) / 255.0
            if eye_brightness > 0.7:
                result["visual_focus"] = "Eyes well-lit and prominent"
            else:
                result["visual_focus"] = "Eyes in shadow"
        
        # Edge detection for scene analysis
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # If no faces detected, use edge density to determine shot type
        if len(faces) == 0:
            if edge_density > 0.3:
                result["shot_type"] = ShotType.WIDE.value
                result["notes"] = "Complex scene with many edges"
            else:
                result["shot_type"] = ShotType.EXTREME_WIDE.value
                result["notes"] = "Wide open space with few features"
        
        # Color analysis for visual focus
        if hsv is not None:
            # Calculate color histogram
            hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_val = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Find dominant colors
            dominant_hue = np.argmax(hist_hue)
            dominant_sat = np.argmax(hist_sat)
            dominant_val = np.argmax(hist_val)
            
            if dominant_val < 50:
                result["visual_focus"] = "Low-key lighting"
            elif dominant_val > 200:
                result["visual_focus"] = "High-key lighting"
        
        return result
        return self.focal_length
