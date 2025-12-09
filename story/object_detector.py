import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime

class SceneType(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    URBAN = "urban"
    NATURE = "nature"
    INTERIOR = "interior"
    UNKNOWN = "unknown"

class TimeOfDay(Enum):
    DAY = "day"
    NIGHT = "night"
    SUNRISE_SUNSET = "sunrise/sunset"
    UNKNOWN = "unknown"

class ActionType(Enum):
    WALKING = "walking"
    RUNNING = "running"
    SITTING = "sitting"
    STANDING = "standing"
    DRIVING = "driving"
    TALKING = "talking"
    UNKNOWN = "unknown"

@dataclass
class SceneAnalysis:
    scene_type: SceneType = SceneType.UNKNOWN
    time_of_day: TimeOfDay = TimeOfDay.UNKNOWN
    location_type: str = "unknown"
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class ActionAnalysis:
    action_type: ActionType = ActionType.UNKNOWN
    subject: str = "unknown"
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    track_id: Optional[int] = None
    velocity: Tuple[float, float] = (0.0, 0.0)  # x, y velocity in pixels per second

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov8x.pt', device: str = None):
        """
        Initialize YOLO object detector with enhanced scene and action recognition
        
        Args:
            model_name: YOLO model name or path (default: 'yolov8x.pt' for best accuracy)
            device: 'cuda' for GPU or 'cpu' for CPU (default: auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing YOLO model: {model_name}...")
        
        try:
            # Use Ultralytics YOLO interface
            from ultralytics import YOLO
            
            # Try to load the model with error handling
            try:
                self.model = YOLO(model_name)
                self.model.to(self.device)
                # Set default confidence and IoU thresholds
                self.model.conf = 0.4  # Confidence threshold
                self.model.iou = 0.45  # NMS IoU threshold
                print(f"Successfully loaded YOLO model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                print("Falling back to default YOLOv8s model...")
                self.model = YOLO('yolov8s.pt')
                self.model.to(self.device)
                self.model.conf = 0.4
                self.model.iou = 0.45
        except ImportError:
            raise ImportError("Ultralytics YOLO is required. Install with: pip install ultralytics")
        
        # Enhanced scene classification parameters
        self.scene_objects = {
            SceneType.INDOOR: ['chair', 'sofa', 'tv', 'bed', 'table', 'laptop', 'book', 'keyboard', 
                              'mouse', 'monitor', 'refrigerator', 'oven', 'sink', 'toilet', 'sink'],
            SceneType.OUTDOOR: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign'],
            SceneType.URBAN: ['car', 'truck', 'bus', 'traffic light', 'street sign', 'building', 'bench',
                            'parking meter', 'fire hydrant', 'stop sign', 'person', 'bicycle', 'motorcycle'],
            SceneType.NATURE: ['tree', 'grass', 'mountain', 'sheep', 'cow', 'horse', 'dog', 'bird',
                             'bear', 'zebra', 'giraffe', 'elephant', 'potted plant', 'bench', 'person'],
            SceneType.INTERIOR: ['chair', 'sofa', 'tv', 'bed', 'dining table', 'laptop', 'book', 'vase',
                               'clock', 'sink', 'refrigerator', 'oven', 'microwave', 'toilet', 'sink']
        }
        
        # Action recognition parameters
        self.tracked_objects: Dict[int, List[Tuple[float, Tuple[float, float]]]] = {}  # track_id -> [(timestamp, (x, y))]
        self.min_track_length = 5  # Minimum frames to consider for action recognition
        self.action_history: Dict[int, List[ActionAnalysis]] = {}
        
        # Time of day parameters
        self.brightness_threshold = 0.3  # Threshold for day/night classification
        self.color_temp_thresholds = {
            'sunrise_sunset': (2000, 3500),  # Color temperature range in Kelvin
            'day': (5000, 7000),
            'night': (1000, 2000)
        }
        self.classes = self.model.names
        print(f"Object Detector initialized with {model_name} on {self.device.upper()}")
    
    def _detect_time_of_day(self, frame: np.ndarray) -> Tuple[TimeOfDay, float]:
        """
        Detect time of day from frame with confidence score
        
        Returns:
            Tuple of (time_of_day, confidence) where confidence is between 0 and 1
        """
        if frame is None or frame.size == 0:
            return TimeOfDay.UNKNOWN, 0.0
            
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate brightness statistics
            v_channel = hsv[:,:,2]
            avg_brightness = np.mean(v_channel) / 255.0
            brightness_std = np.std(v_channel) / 255.0
            
            # Calculate color temperature (simplified)
            b, g, r = cv2.split(frame.astype('float'))
            r_avg = np.mean(r)
            b_avg = np.mean(b)
            
            # Avoid division by zero
            if b_avg > 10:
                temp_score = r_avg / b_avg
            else:
                temp_score = 1.0
            
            # Calculate confidence based on brightness distribution
            confidence = min(1.0, brightness_std * 3)  # Higher std -> more confident
            
            # Classify based on brightness and color temperature
            if avg_brightness < 0.25:
                return TimeOfDay.NIGHT, confidence
            elif 0.25 <= avg_brightness < 0.5:
                if temp_score > 1.2:  # Warmer colors
                    return TimeOfDay.SUNRISE_SUNSET, confidence
                return TimeOfDay.NIGHT, confidence
            elif 0.5 <= avg_brightness < 0.7:
                if temp_score > 1.1:  # Slightly warm
                    return TimeOfDay.SUNRISE_SUNSET, confidence
                return TimeOfDay.DAY, confidence
            else:
                return TimeOfDay.DAY, confidence
                
        except Exception as e:
            print(f"Error in time of day detection: {str(e)}")
            return TimeOfDay.UNKNOWN, 0.0
    
    def _classify_scene(self, detections: List[Dict]) -> Tuple[SceneType, float]:
        """
        Classify scene type based on detected objects with confidence score
        
        Returns:
            Tuple of (scene_type, confidence) where confidence is between 0 and 1
        """
        if not detections:
            return SceneType.UNKNOWN, 0.0
            
        # Count objects by scene type with confidence weighting
        scene_scores = {st: 0.0 for st in SceneType}
        total_confidence = 0.0
        
        for det in detections:
            obj_class = det.get('class', '').lower()
            confidence = det.get('confidence', 0.5)  # Default to 0.5 if not provided
            
            # Add to all matching scene types
            for scene_type, objects in self.scene_objects.items():
                if obj_class in objects:
                    scene_scores[scene_type] += confidence
                    total_confidence += confidence
        
        if total_confidence == 0:
            return SceneType.UNKNOWN, 0.0
            
        # Normalize scores
        for scene_type in scene_scores:
            scene_scores[scene_type] /= total_confidence
        
        # Get scene with highest score
        best_scene, best_score = max(scene_scores.items(), key=lambda x: x[1])
        
        # Apply threshold
        if best_score < 0.3:  # Low confidence threshold
            return SceneType.UNKNOWN, best_score
            
        return best_scene, best_score
    
    def _analyze_actions(self, detections: List[Detection], timestamp: float) -> List[ActionAnalysis]:
        """
        Analyze actions based on object tracking
        
        Args:
            detections: Current frame detections
            timestamp: Current timestamp in seconds
            
        Returns:
            List of detected actions
        """
        actions = []
        current_frame_objects = {}
        
        # Update tracks and detect actions
        for det in detections:
            if det.track_id is None:
                det.track_id = len(self.tracked_objects) + 1
                self.tracked_objects[det.track_id] = []
            
            # Calculate center of bbox
            x1, y1, x2, y2 = det.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Update track
            track = self.tracked_objects[det.track_id]
            track.append((timestamp, center))
            
            # Keep only recent positions (last 2 seconds at 30fps = 60 frames)
            if len(track) > 60:
                track.pop(0)
            
            # If we have enough history, analyze movement
            if len(track) >= self.min_track_length:
                speed = self._calculate_speed(track)
                action = self._classify_action(det.class_name, speed)
                if action:
                    actions.append(action)
            
            current_frame_objects[det.track_id] = det
        
        # Clean up old tracks
        self._cleanup_tracks(current_frame_objects.keys())
        
        return actions
    
    def _calculate_speed(self, track: List[Tuple[float, Tuple[float, float]]]) -> float:
        """Calculate speed in pixels per second"""
        if len(track) < 2:
            return 0.0
            
        # Get first and last position
        t1, (x1, y1) = track[0]
        t2, (x2, y2) = track[-1]
        
        if t2 <= t1:
            return 0.0
            
        # Calculate speed in pixels per second
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        time_elapsed = t2 - t1
        return distance / time_elapsed if time_elapsed > 0 else 0.0
    
    def _classify_action(self, class_name: str, speed: float) -> Optional[ActionAnalysis]:
        """Classify action based on object type and movement"""
        if 'person' in class_name.lower():
            if speed > 5.0:
                return ActionAnalysis(ActionType.RUNNING, 'person', min(1.0, speed / 20))
            elif speed > 1.0:
                return ActionAnalysis(ActionType.WALKING, 'person', min(0.9, speed / 10))
            else:
                return ActionAnalysis(ActionType.STANDING, 'person', 0.8)
        elif 'car' in class_name.lower() or 'truck' in class_name.lower():
            if speed > 2.0:
                return ActionAnalysis(ActionType.DRIVING, class_name, min(1.0, speed / 50))
        return None
    
    def _get_dominant_colors(self, frame: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the frame using k-means clustering"""
        if frame is None or frame.size == 0:
            return []
            
        try:
            # Resize for faster processing
            pixels = cv2.resize(frame, (100, 100)).reshape(-1, 3)
            
            # Convert to float32 for k-means
            pixels = np.float32(pixels)
            
            # Define criteria and apply k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert back to uint8
            centers = np.uint8(centers)
            
            # Convert BGR to RGB
            return [tuple(center.tolist()[::-1]) for center in centers]
            
        except Exception as e:
            print(f"Error in color extraction: {str(e)}")
            return []
    
    def _cleanup_tracks(self, current_ids: Set[int]) -> None:
        """Remove tracks that are no longer active"""
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in current_ids:
                del self.tracked_objects[track_id]
    
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Detect objects and analyze scene and actions in a frame
        
        Args:
            frame: Input frame in BGR format
            timestamp: Current timestamp in seconds
            
        Returns:
            Dictionary containing:
            - detections: List of Detection objects
            - scene_analysis: SceneAnalysis object
            - actions: List of ActionAnalysis objects
            - time_of_day: Estimated TimeOfDay
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run object detection with proper parameters
            with torch.no_grad():
                # Use the model's predict method with the correct parameters
                results = self.model.predict(
                    source=frame_rgb,
                    conf=0.4,  # Confidence threshold
                    iou=0.45,  # NMS IoU threshold
                    verbose=False  # Disable verbose output
                )
            
            # Process detections using the dedicated method
            detections = self._process_detections(results)
            
            # Get scene analysis
            scene_analysis = self.analyze_scene(frame)
            
            # Analyze actions
            actions = self._analyze_actions(detections, timestamp)
            
            # Detect time of day
            time_of_day_result = self._detect_time_of_day(frame)
            time_of_day = time_of_day_result[0] if isinstance(time_of_day_result, tuple) else TimeOfDay.UNKNOWN
            
            # Get scene type and confidence
            scene_type, scene_confidence = self._classify_scene([{'class': d.class_name, 'confidence': d.confidence} for d in detections])
            
            # Create scene analysis with all available information
            scene_analysis = SceneAnalysis(
                scene_type=scene_type,
                time_of_day=time_of_day,
                confidence=scene_confidence,
                dominant_colors=self._get_dominant_colors(frame)
            )
            
            return {
                'detections': detections,
                'scene_analysis': scene_analysis,
                'actions': actions,
                'time_of_day': time_of_day.value if hasattr(time_of_day, 'value') else TimeOfDay.UNKNOWN.value,
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"Error in detect method: {str(e)}")
            # Return empty results in case of error
            return {
                'detections': [],
                'scene_analysis': SceneAnalysis(),
                'actions': [],
                'time_of_day': TimeOfDay.UNKNOWN.value,
                'timestamp': timestamp
            }
    
    def analyze_scene(self, frame: np.ndarray) -> SceneAnalysis:
        """
        Analyze a frame and return detailed scene information
        
        Args:
            frame: Input BGR image
            
        Returns:
            SceneAnalysis object with detailed scene information
        """
        if frame is None or frame.size == 0:
            return SceneAnalysis()
            
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use predict with proper parameters instead of direct model call
            results = self.model.predict(
                source=frame_rgb,
                conf=0.4,  # Confidence threshold
                iou=0.45,  # NMS IoU threshold
                verbose=False  # Disable verbose output
            )
            
            # Get detailed detections
            detections = self._process_detections(results)
            
            # Classify scene with more detailed analysis
            scene_type, scene_confidence = self._classify_scene(detections)
            time_of_day, time_confidence = self._detect_time_of_day(frame)
            
            # Get dominant colors
            dominant_colors = self._get_dominant_colors(frame)
            
            # Get environment type based on detections
            environment = self._get_environment_type(detections, frame)
            
            return SceneAnalysis(
                scene_type=scene_type,
                time_of_day=time_of_day,
                location_type=environment,
                confidence=min(scene_confidence, time_confidence),
                dominant_colors=dominant_colors[:3]  # Top 3 dominant colors
            )
            
        except Exception as e:
            print(f"Error in scene analysis: {str(e)}")
            return SceneAnalysis()
    
    def _get_environment_type(self, detections: List[Dict], frame: np.ndarray) -> str:
        """Determine the type of environment based on detections and frame analysis"""
        if not detections:
            return "unknown"
            
        # Count different types of objects
        obj_counts = {}
        for det in detections:
            cls_name = det.get('class', '').lower()
            obj_counts[cls_name] = obj_counts.get(cls_name, 0) + 1
        
        # Check for indoor indicators
        indoor_objs = sum(obj_counts.get(obj, 0) for obj in self.scene_objects[SceneType.INDOOR])
        outdoor_objs = sum(obj_counts.get(obj, 0) for obj in self.scene_objects[SceneType.OUTDOOR])
        
        if indoor_objs > outdoor_objs + 2:  # More indoor objects
            return "indoor"
        elif outdoor_objs > indoor_objs + 2:  # More outdoor objects
            return "outdoor"
            
        # If not clear, analyze the frame
        return self._analyze_frame_environment(frame)
    
    def _analyze_frame_environment(self, frame: np.ndarray) -> str:
        """Analyze frame characteristics to determine environment"""
        if frame is None or frame.size == 0:
            return "unknown"
            
        # Convert to grayscale and calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # Calculate colorfulness (variance of saturation and value in HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1].std()
        v = hsv[:,:,2].std()
        colorfulness = (s + v) / 2.0
        
        # Simple heuristic for indoor/outdoor
        if brightness > 0.6 and colorfulness > 20:
            return "outdoor"
        return "indoor"
    
    def _process_detections(self, results):
        """Process YOLO detection results into a list of Detection objects"""
        detections = []
        
        # Handle case where results is a list of Results objects or a single Results object
        results_list = results if isinstance(results, (list, tuple)) else [results]
        
        for result in results_list:
            if hasattr(result, 'boxes') and result.boxes is not None:  # YOLOv8 format
                boxes = result.boxes
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box[:4].tolist())
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    class_name = self.model.names.get(cls, f"class_{cls}")
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        class_id=cls
                    ))
            elif hasattr(result, 'xyxy') and result.xyxy[0] is not None:  # YOLOv5 format
                for det in result.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                    class_id = int(cls)
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=class_id
                    ))
        
        return detections
    
    def get_most_common_objects(self, frames: List[np.ndarray], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get most common objects across multiple frames
        
        Args:
            frames: List of frames to analyze
            top_k: Number of top objects to return
            
        Returns:
            List of dicts with class name and count, sorted by frequency
        """
        class_counts = {}
        
        for frame in frames:
            detections = self.detect(frame)
            for det in detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                
        # Sort by count in descending order
        sorted_objects = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'class': cls, 'count': count} for cls, count in sorted_objects[:top_k]]
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of Detection objects
            
        Returns:
            Frame with detection boxes drawn
        """
        output = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            label = f"{det.class_name} {det.confidence:.2f}"
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output, (x, y - 20), (x + label_w, y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(output, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
        return output
