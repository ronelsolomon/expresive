"""
Advanced pose and motion analysis using MediaPipe and OpenCV
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseLandmark(Enum):
    """Pose landmark indices for MediaPipe Pose"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

@dataclass
class PoseAnalysisResult:
    """Container for pose analysis results"""
    pose_landmarks: List[Dict[str, float]]
    pose_world_landmarks: List[Dict[str, float]]
    pose_landmarks_world: List[Dict[str, float]]
    segmentation_mask: Optional[np.ndarray] = None
    pose_classification: Optional[str] = None
    movement_analysis: Optional[Dict[str, float]] = None

class PoseAnalyzer:
    """Advanced pose and motion analysis using MediaPipe"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Initialize the pose analyzer
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Model complexity (0=Light, 1=Full, 2=Heavy)
        """
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def analyze_pose_in_frame(self, frame: np.ndarray) -> Optional[PoseAnalysisResult]:
        """
        Analyze pose in the given frame and return analysis results.
        This is an alias for analyze_pose for backward compatibility.
        
        Args:
            frame: Input BGR image
            
        Returns:
            PoseAnalysisResult object with analysis results, or None if no pose is detected
        """
        return self.analyze_frame(frame)
        
    def analyze_frame(self, frame: np.ndarray) -> Optional[PoseAnalysisResult]:
        """
        Analyze pose in the given frame and return analysis results
        
        Args:
            frame: Input BGR image
            
        Returns:
            PoseAnalysisResult object with analysis results, or None if no pose is detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        # Extract landmarks
        landmarks = []
        world_landmarks = []
        
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
                
        if results.pose_world_landmarks:
            for landmark in results.pose_world_landmarks.landmark:
                world_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
        
        # Classify pose
        pose_class = self._classify_pose(landmarks)
        
        # Analyze movement
        movement = self._analyze_movement(landmarks)
        
        return PoseAnalysisResult(
            pose_landmarks=landmarks,
            pose_world_landmarks=world_landmarks,
            pose_landmarks_world=world_landmarks,  # For backward compatibility
            pose_classification=pose_class,
            movement_analysis=movement
        )
    
    def _classify_pose(self, landmarks: List[Dict[str, float]]) -> str:
        """Classify the detected pose"""
        if not landmarks:
            return "no_pose"
            
        # Simple pose classification based on keypoint positions
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]
        
        # Calculate shoulder and hip angles
        shoulder_slope = abs(left_shoulder['y'] - right_shoulder['y'])
        hip_slope = abs(left_hip['y'] - right_hip['y'])
        
        if shoulder_slope > 0.1 or hip_slope > 0.1:
            return "asymmetric_pose"
            
        return "neutral_pose"
    
    def _analyze_movement(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze movement characteristics"""
        if not landmarks or len(landmarks) < 33:  # MediaPipe Pose has 33 landmarks
            return {}
            
        # Calculate movement metrics
        movement_metrics = {
            'upper_body_movement': self._calculate_upper_body_movement(landmarks),
            'lower_body_movement': self._calculate_lower_body_movement(landmarks),
            'overall_activity': self._calculate_overall_activity(landmarks)
        }
        
        return movement_metrics
    
    def _calculate_upper_body_movement(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate upper body movement metric"""
        keypoints = [
            PoseLandmark.LEFT_SHOULDER.value,
            PoseLandmark.RIGHT_SHOULDER.value,
            PoseLandmark.LEFT_ELBOW.value,
            PoseLandmark.RIGHT_ELBOW.value,
            PoseLandmark.LEFT_WRIST.value,
            PoseLandmark.RIGHT_WRIST.value
        ]
        
        return self._calculate_joint_variability(landmarks, keypoints)
    
    def _calculate_lower_body_movement(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate lower body movement metric"""
        keypoints = [
            PoseLandmark.LEFT_HIP.value,
            PoseLandmark.RIGHT_HIP.value,
            PoseLandmark.LEFT_KNEE.value,
            PoseLandmark.RIGHT_KNEE.value,
            PoseLandmark.LEFT_ANKLE.value,
            PoseLandmark.RIGHT_ANKLE.value
        ]
        
        return self._calculate_joint_variability(landmarks, keypoints)
    
    def _calculate_overall_activity(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate overall body movement metric"""
        keypoints = list(range(33))  # All 33 MediaPipe Pose landmarks
        return self._calculate_joint_variability(landmarks, keypoints)
    
    def _calculate_joint_variability(self, 
                                   landmarks: List[Dict[str, float]], 
                                   keypoints: List[int]) -> float:
        """
        Calculate movement variability for specified keypoints
        
        Args:
            landmarks: List of landmark positions
            keypoints: Indices of landmarks to include in calculation
            
        Returns:
            Movement variability metric (0-1)
        """
        if not landmarks or not keypoints:
            return 0.0
            
        # Calculate mean position of keypoints
        mean_x = np.mean([landmarks[i]['x'] for i in keypoints])
        mean_y = np.mean([landmarks[i]['y'] for i in keypoints])
        
        # Calculate mean distance from center
        distances = []
        for i in keypoints:
            dx = landmarks[i]['x'] - mean_x
            dy = landmarks[i]['y'] - mean_y
            distances.append(np.sqrt(dx*dx + dy*dy))
            
        # Return normalized variability
        return float(np.mean(distances))
    
    def draw_landmarks(self, 
                      frame: np.ndarray, 
                      results: PoseAnalysisResult) -> np.ndarray:
        """
        Draw pose landmarks on the frame
        
        Args:
            frame: Input frame in BGR format
            results: PoseAnalysisResult with landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        if not results.pose_landmarks:
            return frame
            
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw the pose annotation on the frame
        mp_drawing.draw_landmarks(
            annotated_frame,
            self._convert_to_mp_landmarks(results.pose_landmarks),
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        # Add pose classification text
        if results.pose_classification:
            cv2.putText(annotated_frame, 
                       f"Pose: {results.pose_classification}",
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return annotated_frame
    
    def _convert_to_mp_landmarks(self, landmarks: List[Dict[str, float]]):
        """Convert our landmark format to MediaPipe format"""
        class Landmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
                
        class LandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks
                
        mp_landmarks = []
        for lm in landmarks:
            mp_landmarks.append(Landmark(
                x=lm['x'],
                y=lm['y'],
                z=lm.get('z', 0),
                visibility=lm.get('visibility', 1.0)
            ))
            
        return LandmarkList(mp_landmarks)
    
    def get_pose_activity(self, cap, frame_num):
        """Analyze pose activity for a specific frame
        
        Args:
            cap: VideoCapture object
            frame_num: Frame number to analyze
            
        Returns:
            Dictionary with activity metrics or None if no pose detected
        """
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return None
            
        # Analyze the frame
        result = self.analyze_frame(frame)
        if not result:
            return None
            
        # Return the movement metrics if available
        if result.movement_analysis:
            return {
                'overall_activity': result.movement_analysis.get('overall_activity', 0.0),
                'upper_body_activity': result.movement_analysis.get('upper_body_movement', 0.0),
                'lower_body_activity': result.movement_analysis.get('lower_body_movement', 0.0),
                'pose_class': result.pose_classification or 'unknown',
                'timestamp': frame_num / cap.get(cv2.CAP_PROP_FPS)
            }
        return None
        
    def __del__(self):
        """Release resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
