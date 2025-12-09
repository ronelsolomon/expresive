"""
models.py - Core Data Models for Storyboard System
Defines all data structures used throughout the pipeline
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json


class ShotType(Enum):
    """Standard cinematographic shot types"""
    BIRDS_EYE = "Birds Eye"
    EARTH_EYE =  "Earth Eye"
    ENVIRONMENTAL = "Environmental"
    WIDE = "Wide"
    MEDIUM = "Medium"
    CLOSE_UP = "Close-Up"
    ECU = "ECU"  # Extreme Close-Up
    OTS = "Over-The-Shoulder"  # Over the shoulder
    TWO_SHOT = "Two-Shot"


class CameraMove(Enum):
    """Camera movement types"""
    STILL = "Still"
    PUSH = "Push"  # Dolly in
    PULL = "Pull"  # Dolly out
    PAN_LEFT = "Pan Left"
    PAN_RIGHT = "Pan Right"
    TILT_UP = "Tilt Up"
    TILT_DOWN = "Tilt Down"
    TRUCK_LEFT = "Truck Left"  # Lateral movement
    TRUCK_RIGHT = "Truck Right"
    DOLLY = "Dolly"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    HANDHELD = "Handheld"
    CRANE = "Crane"


class TransitionType(Enum):
    """Scene transition types"""
    CUT = "cut"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    DISSOLVE = "dissolve"
    CROSS_DISSOLVE = "cross_dissolve"
    WHIP_PAN = "whip_pan"
    MATCH_CUT = "match_cut"
    L_CUT = "l_cut"  # Audio leads video
    J_CUT = "j_cut"  # Video leads audio
    CROSS_ZOOM = "cross_zoom"
    WIPE = "wipe"
    IRIS = "iris"


class TimeOfDay(Enum):
    """Time of day classification"""
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"
    UNKNOWN = "unknown"


class Mood(Enum):
    """Emotional mood classifications"""
    CALM = "calm"
    TENSE = "tense"
    JOYFUL = "joyful"
    MELANCHOLIC = "melancholic"
    ENERGETIC = "energetic"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"
    DRAMATIC = "dramatic"
    NEUTRAL = "neutral"


@dataclass
class Detection:
    """Object detection result"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    def area(self) -> int:
        """Calculate bounding box area"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class Transition:
    """Transition between scenes"""
    type: str
    from_scene_id: Optional[int] = None
    to_scene_id: Optional[int] = None
    description: str = ""
    duration: float = 0.0


@dataclass
class Shot:
    """Individual shot within a scene"""
    shot_id: int
    shot_type: str
    camera_move: str
    framing: str
    visual_focus: str
    notes: str = ""
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CameraAnalysis:
    """Detailed camera movement analysis"""
    movement: str
    shot_type: str
    stability: float  # 0-1, higher is more stable
    speed: float  # Relative speed of movement
    focal_length_estimate: str  # "wide", "normal", "telephoto"


@dataclass
class SceneMetadata:
    """Rich metadata for a scene"""
    duration: float
    description: str
    key_objects: List[Dict]
    time_of_day: str
    environment: str  # "indoor", "outdoor", "ambiguous"
    mood: str
    camera_analysis: Dict
    color_palette: Optional[List[str]] = None
    motion_intensity: float = 0.0
    audio_features: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Scene:
    """Complete scene with metadata and shots"""
    scene_id: int
    start_time: float
    end_time: float
    transition_in: Optional[Dict] = None
    transition_out: Optional[Dict] = None
    scene_metadata: Optional[Dict] = None
    shots: List[Dict] = field(default_factory=list)
    
    def duration(self) -> float:
        """Get scene duration"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def add_shot(self, shot: Shot, start_time: float, end_time: float):
        """Add a shot to the scene"""
        self.shots.append({
            'shot': shot.to_dict(),
            'start_time': start_time,
            'end_time': end_time
        })


@dataclass
class Storyboard:
    """Complete storyboard representation"""
    scenes: List[Scene]
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'scenes': [scene.to_dict() for scene in self.scenes],
            'metadata': self.metadata
        }
    
    def to_json(self, filepath: str, indent: int = 2):
        """Export to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Storyboard':
        """Load from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenes = [Scene(**scene_data) for scene_data in data['scenes']]
        return cls(scenes=scenes, metadata=data['metadata'])


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder for Enum types"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
