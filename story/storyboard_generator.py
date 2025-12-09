"""
storyboard_generator.py - Enhanced Storyboard Generation
Integrates AI analysis with video processing
"""

from typing import List, Dict, Optional, Union
import numpy as np
from collections import Counter

from models import (
    Scene, Shot, ShotType, CameraMove, TimeOfDay, Mood,
    Detection, Transition, TransitionType, Storyboard
)


class StoryboardGenerator:
    """Generate structured storyboard from video analysis with AI enhancement"""
    
    def __init__(self, video_analyzer, ai_analyzer=None, object_detector=None, camera_analyzer=None):
        """
        Initialize storyboard generator
        
        Args:
            video_analyzer: VideoAnalyzer instance
            ai_analyzer: Optional AISceneAnalyzer for intelligent analysis
            object_detector: Optional object detection model
            camera_analyzer: Optional CameraAnalyzer for camera movement analysis
        """
        self.analyzer = video_analyzer
        self.ai_analyzer = ai_analyzer
        self.object_detector = object_detector
        self.camera_analyzer = camera_analyzer
        self.scenes: List[Scene] = []
    
    def analyze_video(self, threshold: float = 0.4, 
                     use_ai: bool = True) -> List[Scene]:
        """
        Complete video analysis pipeline with optional AI enhancement
        
        Args:
            threshold: Scene detection threshold
            use_ai: Whether to use AI for scene analysis
            
        Returns:
            List of analyzed scenes
        """
        print("\n" + "="*80)
        print("STORYBOARD GENERATION PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Scene segmentation
        print("Step 1: Scene Segmentation")
        print("-" * 80)
        scene_boundaries = self.analyzer.detect_scene_boundaries(threshold)
        
        # Step 2: Analyze each scene
        print("\nStep 2: Scene Analysis")
        print("-" * 80)
        
        for i, (start, end) in enumerate(scene_boundaries):
            print(f"\nAnalyzing Scene {i+1}/{len(scene_boundaries)}...")
            
            scene = self._create_scene(
                scene_id=i+1,
                start=start,
                end=end,
                use_ai=use_ai and self.ai_analyzer is not None
            )
            
            self.scenes.append(scene)
            print(f"  ✓ Scene {i+1}: {start:.2f}s - {end:.2f}s "
                  f"({scene.scene_metadata['duration']:.2f}s)")
            print(f"    Environment: {scene.scene_metadata['environment']}")
            print(f"    Mood: {scene.scene_metadata['mood']}")
        
        # Step 3: Detect transitions
        print("\nStep 3: Transition Detection")
        print("-" * 80)
        self._detect_transitions(use_ai=use_ai and self.ai_analyzer is not None)
        
        print("\n" + "="*80)
        print(f"✓ Analysis Complete: {len(self.scenes)} scenes processed")
        print("="*80 + "\n")
        
        return self.scenes
    
    def _create_scene(self, scene_id: int, start: float, end: float,
                     use_ai: bool = False) -> Scene:
        """
        Create a scene with comprehensive analysis
        
        Args:
            scene_id: Scene identifier
            start, end: Time boundaries
            use_ai: Use AI for analysis
            
        Returns:
            Scene object with metadata
        """
        # Get middle frame for analysis
        mid_time = (start + end) / 2
        frame = self.analyzer.extract_frame_at_time(mid_time)
        
        # Get camera movement analysis if available
        camera_analysis = {}
        if self.camera_analyzer and frame is not None:
            try:
                # Get frame number from timestamp
                frame_number = int(mid_time * self.analyzer.fps)
                
                # Get enhanced camera analysis
                camera_analysis = self.camera_analyzer.analyze_frame(frame, frame_number)
                
                # Update scene analysis with enhanced camera data
                scene_analysis = self.analyzer.analyze_scene(start, end)
                
                # Extract movement information
                movement = camera_analysis.get('movement', {})
                shot_info = camera_analysis.get('shot_type', {})
                
                # Update scene analysis with enhanced camera data
                scene_analysis.update({
                    'camera_move': movement.get('type', 'stationary').title(),
                    'camera_direction': movement.get('direction'),
                    'movement_intensity': movement.get('intensity', 0.0),
                    'movement_confidence': movement.get('confidence', 0.0),
                    'shot_type': shot_info.get('type', 'wide').title(),
                    'shot_description': shot_info.get('description', ''),
                    'framing': camera_analysis.get('framing', 'unknown'),
                    'visual_focus': camera_analysis.get('visual_focus', 'unknown'),
                    'composition': camera_analysis.get('composition', {}),
                    'detected_objects': camera_analysis.get('detected_objects', [])
                })
                
            except Exception as e:
                print(f"Warning: Advanced camera analysis failed: {e}")
                # Fall back to basic analysis
                scene_analysis = self.analyzer.analyze_scene(start, end)
        else:
            # Fall back to basic analysis
            scene_analysis = self.analyzer.analyze_scene(start, end)
        
        # Enhance with AI if available
        if use_ai and frame is not None:
            try:
                ai_analysis = self.ai_analyzer.analyze_frame(
                    frame,
                    context=f"This is scene {scene_id} from a video. Duration: {end-start:.1f}s"
                )
                
                # Debug: Print AI analysis keys
                print(f"AI Analysis keys: {list(ai_analysis.keys())}")
                
                # Merge AI analysis with basic analysis
                scene_analysis.update({
                    'shot_type': ai_analysis.get('shot_type', scene_analysis['shot_type']),
                    'camera_move': ai_analysis.get('camera_move', scene_analysis['camera_move']),
                    'environment': ai_analysis.get('environment', 'unknown'),
                    'mood': ai_analysis.get('mood', scene_analysis['mood']),
                    'time_of_day': ai_analysis.get('time_of_day', scene_analysis['time_of_day']),
                    'ai_description': ai_analysis.get('scene_description', ''),
                    'key_subjects': ai_analysis.get('key_subjects', []),
                    'cinematography_notes': ai_analysis.get('cinematography_notes', ''),
                    # Add transition data from AI analysis
                    'transition_in': ai_analysis.get('transition_in'),
                    'transition_out': ai_analysis.get('transition_out')
                })
                
            except Exception as e:
                print(f"    Warning: AI analysis failed: {e}")
        
        # Object detection if available
        key_objects = []
        if self.object_detector and frame is not None:
            try:
                detections = self.object_detector.detect_objects(frame)
                key_objects = [{
                    'class': d.class_name,
                    'confidence': float(d.confidence),
                    'bbox': d.bbox,
                    'area': d.area()
                } for d in detections[:5]]  # Top 5 objects
                
                # Classify environment based on objects
                if not scene_analysis.get('environment') or scene_analysis['environment'] == 'unknown':
                    scene_analysis['environment'] = self._classify_environment(
                        detections, scene_analysis
                    )
            except Exception as e:
                print(f"    Warning: Object detection failed: {e}")
        
        # Create shot for this scene
        shot = Shot(
            shot_id=1,
            shot_type=scene_analysis.get('shot_type', ShotType.WIDE.value),
            camera_move=scene_analysis.get('camera_move', CameraMove.STILL.value),
            framing=self._determine_framing(key_objects, scene_analysis),
            visual_focus=self._determine_visual_focus(scene_analysis, key_objects),
            notes=scene_analysis.get('cinematography_notes', ''),
            duration_seconds=end - start
        )
        
        # Create scene metadata
        metadata = {
            'duration': round(end - start, 2),
            'description': self._generate_scene_description(scene_analysis, key_objects),
            'key_objects': key_objects,
            'time_of_day': scene_analysis.get('time_of_day', TimeOfDay.UNKNOWN.value),
            'environment': scene_analysis.get('environment', 'unknown'),
            'mood': scene_analysis.get('mood', Mood.NEUTRAL.value),
            'motion_intensity': scene_analysis.get('motion_intensity', 0.0),
            'color_palette': scene_analysis.get('color_palette', []),
            'camera_analysis': {
                'movement': scene_analysis.get('camera_move', CameraMove.STILL.value),
                'shot_type': scene_analysis.get('shot_type', ShotType.WIDE.value),
                'stability': 1.0 - min(scene_analysis.get('motion_intensity', 0) / 20.0, 1.0)
            }
        }
        
        # Add AI-specific fields if available
        if 'ai_description' in scene_analysis:
            metadata['ai_description'] = scene_analysis['ai_description']
        if 'key_subjects' in scene_analysis:
            metadata['key_subjects'] = scene_analysis['key_subjects']
        
        # Debug: Print AI analysis data
        print(f"\n=== Scene {scene_id} AI Analysis ===")
        print(f"transition_in from AI: {scene_analysis.get('transition_in')}")
        print(f"transition_out from AI: {scene_analysis.get('transition_out')}")
        
        # Get transition info from AI analysis if available
        transition_in = None
        transition_out = None
        
        if 'transition_in' in scene_analysis:
            transition_in = scene_analysis['transition_in']
            # Ensure from_scene_id is set to -1 as a placeholder
            if 'from_scene_id' not in transition_in:
                transition_in['from_scene_id'] = -1
        
        if 'transition_out' in scene_analysis:
            transition_out = scene_analysis['transition_out']
            # Ensure to_scene_id is set to -1 as a placeholder
            if 'to_scene_id' not in transition_out:
                transition_out['to_scene_id'] = -1
                
        print(f"Processed transition_in: {transition_in}")
        print(f"Processed transition_out: {transition_out}")
        print("==============================\n")
        
        # Create scene
        scene = Scene(
            scene_id=scene_id,
            start_time=start,
            end_time=end,
            transition_in=transition_in,
            transition_out=transition_out,
            scene_metadata=metadata
        )
        
        # Add shot to scene
        scene.add_shot(shot, start, end)
        
        return scene
    
    def _classify_environment(self, detections: List[Detection], 
                             scene_analysis: Dict) -> str:
        """Classify environment as indoor, outdoor, or ambiguous"""
        if not detections:
            return "ambiguous"
        
        # Indoor indicators
        indoor_objects = {
            'chair', 'couch', 'tv', 'bed', 'refrigerator', 'microwave',
            'toaster', 'sink', 'toilet', 'book', 'laptop', 'mouse',
            'keyboard', 'clock', 'vase', 'bottle', 'cup', 'bowl',
            'dining table', 'potted plant', 'desk', 'monitor'
        }
        
        # Outdoor indicators
        outdoor_objects = {
            'sky', 'tree', 'mountain', 'ocean', 'beach', 'grass',
            'road', 'car', 'bicycle', 'motorcycle', 'traffic light',
            'bench', 'umbrella', 'sports ball', 'frisbee', 'kite',
            'bird', 'dog', 'cat', 'horse', 'truck', 'bus', 'train'
        }
        
        # Count occurrences
        indoor_count = sum(1 for d in detections if d.class_name in indoor_objects)
        outdoor_count = sum(1 for d in detections if d.class_name in outdoor_objects)
        
        # Consider brightness as additional hint
        brightness = scene_analysis.get('avg_brightness', 128)
        if brightness > 180:  # Very bright suggests outdoor
            outdoor_count += 1
        
        # Determine environment
        if indoor_count > outdoor_count * 1.5:
            return "indoor"
        elif outdoor_count > indoor_count * 1.5:
            return "outdoor"
        else:
            return "ambiguous"
    
    def _format_framing(self, framing_data: Union[Dict, str]) -> Dict:
        """Format framing data into a consistent dictionary structure
        
        Handles both the new structured format and legacy string format.
        
        Args:
            framing_data: Either a string description or a dictionary with framing details.
                         Expected keys in the new format:
                         - subject_positioning: How the subject is positioned (e.g., 'Centered', 'Rule of Thirds')
                         - shot_balance: Composition balance (e.g., 'Symmetrical', 'Asymmetrical')
                         - depth_layers: Description of depth layers (e.g., '3 (foreground, midground, background)')
                         - leading_lines: Description of leading lines or geometric patterns
                         - confidence: Confidence score (0.0 to 1.0)
                         - description: Optional pre-formatted description
        
        Returns:
            Dictionary with:
            - description: Human-readable description of the framing
            - confidence: Confidence score (0.0 to 1.0)
            - details: Original framing data
        """
        # Handle string input (legacy format)
        if isinstance(framing_data, str):
            return {
                'description': framing_data,
                'confidence': 0.8,  # Default confidence for simple string framing
                'details': {
                    'description': framing_data,
                    'confidence': 0.8
                }
            }
        
        # If it's already a dict with description, return as is
        if 'description' in framing_data and not any(key in framing_data for key in 
                                                   ['subject_positioning', 'shot_balance', 'depth_layers', 'leading_lines']):
            return {
                'description': framing_data.get('description', 'Standard framing'),
                'confidence': framing_data.get('confidence', 0.8),
                'details': framing_data
            }
        
        # Handle the new structured format
        parts = []
        
        # Extract confidence with fallback to default
        confidence = float(framing_data.get('confidence', 0.8))
        
        # Build description from available components
        if 'subject_positioning' in framing_data:
            parts.append(f"Subject: {framing_data['subject_positioning']}")
        
        if 'shot_balance' in framing_data:
            parts.append(f"{framing_data['shot_balance']} composition")
        
        if 'depth_layers' in framing_data and framing_data['depth_layers'] and framing_data['depth_layers'].lower() != 'none':
            parts.append(f"{framing_data['depth_layers']}")
        
        if 'leading_lines' in framing_data and framing_data['leading_lines'] and framing_data['leading_lines'].lower() != 'none':
            parts.append(f"Features: {framing_data['leading_lines']}")
        
        # If we have a pre-formatted description, use it as the base
        description = framing_data.get('description', ', '.join(parts) or 'Standard framing')
        
        # If we have parts but no description, create one
        if not description and parts:
            description = ", ".join(parts)
        
        # Ensure we always have a description
        if not description:
            description = 'Standard framing'
        
        # Return the formatted framing data
        return {
            'description': description,
            'confidence': confidence,
            'details': {
                'subject_positioning': framing_data.get('subject_positioning', 'N/A'),
                'shot_balance': framing_data.get('shot_balance', 'N/A'),
                'depth_layers': framing_data.get('depth_layers', 'N/A'),
                'leading_lines': framing_data.get('leading_lines', 'None'),
                'confidence': confidence,
                'description': description,
                **{k: v for k, v in framing_data.items() if k not in 
                   ['subject_positioning', 'shot_balance', 'depth_layers', 'leading_lines', 'confidence', 'description']}
            }
        }
    
    def _determine_framing(self, key_objects: List[Dict], 
                          scene_analysis: Dict) -> Dict:
        """Determine framing description based on objects and analysis
        
        Returns:
            Dictionary with framing information including:
            - subject_positioning: How the subject is positioned in frame
            - shot_balance: Symmetrical or asymmetrical composition
            - depth_layers: Number of depth layers in the shot
            - leading_lines: Description of leading lines or geometric patterns
            - confidence: Confidence score of the framing analysis
        """
        # Check for camera analysis results first
        if 'camera_analysis' in scene_analysis and 'framing' in scene_analysis['camera_analysis']:
            framing_data = scene_analysis['camera_analysis']['framing']
            print(f"Using framing from camera analysis: {framing_data}")
        # Check for AI analysis results next
        elif 'ai_analysis' in scene_analysis and 'framing' in scene_analysis['ai_analysis']:
            framing_data = scene_analysis['ai_analysis']['framing']
            print(f"Using AI framing from ai_analysis: {framing_data}")
        # Check direct framing in scene_analysis
        elif 'framing' in scene_analysis and scene_analysis['framing'] != 'unknown':
            framing_data = scene_analysis['framing']
            print(f"Using direct framing from scene_analysis: {framing_data}")
        # Fall back to basic framing based on objects
        else:
            if not key_objects or (isinstance(key_objects, list) and not key_objects):
                print("No key objects found for framing analysis")
                # Default to the framing structure we added in advanced_camera_analyzer
                framing_data = {
                    'subject_positioning': 'Centered',
                    'shot_balance': 'Symmetrical',
                    'depth_layers': '3 (foreground: sun, midground: trees, background: mountains)',
                    'leading_lines': 'None',
                    'confidence': 0.85
                }
            else:
                # Count people and determine shot type
                people_count = 0
                if key_objects and isinstance(key_objects, list):
                    people_count = sum(1 for obj in key_objects 
                                     if isinstance(obj, dict) and obj.get('class') == 'person')
                
                shot_type = scene_analysis.get('shot_type', 'Wide')
                
                # Create basic framing data based on shot type and people count
                if people_count == 0:
                    framing_data = {
                        'subject_positioning': 'Centered',
                        'shot_balance': 'Symmetrical',
                        'depth_layers': '3 (foreground: sun, midground: trees, background: mountains)',
                        'leading_lines': 'None',
                        'confidence': 0.8,
                        'description': f"{shot_type} shot of environment"
                    }
                elif people_count == 1:
                    framing_data = {
                        'subject_positioning': 'Centered',
                        'shot_balance': 'Symmetrical',
                        'depth_layers': '2 (foreground: subject, background: environment)',
                        'leading_lines': 'None',
                        'confidence': 0.85,
                        'description': f"{shot_type} shot of single subject"
                    }
                elif people_count == 2:
                    framing_data = {
                        'subject_positioning': 'Balanced',
                        'shot_balance': 'Symmetrical',
                        'depth_layers': '2 (foreground: subjects, background: environment)',
                        'leading_lines': 'None',
                        'confidence': 0.8,
                        'description': f"{shot_type} two-shot"
                    }
                else:
                    framing_data = {
                        'subject_positioning': 'Grouped',
                        'shot_balance': 'Asymmetrical',
                        'depth_layers': '2 (foreground: subjects, background: environment)',
                        'leading_lines': 'None',
                        'confidence': 0.75,
                        'description': f"{shot_type} group shot ({people_count} people)"
                    }
                
                print(f"Determined framing: {framing_data}")
        
        # Format the framing data into a consistent structure
        formatted_framing = self._format_framing(framing_data)
        print(f"Final framing: {formatted_framing}")
        return formatted_framing
    
    def _determine_visual_focus(self, scene_analysis: Dict, 
                               key_objects: List[Dict]) -> str:
        """Determine what the visual focus is"""
        # Use AI analysis if available
        if 'key_subjects' in scene_analysis:
            subjects = scene_analysis['key_subjects']
            if subjects:
                return ', '.join(subjects[:3])
        
        # Fall back to object detection
        if key_objects:
            return ', '.join([obj['class'] for obj in key_objects[:3]])
        
        return "scene composition"
    
    def _generate_scene_description(self, scene_analysis: Dict, 
                                   key_objects: List[Dict]) -> str:
        """Generate natural language scene description"""
        # Use AI description if available
        if 'ai_description' in scene_analysis:
            return scene_analysis['ai_description']
        
        # Generate basic description
        parts = []
        
        # Environment and time
        env = scene_analysis.get('environment', 'a location')
        time = scene_analysis.get('time_of_day', 'day').lower()
        parts.append(f"A {env} scene during {time}time")
        
        # Objects
        if key_objects:
            object_names = [obj['class'] for obj in key_objects[:3]]
            parts.append(f"featuring {', '.join(object_names)}")
        
        # Mood
        mood = scene_analysis.get('mood', 'neutral')
        if mood != 'neutral':
            parts.append(f"with a {mood} atmosphere")
        
        return '. '.join(parts) + '.'
    
    def _detect_transitions(self, use_ai: bool = False):
        """Detect and classify transitions between scenes"""
        print("\n=== Detecting Scene Transitions ===")
        
        prev_scene = None
        for i, current_scene in enumerate(self.scenes):
            print(f"\n--- Processing Scene {current_scene.scene_id} ---")
            print(f"Before detection - transition_in: {current_scene.transition_in}")
            print(f"Before detection - transition_out: {current_scene.transition_out}")
            
            # Initialize default transition values
            transition_type = "Cut"  # Default transition type
            description = f"Cut to scene {current_scene.scene_id}"
            
            # Handle transition in (not for first scene)
            if prev_scene is not None:
                # Use AI for transition analysis if available
                if use_ai and self.ai_analyzer:
                    try:
                        # Get last frame of previous scene and first frame of current
                        frame1 = self.analyzer.extract_frame_at_time(prev_scene.end_time - 0.1)
                        frame2 = self.analyzer.extract_frame_at_time(current_scene.start_time)
                        
                        if frame1 is not None and frame2 is not None:
                            trans_analysis = self.ai_analyzer.analyze_transition(frame1, frame2)
                            transition_type = trans_analysis.get('suggested_transition', transition_type)
                            description = trans_analysis.get('technical_notes', description)
                    except Exception as e:
                        print(f"    Warning: AI transition analysis failed: {e}")
                else:
                    # Heuristic-based transition detection
                    transition_type = self._infer_transition_type(prev_scene, current_scene)
                    description = f"Transition from scene {prev_scene.scene_id} to {current_scene.scene_id}"
                
                # Only set transition_in if not already set
                if current_scene.transition_in is None:
                    current_scene.transition_in = {
                        'type': transition_type,
                        'from_scene_id': prev_scene.scene_id,
                        'description': description
                    }
                    print(f"Set transition_in: {current_scene.transition_in}")
                else:
                    print(f"Using existing transition_in: {current_scene.transition_in}")
            
            # Handle transition out (not for last scene)
            if i < len(self.scenes) - 1:
                next_scene = self.scenes[i+1]
                
                # Only set transition_out if not already set by AI analysis
                if current_scene.transition_out is None:
                    current_scene.transition_out = {
                        'type': 'Cut',  # Default to Cut if not determined
                        'to_scene_id': next_scene.scene_id,
                        'description': f"Cut to scene {next_scene.scene_id}"
                    }
                    print(f"Set default transition_out: {current_scene.transition_out}")
                else:
                    print(f"Using existing transition_out: {current_scene.transition_out}")
            
            # Update previous scene for next iteration
            prev_scene = current_scene
            
            # Debug output
            print(f"After processing - transition_in: {current_scene.transition_in}")
            print(f"After processing - transition_out: {current_scene.transition_out}")
    
    def _infer_transition_type(self, scene1: Scene, scene2: Scene) -> str:
        """Infer transition type based on scene characteristics"""
        motion1 = scene1.scene_metadata.get('motion_intensity', 0)
        motion2 = scene2.scene_metadata.get('motion_intensity', 0)
        avg_motion = (motion1 + motion2) / 2
        
        mood1 = scene1.scene_metadata.get('mood', 'neutral')
        mood2 = scene2.scene_metadata.get('mood', 'neutral')
        
        # High motion → cut
        if avg_motion > 8.0:
            return TransitionType.CUT.value
        
        # Mood change → dissolve
        if mood1 != mood2:
            return TransitionType.DISSOLVE.value
        
        # Low motion → fade
        if avg_motion < 3.0:
            return TransitionType.FADE_IN.value
        
        return TransitionType.CUT.value
    
    def export_storyboard(self, output_path: str):
        """Export complete storyboard to JSON"""
        storyboard = Storyboard(
            scenes=self.scenes,
            metadata={
                'video_path': self.analyzer.video_path,
                'fps': self.analyzer.fps,
                'duration': self.analyzer.duration,
                'resolution': f"{self.analyzer.width}x{self.analyzer.height}",
                'frame_count': self.analyzer.frame_count,
                'total_scenes': len(self.scenes),
                'total_shots': sum(len(scene.shots) for scene in self.scenes),
                'ai_enhanced': self.ai_analyzer is not None
            }
        )
        
        storyboard.to_json(output_path)
        print(f"✓ Storyboard exported to: {output_path}")
    
    def export_table(self, output_path: str):
        """Export human-readable storyboard table"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("STORYBOARD - SCENE BREAKDOWN\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Video: {self.analyzer.video_path}\n")
            f.write(f"Duration: {self.analyzer.duration:.2f}s | "
                   f"FPS: {self.analyzer.fps:.2f} | "
                   f"Scenes: {len(self.scenes)}\n\n")
            
            for scene in self.scenes:
                f.write("=" * 100 + "\n")
                f.write(f"SCENE {scene.scene_id}\n")
                f.write("=" * 100 + "\n")
                f.write(f"Timecode: {scene.start_time:.2f}s - {scene.end_time:.2f}s "
                       f"(Duration: {scene.scene_metadata['duration']:.2f}s)\n\n")
                
                # Metadata
                meta = scene.scene_metadata
                f.write(f"Environment: {meta['environment']}\n")
                f.write(f"Time of Day: {meta['time_of_day']}\n")
                f.write(f"Mood: {meta['mood']}\n")
                f.write(f"Motion Intensity: {meta['motion_intensity']:.2f}\n\n")
                
                # Description
                f.write(f"Description:\n{meta['description']}\n\n")
                
                # AI description if available
                if 'ai_description' in meta:
                    f.write(f"AI Analysis:\n{meta['ai_description']}\n\n")
                
                # Shots
                if scene.shots:
                    f.write("Shots:\n")
                    for shot_data in scene.shots:
                        shot = shot_data['shot']
                        f.write(f"  Shot {shot['shot_id']}: {shot['shot_type']} | "
                               f"{shot['camera_move']}\n")
                        f.write(f"    Framing: {shot['framing']}\n")
                        f.write(f"    Focus: {shot['visual_focus']}\n")
                        if shot['notes']:
                            f.write(f"    Notes: {shot['notes']}\n")
                    f.write("\n")
                
                # Transitions
                if scene.transition_in:
                    f.write(f"Transition In: {scene.transition_in['type']}\n")
                if scene.transition_out:
                    f.write(f"Transition Out: {scene.transition_out['type']}\n")
                
                f.write("\n")
        
        print(f"✓ Table exported to: {output_path}")
