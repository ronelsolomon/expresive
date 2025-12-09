"""
ai_analyzer.py - AI-Powered Scene Analysis with Ollama
Local AI analysis using Ollama vision models (LLaVA, LLaMA-Vision, etc.)
"""

import base64
import json
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import requests
from pathlib import Path


class OllamaAnalyzer:
    """
    AI-powered scene analysis using local Ollama vision models
    Supports LLaVA, Bakllava, and other vision-capable models
    """
    
    # Recommended models for video analysis
    RECOMMENDED_MODELS = {
        'llava': 'llava:latest',  # Good balance of speed and accuracy
        'llava-13b': 'llava:13b',  # More accurate but slower
        'bakllava': 'bakllava:latest',  # Alternative vision model
        'llava-llama3': 'llava-llama3:latest',  # Latest LLaVA with Llama 3
    }
    
    def __init__(self, 
                 model: str = "llava:latest",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120):
        """
        Initialize Ollama analyzer
        
        Args:
            model: Model name (default: llava:latest)
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 120)
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.available = False
        
        print(f"Initializing Ollama analyzer with model: {model}")
        self._check_connection()
        
    def _check_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            # Get available models
            models_data = response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            
            if self.model in available_models:
                print(f"✓ Model '{self.model}' is available")
                self.available = True
            else:
                print(f"⚠️  Model '{self.model}' not found")
                print(f"   Available models: {', '.join(available_models) if available_models else 'None'}")
                print(f"   Install with: ollama pull {self.model}")
                
                # Suggest alternatives
                for rec_name, rec_model in self.RECOMMENDED_MODELS.items():
                    if rec_model in available_models:
                        print(f"   Or use available model: {rec_model}")
                        break
                        
        except requests.exceptions.ConnectionError:
            print(f"⚠️  Cannot connect to Ollama at {self.base_url}")
            print("   Make sure Ollama is running: ollama serve")
            print("   Or check if the base_url is correct")
        except Exception as e:
            print(f"⚠️  Error checking Ollama: {e}")
    
    def encode_frame(self, frame: np.ndarray, max_size: int = 1024) -> str:
        """
        Encode frame to base64 for Ollama API
        
        Args:
            frame: OpenCV frame (BGR format)
            max_size: Maximum dimension (width or height)
            
        Returns:
            Base64 encoded JPEG string
        """
        # Resize if needed
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Encode to JPEG
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to base64
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_frame(self, 
                     frame: np.ndarray, 
                     context: str = "",
                     camera_analysis: Optional[Dict] = None) -> Dict:
        """
        Analyze a single frame with comprehensive cinematography breakdown
        
        Args:
            frame: OpenCV frame (BGR format)
            context: Additional context about the video
            camera_analysis: Optional camera analysis for fallback
            
        Returns:
            Dictionary with scene analysis
        """
        if not self.available:
            print("⚠️  Ollama not available, using fallback analysis")
            return self._generate_fallback_analysis(frame, camera_analysis)
        
        try:
            base64_frame = self.encode_frame(frame)
            
            # Build comprehensive prompt
            prompt = self._build_analysis_prompt(context)
            
            # Make API request
            response = self._call_ollama_api(
                prompt=prompt,
                images=[base64_frame]
            )
            
            # Parse and validate response
            analysis = self._parse_analysis_response(response)
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            return self._generate_fallback_analysis(frame, camera_analysis)
    
    def _build_analysis_prompt(self, context: str = "") -> str:
        """Build comprehensive analysis prompt"""
        prompt = """You are a professional cinematographer analyzing a video frame. Provide a detailed breakdown.

{context}

Analyze and respond in this EXACT JSON format (no additional text):
{{
  "shot_type": "Choose ONE: BIRDS EYE, EARTH EYE,ENVIRONMENTAL, WIDE, MEDIUM, CLOSE-UP, ECU, OTS, TWO-SHOT",
  "camera_move": "Choose ONE: Still, Push, Pull, Pan Left, Pan Right, Tilt Up, Tilt Down, Handheld, Zoom In, Zoom Out",
  "framing": "Describe the composition in detail. Include: 1) Subject positioning (centered, rule of thirds, symmetrical, etc.) 2) Shot balance 3) Depth layers (foreground/midground/background) 4) Any leading lines or geometric patterns 5) Framing confidence score if available. Example: 'Subject on left third with strong leading lines, 3 distinct depth layers, high confidence (0.85)'",
  "visual_focus": "What draws the viewer's attention",
  "environment": "Choose ONE: indoor, outdoor, ambiguous",
  "time_of_day": "Choose ONE: dawn, day, dusk, night, unknown",
  "mood": "Choose ONE: calm, tense, joyful, melancholic, energetic, mysterious, romantic, dramatic, neutral",
  "key_subjects": ["list", "of", "main", "subjects", "or", "objects"],
  "scene_description": "Brief narrative description (1-2 sentences)",
  "cinematography_notes": "Technical observations about lighting, depth of field, color grading",
  "suggested_next_shot": "Logical continuation for storytelling",
  "transition_in": {
    "type": "cut",
    "from_scene_id": -1,
    "description": "Description of how this scene is transitioned into"
  },
  "transition_out": {
    "type": "cut",
    "to_scene_id": -1,
    "description": "Description of how this scene transitions out"
  }
}}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no explanations."""

        if context:
            prompt = prompt.replace("{context}", f"Context: {context}")
        else:
            prompt = prompt.replace("{context}\n\n", "")
        
        return prompt
    
    def _call_ollama_api(self, 
                        prompt: str, 
                        images: List[str],
                        stream: bool = False) -> str:
        """
        Make API call to Ollama
        
        Args:
            prompt: Text prompt
            images: List of base64-encoded images
            stream: Whether to stream the response
            
        Returns:
            Model's response text
        """
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": images
            }],
            "stream": stream,
            "options": {
                "temperature": 0.3,  # Lower for more consistent structured output
                "num_predict": 1024   # Max tokens
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )
        response.raise_for_status()
        print(response)
        
        result = response.json()
        return result["message"]["content"]
    
    def _parse_analysis_response(self, response: str) -> Dict:
        """
        Parse and validate the model's response
        
        Args:
            response: Raw response from model
            
        Returns:
            Parsed and validated dictionary
        """
        # Debug: Print raw response
        print(f"\n=== AI Raw Response ===\n{response[:500]}{'...' if len(response) > 500 else ''}\n======================\n")
        
        # Clean up the response (sometimes models add markdown code blocks)
        content = response.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
            
        try:
            data = json.loads(content)
            
            # Debug: Print parsed transitions
            print(f"Parsed AI Response - transition_in: {data.get('transition_in')}")
            print(f"Parsed AI Response - transition_out: {data.get('transition_out')}")
            
            # Validate required fields
            required_fields = [
                'shot_type', 'camera_move', 'framing', 'visual_focus',
                'environment', 'time_of_day', 'mood', 'key_subjects',
                'scene_description', 'cinematography_notes', 'suggested_next_shot'
            ]
            
            # Set default values for required fields
            for field in required_fields:
                if field not in data:
                    print(f"⚠️  Missing field: {field}")
                    data[field] = "unknown"
            
            # Set default values for transition fields if not provided
            if 'transition_in' not in data:
                data['transition_in'] = {
                    'type': 'cut',
                    'from_scene_id': -1,
                    'description': 'Cut from previous scene'
                }
                
            if 'transition_out' not in data:
                data['transition_out'] = {
                    'type': 'cut',
                    'to_scene_id': -1,
                    'description': 'Cut to next scene'
                }
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON: {e}")
            print(f"   Raw response: {content[:200]}...")
            raise
        
        # Validate required fields
        required_fields = [
            'shot_type', 'camera_move', 'framing', 'visual_focus',
            'environment', 'time_of_day', 'mood', 'key_subjects',
            'scene_description', 'cinematography_notes', 'suggested_next_shot'
        ]
        
        # Set default values for required fields
        for field in required_fields:
            if field not in data:
                print(f"⚠️  Missing field: {field}")
                data[field] = "unknown"
        
        # Set default values for transition fields if not provided
        if 'transition_in' not in data:
            data['transition_in'] = {
                'type': 'cut',
                'from_scene_id': -1,
                'description': 'Cut from previous scene'
            }
            
        if 'transition_out' not in data:
            data['transition_out'] = {
                'type': 'cut',
                'to_scene_id': -1,
                'description': 'Cut to next scene'
            }
        
        return data
    
    def _generate_fallback_analysis(self, 
                                   frame: np.ndarray,
                                   camera_analysis: Optional[Dict] = None) -> Dict:
        """
        Generate basic analysis using computer vision when AI is unavailable
        
        Args:
            frame: OpenCV frame
            camera_analysis: Optional camera analysis data
            
        Returns:
            Basic analysis dictionary
        """
        # Calculate basic statistics from frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Estimate time of day from brightness
        if brightness < 50:
            time_of_day = "night"
        elif brightness < 100:
            time_of_day = "dusk"
        elif brightness < 200:
            time_of_day = "day"
        else:
            time_of_day = "dawn"
        
        # Use camera analysis if available
        if camera_analysis:
            shot_type = camera_analysis.get('shot_type', 'Wide')
            camera_move = camera_analysis.get('movement', 'Still')
            framing = camera_analysis.get('framing', 'centered')
            visual_focus = camera_analysis.get('visual_focus', 'the scene')
        else:
            shot_type = "Wide"
            camera_move = "Still"
            framing = "centered"
            visual_focus = "the scene"
        
        return {
            "shot_type": shot_type,
            "camera_move": camera_move,
            "framing": framing,
            "visual_focus": visual_focus,
            "environment": "ambiguous",
            "time_of_day": time_of_day,
            "mood": "neutral",
            "key_subjects": [visual_focus] if visual_focus != "the scene" else [],
            "scene_description": f"A {time_of_day} scene showing {visual_focus}",
            "cinematography_notes": "Basic analysis - AI not available",
            "transition_in": {
                "type": "cut",
                "from_scene_id": -1,
                "description": "Cut from previous scene"
            },
            "transition_out": {
                "type": "cut",
                "to_scene_id": -1,
                "description": "Cut to next scene"
            },
            "suggested_next_shot": "Continuation of the narrative"
        }
    
    def analyze_transition(self, 
                          frame1: np.ndarray, 
                          frame2: np.ndarray) -> Dict:
        """
        Analyze transition between two consecutive frames
        
        Args:
            frame1: Last frame of previous scene
            frame2: First frame of next scene
            
        Returns:
            Transition analysis dictionary
        """
        if not self.available:
            return self._generate_fallback_transition(frame1, frame2)
        
        try:
            # Encode both frames
            base64_frame1 = self.encode_frame(frame1)
            base64_frame2 = self.encode_frame(frame2)
            
            prompt = """Analyze the transition between these two consecutive video frames.

Frame 1: End of previous scene
Frame 2: Start of next scene

Respond in this EXACT JSON format (no additional text):
{{
  "transition_type": "Choose ONE: cut, fade, dissolve, cross_dissolve, whip_pan, match_cut, l_cut, j_cut, wipe",
  "visual_continuity": "Choose ONE: high, medium, low",
  "narrative_flow": "Does this feel like smooth story progression? (yes/no/unclear)",
  "suggested_transition": "What transition would work best and why",
  "technical_notes": "Observations about motion, composition, color continuity"
}}

Return ONLY the JSON object."""

            # Note: Ollama currently has limited multi-image support
            # We'll send both images but analyze them sequentially
            response = self._call_ollama_api(
                prompt=prompt,
                images=[base64_frame1, base64_frame2]
            )
            
            return self._parse_transition_response(response)
            
        except Exception as e:
            print(f"⚠️  Transition analysis failed: {e}")
            return self._generate_fallback_transition(frame1, frame2)
    
    def _parse_transition_response(self, response: str) -> Dict:
        """Parse transition analysis response"""
        content = response.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(content)
            
            # Ensure required fields
            defaults = {
                'transition_type': 'cut',
                'visual_continuity': 'medium',
                'narrative_flow': 'unclear',
                'suggested_transition': 'Standard cut',
                'technical_notes': 'No specific notes'
            }
            
            for key, default_value in defaults.items():
                if key not in data:
                    data[key] = default_value
            
            return data
            
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse transition response")
            return self._generate_fallback_transition(None, None)
    
    def _generate_fallback_transition(self, 
                                     frame1: Optional[np.ndarray],
                                     frame2: Optional[np.ndarray]) -> Dict:
        """Generate basic transition analysis"""
        if frame1 is not None and frame2 is not None:
            # Calculate histogram difference
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            if similarity > 0.8:
                transition_type = "dissolve"
                continuity = "high"
            elif similarity > 0.5:
                transition_type = "cut"
                continuity = "medium"
            else:
                transition_type = "cut"
                continuity = "low"
        else:
            transition_type = "cut"
            continuity = "medium"
        
        return {
            "transition_type": transition_type,
            "visual_continuity": continuity,
            "narrative_flow": "unclear",
            "suggested_transition": f"Standard {transition_type}",
            "technical_notes": "Basic analysis (AI unavailable)"
        }
    
    def batch_analyze_scenes(self, 
                            frames: List[np.ndarray],
                            video_context: str = "",
                            show_progress: bool = True) -> List[Dict]:
        """
        Analyze multiple frames in batch
        
        Args:
            frames: List of frames to analyze
            video_context: Context about the video
            show_progress: Whether to show progress
            
        Returns:
            List of analysis dictionaries
        """
        results = []
        total = len(frames)
        
        for i, frame in enumerate(frames, 1):
            if show_progress:
                print(f"Analyzing frame {i}/{total}...")
            
            try:
                analysis = self.analyze_frame(frame, video_context)
                results.append(analysis)
            except Exception as e:
                print(f"⚠️  Error analyzing frame {i}: {e}")
                results.append(self._generate_fallback_analysis(frame))
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test Ollama connection and model availability
        
        Returns:
            True if ready to use, False otherwise
        """
        print(f"\nTesting Ollama connection...")
        print(f"Base URL: {self.base_url}")
        print(f"Model: {self.model}")
        
        self._check_connection()
        
        if self.available:
            print("✓ Ollama is ready for AI analysis\n")
        else:
            print("✗ Ollama is not ready\n")
        
        return self.available


# Convenience function for quick setup
def create_analyzer(model: str = "llava:latest", 
                   base_url: str = "http://localhost:11434") -> OllamaAnalyzer:
    """
    Create and test an Ollama analyzer
    
    Args:
        model: Model name (default: llava:latest)
        base_url: Ollama server URL
        
    Returns:
        Configured OllamaAnalyzer instance
    """
    analyzer = OllamaAnalyzer(model=model, base_url=base_url)
    analyzer.test_connection()
    return analyzer


if __name__ == "__main__":
    # Test script
    print("="*80)
    print("Ollama Analyzer Test")
    print("="*80 + "\n")
    
    # Create analyzer
    analyzer = create_analyzer()
    
    if analyzer.available:
        print("Ready to analyze frames!")
        print("\nRecommended models:")
        for name, model in OllamaAnalyzer.RECOMMENDED_MODELS.items():
            print(f"  - {name}: {model}")
    else:
        print("Please start Ollama and install a vision model:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Install model: ollama pull llava:latest")