import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import torch
import insightface
from insightface.app import FaceAnalysis
import logging

logger = logging.getLogger(__name__)

class FaceProcessor:
    """Handles face detection, alignment, and swapping for video frames"""
    
    def __init__(self, device: str = None):
        """Initialize face processor with models"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_analyser = None
        self.face_swapper = None
        self.reference_face = None
        self.reference_embedding = None
        self.initialize_models()
    
    def initialize_models(self) -> None:
        """Initialize face analysis and swapping models"""
        try:
            # Initialize face analysis model
            self.face_analyser = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider']
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            
            # Initialize face swapper
            self.face_swapper = insightface.model_zoo.get_model(
                'inswapper_128.onnx',
                download=True,
                download_zip=True
            )
            
            if self.device == 'cuda':
                self.face_swapper.to('cuda')
                
            logger.info(f"Face processor initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize face models: {str(e)}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect all faces in an image"""
        if self.face_analyser is None:
            self.initialize_models()
            
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Detect faces
        faces = self.face_analyser.get(image)
        return faces
    
    def set_reference_face(self, image: np.ndarray, face_index: int = 0) -> bool:
        """Set the reference face from an image"""
        faces = self.detect_faces(image)
        if not faces:
            logger.warning("No faces found in reference image")
            return False
            
        if face_index >= len(faces):
            logger.warning(f"Face index {face_index} out of range, using first face")
            face_index = 0
            
        self.reference_face = faces[face_index]
        self.reference_embedding = self.reference_face.embedding
        logger.info("Reference face set successfully")
        return True
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        """Swap faces from source to target image"""
        if self.reference_face is None:
            logger.warning("No reference face set, using first face from source")
            source_faces = self.detect_faces(source_image)
            if not source_faces:
                logger.warning("No faces found in source image")
                return target_image
            source_face = source_faces[0]
        else:
            source_face = self.reference_face
            
        target_faces = self.detect_faces(target_image)
        if not target_faces:
            logger.warning("No faces found in target image")
            return target_image
            
        # Convert to BGR for processing
        if isinstance(target_image, Image.Image):
            target_image = np.array(target_image)
        if len(target_image.shape) == 3 and target_image.shape[2] == 3:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
            
        # Swap faces
        result = target_image.copy()
        for face in target_faces:
            result = self.face_swapper.get(result, face, source_face, paste_back=True)
            
        # Convert back to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
    
    def process_frame(self, frame: np.ndarray, reference_face: bool = True) -> np.ndarray:
        """Process a single frame with face consistency"""
        if reference_face and self.reference_face is None:
            logger.warning("No reference face set, detecting one from the first frame")
            if not self.set_reference_face(frame):
                return frame
                
        if reference_face and self.reference_face is not None:
            return self.swap_faces(None, frame)
            
        return frame

def test_face_processor():
    """Test function for the FaceProcessor class"""
    import matplotlib.pyplot as plt
    
    # Initialize processor
    processor = FaceProcessor()
    
    # Load test images
    source_img = Image.open("test_source.jpg").convert("RGB")
    target_img = Image.open("test_target.jpg").convert("RGB")
    
    # Set reference face
    processor.set_reference_face(source_img)
    
    # Process target image
    result = processor.swap_faces(source_img, target_img)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Source")
    plt.imshow(source_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Target")
    plt.imshow(target_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Result")
    plt.imshow(result)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_face_processor()
