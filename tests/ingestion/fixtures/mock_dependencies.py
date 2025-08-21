"""
Mock implementations of external dependencies for testing.

Provides mock versions of OpenCV, Whisper, PyTorch, and other
dependencies used by the ingestion pipeline.
"""

import numpy as np
from unittest.mock import Mock
from typing import Dict, Any, List, Optional
from pathlib import Path


class MockVideoCapture:
    """Mock implementation of cv2.VideoCapture."""
    
    def __init__(self, video_path: str, fps: float = 30.0, duration: float = 5.0,
                 width: int = 640, height: int = 480):
        self.video_path = video_path
        self.fps = fps
        self.duration = duration
        self.width = width
        self.height = height
        self.total_frames = int(fps * duration)
        self.current_frame = 0
        self.is_opened = True
    
    def get(self, prop: int) -> float:
        """Mock cv2.VideoCapture.get()."""
        import cv2
        prop_map = {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.total_frames,
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height
        }
        return prop_map.get(prop, 0.0)
    
    def isOpened(self) -> bool:
        """Mock cv2.VideoCapture.isOpened()."""
        return self.is_opened
    
    def read(self):
        """Mock cv2.VideoCapture.read()."""
        if self.current_frame >= self.total_frames:
            return False, None
        
        # Generate a frame with varying colors based on frame number
        intensity = int((self.current_frame / max(self.total_frames - 1, 1)) * 255)
        frame = np.full((self.height, self.width, 3), intensity, dtype=np.uint8)
        
        self.current_frame += 1
        return True, frame
    
    def release(self):
        """Mock cv2.VideoCapture.release()."""
        self.is_opened = False
        
    def set(self, prop: int, value: float) -> bool:
        """Mock cv2.VideoCapture.set()."""
        return True


class MockWhisperModel:
    """Mock implementation of Whisper model."""
    
    def __init__(self, model_name: str = "whisper-base"):
        self.model_name = model_name
        
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Mock Whisper transcription."""
        # Generate mock transcription based on audio file name
        video_name = Path(audio_path).stem
        
        mock_transcription = {
            "text": f"This is a mock transcription for {video_name}.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "This is a mock transcription"
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": f"for {video_name}."
                }
            ],
            "language": language or "en"
        }
        
        return mock_transcription


class MockEmbeddingModel:
    """Mock implementation of embedding model."""
    
    def __init__(self, model_name: str, embedding_dim: int = 128):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Mock text encoding."""
        # Generate deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Use text hash to create deterministic embedding
            text_hash = hash(text) % 1000000
            np.random.seed(text_hash)
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Mock image encoding."""
        # Generate embedding based on image properties
        image_hash = hash(image.tobytes()) % 1000000
        np.random.seed(image_hash)
        embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        return embedding


class MockVespaClient:
    """Mock implementation of Vespa client."""
    
    def __init__(self):
        self.documents = []
        self.schema = None
    
    def feed_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Mock document feeding."""
        self.documents.append(document)
        return {"status": "success", "id": document.get("id", "mock_id")}
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock query execution."""
        return {
            "hits": [
                {
                    "id": "mock_hit_1",
                    "score": 0.95,
                    "fields": {"title": "Mock result 1"}
                },
                {
                    "id": "mock_hit_2", 
                    "score": 0.80,
                    "fields": {"title": "Mock result 2"}
                }
            ],
            "total": 2
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Mock schema retrieval."""
        return self.schema or {"name": "mock_schema", "fields": []}


class MockCacheManager:
    """Mock implementation of cache manager."""
    
    def __init__(self):
        self._cache = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Mock cache get."""
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Mock cache set."""
        self._cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Mock cache delete."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Mock cache clear."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
    
    def exists(self, key: str) -> bool:
        """Mock cache exists check."""
        return key in self._cache


class MockFFmpeg:
    """Mock implementation of FFmpeg operations."""
    
    @staticmethod
    def extract_audio(video_path: Path, audio_path: Path) -> bool:
        """Mock audio extraction."""
        # Create a dummy audio file
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.touch()
        return True
    
    @staticmethod
    def extract_frames(video_path: Path, output_dir: Path, fps: float = 1.0) -> List[Path]:
        """Mock frame extraction."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock frame files
        frames = []
        for i in range(5):  # Create 5 mock frames
            frame_path = output_dir / f"frame_{i:04d}.jpg"
            frame_path.touch()
            frames.append(frame_path)
        
        return frames
    
    @staticmethod
    def get_video_info(video_path: Path) -> Dict[str, Any]:
        """Mock video information extraction."""
        return {
            "duration": 5.0,
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "codec": "h264"
        }


class MockOutputManager:
    """Mock implementation of output manager."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure output directories exist."""
        dirs = ["keyframes", "transcripts", "chunks", "embeddings", "metadata", "cache"]
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def get_processing_dir(self, processing_type: str) -> Path:
        """Mock get processing directory."""
        return self.base_dir / processing_type
    
    def get_video_output_dir(self, video_id: str) -> Path:
        """Mock get video-specific output directory."""
        return self.base_dir / "videos" / video_id
    
    def get_cache_dir(self) -> Path:
        """Mock get cache directory."""
        return self.base_dir / "cache"


def create_mock_cv2():
    """Create a comprehensive mock of cv2 module."""
    mock_cv2 = Mock()
    
    # Mock constants
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.HISTCMP_CORREL = 0
    
    # Mock VideoWriter fourcc
    def mock_fourcc(*args):
        return 0x21
    mock_cv2.VideoWriter_fourcc = mock_fourcc
    
    # Mock VideoCapture
    mock_cv2.VideoCapture = MockVideoCapture
    
    # Mock image operations
    mock_cv2.imread = lambda path: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_cv2.imwrite = Mock(return_value=True)
    mock_cv2.calcHist = lambda img, *args: np.random.rand(256).astype(np.float32)
    mock_cv2.compareHist = Mock(return_value=0.8)
    mock_cv2.cvtColor = lambda img, code: img  # Return unchanged for simplicity
    mock_cv2.COLOR_HSV2BGR = 40
    mock_cv2.COLOR_BGR2GRAY = 6
    
    return mock_cv2


def create_mock_whisper():
    """Create a comprehensive mock of whisper module."""
    mock_whisper = Mock()
    mock_whisper.load_model = Mock(return_value=MockWhisperModel())
    return mock_whisper


def create_mock_torch():
    """Create a comprehensive mock of torch module."""
    mock_torch = Mock()
    
    # Mock tensor creation
    mock_torch.tensor = Mock(return_value=Mock())
    mock_torch.load = Mock(return_value=Mock())
    mock_torch.no_grad = lambda: Mock().__enter__()
    
    # Mock CUDA operations
    mock_torch.cuda = Mock()
    mock_torch.cuda.is_available = Mock(return_value=False)
    
    return mock_torch