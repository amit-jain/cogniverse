"""
Unit tests for ChunkProcessor.

Tests video chunking functionality for time-based segmentation.
"""

import pytest
from unittest.mock import Mock, patch

from src.app.ingestion.processors.chunk_processor import ChunkProcessor


class TestChunkProcessor:
    """Test cases for ChunkProcessor."""
    
    @pytest.fixture
    def processor(self, mock_logger):
        """Create a chunk processor for testing."""
        return ChunkProcessor(
            mock_logger,
            chunk_duration=30.0,
            chunk_overlap=2.0,
            cache_chunks=True
        )
    
    @pytest.fixture
    def no_cache_processor(self, mock_logger):
        """Create a chunk processor without caching."""
        return ChunkProcessor(
            mock_logger,
            chunk_duration=15.0,
            chunk_overlap=1.0,
            cache_chunks=False
        )
    
    def test_processor_initialization(self, mock_logger):
        """Test chunk processor initialization."""
        processor = ChunkProcessor(
            mock_logger,
            chunk_duration=60.0,
            chunk_overlap=5.0,
            cache_chunks=False
        )
        
        assert processor.PROCESSOR_NAME == "chunk"
        assert processor.logger == mock_logger
        assert processor.chunk_duration == 60.0
        assert processor.chunk_overlap == 5.0
        assert processor.cache_chunks is False
    
    def test_processor_initialization_defaults(self, mock_logger):
        """Test processor with default values."""
        processor = ChunkProcessor(mock_logger)
        
        assert processor.chunk_duration == 30.0
        assert processor.chunk_overlap == 2.0
        assert processor.cache_chunks is True
    
    def test_from_config_factory_method(self, mock_logger):
        """Test creating processor from configuration."""
        config = {
            "chunk_duration": 45.0,
            "chunk_overlap": 3.0,
            "cache_chunks": False
        }
        
        processor = ChunkProcessor.from_config(config, mock_logger)
        
        assert processor.chunk_duration == 45.0
        assert processor.chunk_overlap == 3.0
        assert processor.cache_chunks is False
    
    def test_from_config_with_defaults(self, mock_logger):
        """Test from_config uses defaults for missing parameters."""
        config = {"chunk_duration": 20.0}
        
        processor = ChunkProcessor.from_config(config, mock_logger)
        
        assert processor.chunk_duration == 20.0
        assert processor.chunk_overlap == 2.0  # default
        assert processor.cache_chunks is True  # default
    
    def test_calculate_chunk_intervals_basic(self, processor):
        """Test basic chunk interval calculation."""
        video_duration = 100.0  # 100 seconds
        
        intervals = processor._calculate_chunk_intervals(video_duration)
        
        # Should create chunks with 30s duration and 2s overlap
        # Expected: [0-30], [28-58], [56-86], [84-100]
        assert len(intervals) >= 3
        
        # First chunk
        assert intervals[0] == (0.0, 30.0)
        
        # Check overlap (next chunk should start 2 seconds before previous ends)
        assert intervals[1][0] == 28.0  # 30 - 2
        assert intervals[1][1] == 58.0  # 30 - 2 + 30
        
        # Last chunk should end at video duration
        assert intervals[-1][1] == video_duration
    
    def test_calculate_chunk_intervals_short_video(self, processor):
        """Test chunk calculation for video shorter than chunk duration."""
        video_duration = 15.0  # Shorter than 30s chunk
        
        intervals = processor._calculate_chunk_intervals(video_duration)
        
        # Should create single chunk covering entire video
        assert len(intervals) == 1
        assert intervals[0] == (0.0, 15.0)
    
    def test_calculate_chunk_intervals_no_overlap(self, mock_logger):
        """Test chunk calculation with no overlap."""
        processor = ChunkProcessor(mock_logger, chunk_duration=10.0, chunk_overlap=0.0)
        video_duration = 35.0
        
        intervals = processor._calculate_chunk_intervals(video_duration)
        
        # Should create non-overlapping chunks: [0-10], [10-20], [20-30], [30-35]
        expected = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 35.0)]
        assert intervals == expected
    
    def test_calculate_chunk_intervals_large_overlap(self, mock_logger):
        """Test chunk calculation with large overlap."""
        processor = ChunkProcessor(mock_logger, chunk_duration=10.0, chunk_overlap=8.0)
        video_duration = 25.0
        
        intervals = processor._calculate_chunk_intervals(video_duration)
        
        # With 8s overlap, chunks should start every 2s
        # [0-10], [2-12], [4-14], [6-16], [8-18], [10-20], [12-22], [14-24], [16-25]
        assert len(intervals) >= 8
        assert intervals[0] == (0.0, 10.0)
        assert intervals[1] == (2.0, 12.0)  # 10 - 8 = 2
        assert intervals[-1][1] == video_duration
    
    @patch('src.app.ingestion.processors.video_chunk_extractor.VideoChunkExtractor')
    def test_extract_chunks_success(self, mock_extractor_class, processor, temp_dir, sample_video_path):
        """Test successful chunk extraction."""
        # Mock the VideoChunkExtractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock chunk extraction results
        mock_extractor.extract_chunks.return_value = {
            "video_id": "test_video",
            "chunks": [
                {"start_time": 0.0, "end_time": 30.0, "filename": "chunk_0.mp4"},
                {"start_time": 28.0, "end_time": 58.0, "filename": "chunk_1.mp4"},
                {"start_time": 56.0, "end_time": 86.0, "filename": "chunk_2.mp4"}
            ],
            "total_chunks": 3
        }
        
        with patch('src.common.utils.output_manager.get_output_manager') as mock_get_output_manager:
            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager
            
            result = processor.extract_chunks(sample_video_path, video_duration=90.0)
        
        # Verify result
        assert result["video_id"] == "test_video"
        assert result["total_chunks"] == 3
        assert len(result["chunks"]) == 3
        
        # Should have called VideoChunkExtractor with correct parameters
        mock_extractor_class.assert_called_once()
        mock_extractor.extract_chunks.assert_called_once()
        
        call_args = mock_extractor.extract_chunks.call_args[1]  # kwargs
        assert "chunk_intervals" in call_args
        intervals = call_args["chunk_intervals"]
        assert len(intervals) >= 3
    
    @patch('src.app.ingestion.processors.video_chunk_extractor.VideoChunkExtractor')
    def test_extract_chunks_with_caching(self, mock_extractor_class, processor, temp_dir, sample_video_path, mock_cache_manager):
        """Test chunk extraction with caching enabled."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        cached_result = {
            "video_id": "test_video",
            "chunks": [{"start_time": 0.0, "end_time": 30.0, "filename": "cached_chunk.mp4"}],
            "total_chunks": 1
        }
        
        with patch.object(processor, '_get_cache_manager', return_value=mock_cache_manager):
            # First call - cache miss
            mock_cache_manager.get.return_value = None
            mock_extractor.extract_chunks.return_value = cached_result
            
            with patch('src.common.utils.output_manager.get_output_manager') as mock_get_output_manager:
                mock_output_manager = Mock()
                mock_output_manager.get_processing_dir.return_value = temp_dir
                mock_get_output_manager.return_value = mock_output_manager
                
                result1 = processor.extract_chunks(sample_video_path, video_duration=60.0)
            
            # Should have cached the result
            mock_cache_manager.set.assert_called_once()
            
            # Second call - cache hit
            mock_cache_manager.get.return_value = cached_result
            result2 = processor.extract_chunks(sample_video_path, video_duration=60.0)
            
            # Should return cached result without calling extractor again
            assert result2 == cached_result
            mock_extractor.extract_chunks.assert_called_once()  # Only called once
    
    @patch('src.app.ingestion.processors.video_chunk_extractor.VideoChunkExtractor')
    def test_extract_chunks_no_caching(self, mock_extractor_class, no_cache_processor, temp_dir, sample_video_path):
        """Test chunk extraction without caching."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        chunk_result = {
            "video_id": "test_video",
            "chunks": [{"start_time": 0.0, "end_time": 15.0, "filename": "chunk.mp4"}],
            "total_chunks": 1
        }
        mock_extractor.extract_chunks.return_value = chunk_result
        
        with patch('src.common.utils.output_manager.get_output_manager') as mock_get_output_manager:
            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager
            
            result = no_cache_processor.extract_chunks(sample_video_path, video_duration=30.0)
        
        assert result == chunk_result
        # Should not have attempted caching
        assert not hasattr(no_cache_processor, '_get_cache_manager') or \
               no_cache_processor._get_cache_manager is None
    
    def test_extract_chunks_invalid_duration(self, processor, sample_video_path):
        """Test handling of invalid video duration."""
        with pytest.raises(ValueError, match="Video duration must be positive"):
            processor.extract_chunks(sample_video_path, video_duration=-10.0)
        
        with pytest.raises(ValueError, match="Video duration must be positive"):
            processor.extract_chunks(sample_video_path, video_duration=0.0)
    
    @patch('src.app.ingestion.processors.video_chunk_extractor.VideoChunkExtractor')
    def test_extract_chunks_extractor_error(self, mock_extractor_class, processor, temp_dir, sample_video_path):
        """Test handling of chunk extractor errors."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_chunks.side_effect = Exception("Extraction failed")
        
        with patch('src.common.utils.output_manager.get_output_manager') as mock_get_output_manager:
            mock_output_manager = Mock()
            mock_output_manager.get_processing_dir.return_value = temp_dir
            mock_get_output_manager.return_value = mock_output_manager
            
            with pytest.raises(Exception, match="Extraction failed"):
                processor.extract_chunks(sample_video_path, video_duration=60.0)
    
    def test_get_config(self, processor):
        """Test retrieving processor configuration."""
        config = processor.get_config()
        
        expected = {
            "chunk_duration": 30.0,
            "chunk_overlap": 2.0,
            "cache_chunks": True
        }
        assert config == expected


class TestChunkProcessorCacheManagement:
    """Test caching behavior of ChunkProcessor."""
    
    @pytest.fixture
    def processor(self, mock_logger):
        return ChunkProcessor(mock_logger, cache_chunks=True)
    
    def test_cache_key_generation(self, processor, sample_video_path):
        """Test that cache keys are generated correctly."""
        cache_key = processor._get_cache_key(sample_video_path, 120.0)
        
        # Should include video path and duration
        assert str(sample_video_path) in cache_key
        assert "120.0" in cache_key
        assert processor.chunk_duration in cache_key
        assert processor.chunk_overlap in cache_key
    
    def test_cache_key_uniqueness(self, processor, temp_dir):
        """Test that different parameters generate different cache keys."""
        video1 = temp_dir / "video1.mp4"
        video2 = temp_dir / "video2.mp4"
        
        key1 = processor._get_cache_key(video1, 60.0)
        key2 = processor._get_cache_key(video2, 60.0)  # Different video
        key3 = processor._get_cache_key(video1, 90.0)  # Different duration
        
        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
    
    def test_cache_disabled_processor(self, mock_logger):
        """Test that caching is properly disabled when cache_chunks=False."""
        processor = ChunkProcessor(mock_logger, cache_chunks=False)
        
        # Should not have cache manager
        with patch.object(processor, '_get_cache_manager') as mock_get_cache:
            mock_get_cache.return_value = None
            
            cache_manager = processor._get_cache_manager()
            assert cache_manager is None