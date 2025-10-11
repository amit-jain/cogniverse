#!/usr/bin/env python3
"""
Video Player Tool for ADK Interface

Tool-based MVP approach for adding video playback with frame tagging.
Generates HTML with embedded video player and timeline markers for search results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import BaseTool
from google.genai.types import Part

from cogniverse_core.config.utils import get_config


class VideoPlayerTool(BaseTool):
    """Tool for playing videos with frame tagging based on search results"""

    def __init__(self):
        super().__init__(
            name="VideoPlayer",
            description="Play videos with frame tagging and timeline markers for search results",
        )
        self.config = get_config()
        self.video_dir = Path(self.config.get("video_dir", "data/videos"))

    async def execute(
        self,
        video_id: str,
        search_results: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate video player HTML with frame tagging

        Args:
            video_id: ID of the video to play
            search_results: JSON string of search results with frame timestamps
            start_time: Optional start time in seconds
        """
        try:
            # Find video file
            video_path = self._find_video_file(video_id)
            if not video_path:
                error_html = f"<html><body><p style='color: red;'>❌ Video not found: {video_id}</p></body></html>"
                error_artifact = Part.from_bytes(
                    data=error_html.encode("utf-8"), mime_type="text/html"
                )
                return {
                    "success": False,
                    "error": f"Video file not found for ID: {video_id}",
                    "video_player": error_artifact,
                    "message": f"❌ Video not found: {video_id}",
                }

            # Parse search results if provided
            frame_markers = []
            if search_results:
                try:
                    results_data = json.loads(search_results)
                    frame_markers = self._extract_frame_markers(results_data)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text and skip markers
                    pass

            # Generate interactive HTML video player with server-based video references
            html_content = self._generate_server_based_video_player_html(
                video_path, frame_markers, start_time
            )

            # Create ADK artifact for the HTML content
            html_artifact = Part.from_bytes(
                data=html_content.encode("utf-8"), mime_type="text/html"
            )

            return {
                "success": True,
                "video_path": str(video_path),
                "frame_count": len(frame_markers),
                "video_player": html_artifact,  # HTML player as separate artifact
                "message": f"🎬 Generated video player info for {video_path.name} with {len(frame_markers)} frame markers.",
            }

        except Exception as e:
            # Return error as HTML artifact for consistency
            error_html = f"<html><body><p style='color: red;'>❌ Error: {str(e)}</p></body></html>"
            error_artifact = Part.from_bytes(
                data=error_html.encode("utf-8"), mime_type="text/html"
            )

            return {
                "success": False,
                "error": str(e),
                "video_player": error_artifact,
                "message": f"❌ Failed to generate video player: {str(e)}",
            }

    def _find_video_file(self, video_id: str) -> Optional[Path]:
        """Find video file by ID in video directory and subdirectories"""
        extensions = ["mp4", "mov", "avi", "mkv", "webm"]

        # Search in main video directory
        for ext in extensions:
            video_path = self.video_dir / f"{video_id}.{ext}"
            if video_path.exists():
                return video_path

        # Search in subdirectories recursively
        for ext in extensions:
            pattern = f"**/{video_id}.{ext}"
            matches = list(self.video_dir.glob(pattern))
            if matches:
                return matches[0]  # Return first match

        return None

    def _create_video_data_url(self, video_path: Path) -> str:
        """Create a data URL for video embedding"""
        import base64

        # Check file size - only embed smaller videos (< 50MB)
        file_size = video_path.stat().st_size
        max_embed_size = 50 * 1024 * 1024  # 50MB

        if file_size > max_embed_size:
            # For large files, return a placeholder or error
            return (
                "data:text/plain;base64,"
                + base64.b64encode(
                    f"Video too large to embed ({file_size / 1024 / 1024:.1f}MB)".encode()
                ).decode()
            )

        # Read video file as bytes
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Determine MIME type based on extension
        video_mime_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }

        mime_type = video_mime_types.get(video_path.suffix.lower(), "video/mp4")

        # Create data URL
        video_base64 = base64.b64encode(video_bytes).decode()
        return f"data:{mime_type};base64,{video_base64}"

    def _create_video_artifact(self, video_path: Path) -> Part:
        """Create a video artifact from the video file"""
        # Read video file as bytes
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Determine MIME type based on extension
        video_mime_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }

        mime_type = video_mime_types.get(video_path.suffix.lower(), "video/mp4")

        # Create artifact with proper MIME type
        return Part.from_bytes(data=video_bytes, mime_type=mime_type)

    def _extract_frame_markers(self, results_data: Any) -> List[Dict[str, Any]]:
        """Extract frame markers from search results"""
        markers = []

        # Handle different result formats
        if isinstance(results_data, dict):
            if "results" in results_data:
                results_list = results_data["results"]
            else:
                results_list = [results_data]
        elif isinstance(results_data, list):
            results_list = results_data
        else:
            return markers

        for result in results_list:
            if isinstance(result, dict):
                # Extract timestamp from various possible fields
                timestamp = None
                score = result.get("score", 0.0)

                # Try different timestamp field names
                for time_field in ["start_time", "timestamp", "time", "frame_time"]:
                    if time_field in result:
                        timestamp = float(result[time_field])
                        break

                if timestamp is not None:
                    # Create marker with description
                    description = result.get(
                        "description", result.get("content", "Search result")
                    )
                    frame_id = result.get("frame_id", result.get("id", len(markers)))

                    markers.append(
                        {
                            "timestamp": timestamp,
                            "score": score,
                            "description": description,
                            "frame_id": frame_id,
                        }
                    )

        # Sort by timestamp
        markers.sort(key=lambda x: x["timestamp"])
        return markers

    def _generate_simple_video_player_html(
        self,
        video_path: Path,
        frame_markers: List[Dict[str, Any]],
        start_time: Optional[float] = None,
    ) -> str:
        """Generate simple HTML that describes the video and frame markers"""

        # Generate frame markers summary
        markers_summary = ""
        if frame_markers:
            markers_list = []
            for marker in frame_markers:
                timestamp = marker["timestamp"]
                score = marker.get("score", 0.0)
                description = marker.get("description", "")

                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                markers_list.append(
                    f"""
                    <div class="marker-item">
                        <strong>{time_str}</strong> (Score: {score:.2f})
                        <br><small>{description}</small>
                    </div>
                """
                )

            markers_summary = f"""
            <div class="markers-section">
                <h4>🎯 Search Result Markers ({len(frame_markers)} frames)</h4>
                {''.join(markers_list)}
            </div>
            """

        start_info = (
            f"<p><strong>Start Time:</strong> {start_time}s</p>" if start_time else ""
        )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Information - {video_path.name}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .video-info {{
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .marker-item {{
                    padding: 10px;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    margin: 10px 0;
                    background-color: #f8f9fa;
                }}
                .alert-info {{
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="video-info">
                    <h2>🎬 {video_path.name}</h2>
                    
                    <div class="alert alert-info">
                        <strong>📁 Video File:</strong> Provided as separate artifact<br>
                        <strong>📊 File Size:</strong> {video_path.stat().st_size / 1024 / 1024:.1f} MB<br>
                        <strong>🎥 Format:</strong> {video_path.suffix.upper()}
                        {start_info}
                    </div>
                    
                    {markers_summary}
                    
                    <div class="alert alert-secondary">
                        <h5>📖 Instructions</h5>
                        <p>This video player provides frame markers and timing information from search results. 
                        The actual video file is provided as a separate artifact that ADK can handle directly.</p>
                        <p>Frame markers show the most relevant moments found by the search system, with timestamps and relevance scores.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _generate_server_based_video_player_html(
        self,
        video_path: Path,
        frame_markers: List[Dict[str, Any]],
        start_time: Optional[float] = None,
    ) -> str:
        """Generate HTML video player that references video via HTTP server (lightweight)"""

        # Get video server port from config
        video_server_port = self.config.get("static_server_port", 8888)

        # Create relative path from video directory
        video_dir = Path(self.config.get("video_dir", "data/videos"))
        try:
            relative_path = video_path.relative_to(video_dir)
        except ValueError:
            # If video is not in video_dir, use filename only
            relative_path = video_path.name

        # Create server URL for video
        video_url = f"http://localhost:{video_server_port}/{relative_path}"

        # Generate timeline markers HTML
        markers_html = ""
        markers_js_data = "[]"

        if frame_markers:
            markers_js_data = json.dumps(frame_markers)

            markers_list = []
            for i, marker in enumerate(frame_markers):
                timestamp = marker["timestamp"]
                score = marker.get("score", 0.0)
                description = marker.get("description", "")

                # Format timestamp
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                # Score color coding
                if score > 0.8:
                    badge_color = "success"
                elif score > 0.6:
                    badge_color = "warning"
                else:
                    badge_color = "secondary"

                markers_list.append(
                    f"""
                    <div class="marker-item" onclick="jumpToTime({timestamp})">
                        <span class="badge bg-{badge_color}">{time_str}</span>
                        <small class="text-muted">Score: {score:.2f}</small>
                        <div class="marker-description">{description[:100]}...</div>
                    </div>
                """
                )

            markers_html = "".join(markers_list)

        start_time_attr = f'data-start-time="{start_time}"' if start_time else ""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Player - {video_path.name}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .video-container {{
                    max-width: 900px;
                    margin: 0 auto;
                }}
                .timeline-markers {{
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    padding: 10px;
                    margin-top: 15px;
                }}
                .marker-item {{
                    cursor: pointer;
                    padding: 8px;
                    border-bottom: 1px solid #eee;
                    transition: background-color 0.2s;
                }}
                .marker-item:hover {{
                    background-color: #f8f9fa;
                }}
                .marker-item:last-child {{
                    border-bottom: none;
                }}
                .marker-description {{
                    margin-top: 5px;
                    font-size: 0.9em;
                }}
                .current-time {{
                    font-family: monospace;
                    font-size: 1.1em;
                    color: #007bff;
                }}
                .video-controls {{
                    margin-top: 10px;
                }}
                .btn-group .btn {{
                    margin-right: 5px;
                }}
                .server-info {{
                    background-color: #e7f3ff;
                    border: 1px solid #b3d9ff;
                    border-radius: 5px;
                    padding: 10px;
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <div class="video-container">
                    <h3>🎬 {video_path.name}</h3>
                    
                    <div class="server-info">
                        <strong>📡 Video Server:</strong> localhost:{video_server_port}<br>
                        <small class="text-muted">Interactive video player with search result markers</small>
                    </div>
                    
                    <video 
                        id="videoPlayer" 
                        controls 
                        width="100%" 
                        {start_time_attr}
                        style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                    >
                        <source src="{video_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    
                    <div class="video-controls">
                        <div class="row">
                            <div class="col-md-6">
                                <strong>Current Time:</strong> 
                                <span id="currentTime" class="current-time">00:00</span>
                            </div>
                            <div class="col-md-6 text-end">
                                <div class="btn-group">
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(0.5)">0.5x</button>
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(1)">1x</button>
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(1.5)">1.5x</button>
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(2)">2x</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {f'''
                    <div class="timeline-markers">
                        <h5>🎯 Search Result Markers ({len(frame_markers)} frames)</h5>
                        {markers_html}
                    </div>
                    ''' if frame_markers else '<div class="alert alert-info mt-3">No search result markers to display</div>'}
                </div>
            </div>
            
            <script>
                const video = document.getElementById('videoPlayer');
                const currentTimeSpan = document.getElementById('currentTime');
                const markers = {markers_js_data};
                
                // Update current time display
                video.addEventListener('timeupdate', function() {{
                    const time = video.currentTime;
                    const minutes = Math.floor(time / 60);
                    const seconds = Math.floor(time % 60);
                    currentTimeSpan.textContent = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;
                }});
                
                // Jump to specific time
                function jumpToTime(timestamp) {{
                    video.currentTime = timestamp;
                    video.play();
                    
                    // Highlight the marker briefly
                    const markers = document.querySelectorAll('.marker-item');
                    markers.forEach(marker => {{
                        if (marker.onclick.toString().includes(timestamp)) {{
                            marker.style.backgroundColor = '#e3f2fd';
                            setTimeout(() => {{
                                marker.style.backgroundColor = '';
                            }}, 1000);
                        }}
                    }});
                }}
                
                // Change playback speed
                function changeSpeed(speed) {{
                    video.playbackRate = speed;
                    
                    // Update button styles
                    document.querySelectorAll('.btn-group .btn').forEach(btn => {{
                        btn.classList.remove('btn-primary');
                        btn.classList.add('btn-outline-primary');
                    }});
                    event.target.classList.remove('btn-outline-primary');
                    event.target.classList.add('btn-primary');
                }}
                
                // Auto-start at specified time if provided
                window.addEventListener('load', function() {{
                    const startTime = video.getAttribute('data-start-time');
                    if (startTime) {{
                        video.currentTime = parseFloat(startTime);
                    }}
                }});
                
                // Keyboard shortcuts
                document.addEventListener('keydown', function(e) {{
                    if (e.target.tagName.toLowerCase() !== 'input') {{
                        switch(e.key) {{
                            case ' ':
                                e.preventDefault();
                                if (video.paused) {{
                                    video.play();
                                }} else {{
                                    video.pause();
                                }}
                                break;
                            case 'ArrowLeft':
                                video.currentTime -= 5;
                                break;
                            case 'ArrowRight':
                                video.currentTime += 5;
                                break;
                            case 'ArrowUp':
                                video.volume = Math.min(1, video.volume + 0.1);
                                break;
                            case 'ArrowDown':
                                video.volume = Math.max(0, video.volume - 0.1);
                                break;
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return html

    def _generate_video_player_html(
        self,
        video_path: Path,
        frame_markers: List[Dict[str, Any]],
        start_time: Optional[float] = None,
    ) -> str:
        """Generate HTML video player with timeline markers"""

        # Create data URL for video embedding
        video_data_url = self._create_video_data_url(video_path)

        # Generate timeline markers HTML
        markers_html = ""
        markers_js_data = "[]"

        if frame_markers:
            markers_js_data = json.dumps(frame_markers)

            markers_list = []
            for i, marker in enumerate(frame_markers):
                timestamp = marker["timestamp"]
                score = marker.get("score", 0.0)
                description = marker.get("description", "")

                # Format timestamp
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                # Score color coding
                if score > 0.8:
                    badge_color = "success"
                elif score > 0.6:
                    badge_color = "warning"
                else:
                    badge_color = "secondary"

                markers_list.append(
                    f"""
                    <div class="marker-item" onclick="jumpToTime({timestamp})">
                        <span class="badge bg-{badge_color}">{time_str}</span>
                        <small class="text-muted">Score: {score:.2f}</small>
                        <div class="marker-description">{description[:100]}...</div>
                    </div>
                """
                )

            markers_html = "".join(markers_list)

        start_time_attr = f'data-start-time="{start_time}"' if start_time else ""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Player - {video_path.name}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .video-container {{
                    max-width: 900px;
                    margin: 0 auto;
                }}
                .timeline-markers {{
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    padding: 10px;
                    margin-top: 15px;
                }}
                .marker-item {{
                    cursor: pointer;
                    padding: 8px;
                    border-bottom: 1px solid #eee;
                    transition: background-color 0.2s;
                }}
                .marker-item:hover {{
                    background-color: #f8f9fa;
                }}
                .marker-item:last-child {{
                    border-bottom: none;
                }}
                .marker-description {{
                    margin-top: 5px;
                    font-size: 0.9em;
                }}
                .current-time {{
                    font-family: monospace;
                    font-size: 1.1em;
                    color: #007bff;
                }}
                .video-controls {{
                    margin-top: 10px;
                }}
                .btn-group .btn {{
                    margin-right: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <div class="video-container">
                    <h3>🎬 {video_path.name}</h3>
                    
                    <video 
                        id="videoPlayer" 
                        controls 
                        width="100%" 
                        {start_time_attr}
                        style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                    >
                        <source src="{video_data_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    
                    <div class="video-controls">
                        <div class="row">
                            <div class="col-md-6">
                                <strong>Current Time:</strong> 
                                <span id="currentTime" class="current-time">00:00</span>
                            </div>
                            <div class="col-md-6 text-end">
                                <div class="btn-group">
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(0.5)">0.5x</button>
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(1)">1x</button>
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(1.5)">1.5x</button>
                                    <button class="btn btn-sm btn-outline-primary" onclick="changeSpeed(2)">2x</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {f'''
                    <div class="timeline-markers">
                        <h5>🎯 Search Result Markers ({len(frame_markers)} frames)</h5>
                        {markers_html}
                    </div>
                    ''' if frame_markers else '<div class="alert alert-info mt-3">No search result markers to display</div>'}
                </div>
            </div>
            
            <script>
                const video = document.getElementById('videoPlayer');
                const currentTimeSpan = document.getElementById('currentTime');
                const markers = {markers_js_data};
                
                // Update current time display
                video.addEventListener('timeupdate', function() {{
                    const time = video.currentTime;
                    const minutes = Math.floor(time / 60);
                    const seconds = Math.floor(time % 60);
                    currentTimeSpan.textContent = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;
                }});
                
                // Jump to specific time
                function jumpToTime(timestamp) {{
                    video.currentTime = timestamp;
                    video.play();
                    
                    // Highlight the marker briefly
                    const markers = document.querySelectorAll('.marker-item');
                    markers.forEach(marker => {{
                        if (marker.onclick.toString().includes(timestamp)) {{
                            marker.style.backgroundColor = '#e3f2fd';
                            setTimeout(() => {{
                                marker.style.backgroundColor = '';
                            }}, 1000);
                        }}
                    }});
                }}
                
                // Change playback speed
                function changeSpeed(speed) {{
                    video.playbackRate = speed;
                    
                    // Update button styles
                    document.querySelectorAll('.btn-group .btn').forEach(btn => {{
                        btn.classList.remove('btn-primary');
                        btn.classList.add('btn-outline-primary');
                    }});
                    event.target.classList.remove('btn-outline-primary');
                    event.target.classList.add('btn-primary');
                }}
                
                // Auto-start at specified time if provided
                window.addEventListener('load', function() {{
                    const startTime = video.getAttribute('data-start-time');
                    if (startTime) {{
                        video.currentTime = parseFloat(startTime);
                    }}
                }});
                
                // Keyboard shortcuts
                document.addEventListener('keydown', function(e) {{
                    if (e.target.tagName.toLowerCase() !== 'input') {{
                        switch(e.key) {{
                            case ' ':
                                e.preventDefault();
                                if (video.paused) {{
                                    video.play();
                                }} else {{
                                    video.pause();
                                }}
                                break;
                            case 'ArrowLeft':
                                video.currentTime -= 5;
                                break;
                            case 'ArrowRight':
                                video.currentTime += 5;
                                break;
                            case 'ArrowUp':
                                video.volume = Math.min(1, video.volume + 0.1);
                                break;
                            case 'ArrowDown':
                                video.volume = Math.max(0, video.volume - 0.1);
                                break;
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return html
