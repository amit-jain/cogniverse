#!/usr/bin/env python3
"""
Video File Server

A simple HTTP server for serving video files to the video player tool.
This server allows the video player to reference videos via HTTP URLs
instead of embedding large video data directly in HTML artifacts.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cogniverse_foundation.config.utils import create_default_config_manager, get_config

if TYPE_CHECKING:
    from cogniverse_foundation.config.system_config import SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoFileServer:
    """HTTP server for serving video files"""

    def __init__(self, port: int = 8888, config: Optional["SystemConfig"] = None):
        """Initialize video file server

        Args:
            port: Port number to serve on
            config: SystemConfig instance (required for dependency injection)
        """
        self.port = port
        if config is None:
            raise ValueError(
                "config is required for VideoFileServer. "
                "Pass SystemConfig instance explicitly."
            )

        self.config = config
        self.video_dir = Path(self.config.get("video_dir", "data/videos"))
        self.app = FastAPI(title="Video File Server", version="1.0.0")

        # Enable CORS for cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup HTTP routes for serving video files"""

        @self.app.get("/")
        async def root():
            """Root endpoint showing available videos"""
            return {
                "message": "Video File Server",
                "video_dir": str(self.video_dir),
                "port": self.port,
                "available_videos": self._get_available_videos(),
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {"status": "healthy", "video_dir_exists": self.video_dir.exists()}

        @self.app.get("/videos")
        async def list_videos():
            """List all available video files"""
            return {
                "videos": self._get_available_videos(),
                "total_count": len(self._get_available_videos()),
            }

        @self.app.get("/player/{video_id}")
        async def video_player(
            video_id: str, search_results: str = None, start_time: float = None
        ):
            """Serve video player HTML for a specific video with search results"""
            from cogniverse_agents.tools.video_player_tool import VideoPlayerTool

            player_tool = VideoPlayerTool()
            result = await player_tool.execute(
                video_id=video_id, search_results=search_results, start_time=start_time
            )

            if result.get("success"):
                # Extract HTML content from the artifact
                html_content = result["video_player"].inline_data.data.decode("utf-8")
                from fastapi.responses import HTMLResponse

                return HTMLResponse(content=html_content)
            else:
                from fastapi.responses import HTMLResponse

                error_html = f"<html><body><h1>Error</h1><p>{result.get('error', 'Unknown error')}</p></body></html>"
                return HTMLResponse(content=error_html, status_code=404)

        # Serve video files directly from the video directory
        if self.video_dir.exists():
            # Mount the video directory as static files
            self.app.mount(
                "/", StaticFiles(directory=str(self.video_dir)), name="videos"
            )
        else:
            print(f"⚠️  Video directory not found: {self.video_dir}")

    def _get_available_videos(self) -> list:
        """Get list of available video files"""
        if not self.video_dir.exists():
            return []

        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        videos = []

        for ext in video_extensions:
            videos.extend(
                [
                    str(p.relative_to(self.video_dir))
                    for p in self.video_dir.rglob(f"*{ext}")
                ]
            )

        return sorted(videos)

    async def start(self):
        """Start the video file server"""
        logger.info(f"🎬 Starting Video File Server on port {self.port}")
        logger.info(f"📁 Serving videos from: {self.video_dir}")

        if not self.video_dir.exists():
            logger.error(f"❌ Video directory not found: {self.video_dir}")
            logger.error(
                "   Please ensure the video directory exists before starting the server"
            )
            return False

        videos = self._get_available_videos()
        logger.info(f"📼 Found {len(videos)} video files")

        # Start the server
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="info"
        )

        server = uvicorn.Server(config)
        await server.serve()

        return True


async def main():
    """Main function to start the video file server"""
    config = get_config(
        tenant_id="default", config_manager=create_default_config_manager()
    )
    port = config.get("static_server_port", 8888)

    server = VideoFileServer(port=port, config=config)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
