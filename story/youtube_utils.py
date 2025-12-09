"""
YouTube video downloading and URL handling utilities.
"""
import os
import tempfile
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

def download_youtube_video(url: str) -> str:
    """
    Download YouTube video using yt-dlp with enhanced configuration
    
    Args:
        url: YouTube URL to download
        
    Returns:
        Path to the downloaded video file
    """
    if not yt_dlp:
        raise ImportError("yt-dlp is required for YouTube support. Install with: pip install yt-dlp")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'merge_output_format': 'mp4',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        raise Exception(f"Error downloading YouTube video: {str(e)}")

def is_youtube_url(url: str) -> bool:
    """
    Check if the given string is a YouTube URL
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if it's a YouTube URL, False otherwise
    """
    if not url:
        return False
    
    youtube_domains = [
        'youtube.com',
        'www.youtube.com',
        'm.youtube.com',
        'youtu.be',
        'www.youtu.be'
    ]
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
            
        # Check domain
        domain = parsed.netloc.lower()
        if any(youtube_domain in domain for youtube_domain in youtube_domains):
            return True
            
        # Check for youtu.be short URL
        if 'youtu.be' in domain:
            return True
            
        # Check for YouTube video ID in query params
        if 'youtube.com' in domain:
            query = parse_qs(parsed.query)
            return 'v' in query
            
    except Exception:
        pass
        
    return False
