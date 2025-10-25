#!/usr/bin/env python3

"""
HTTP Directory FUSE Filesystem - VERSION COMPATÍVEL
Mounts Apache/Nginx directory listings as a local filesystem
"""

import os
import sys
import time
import logging
import requests
import tempfile
import hashlib
import errno
import stat
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from urllib.parse import urljoin, unquote

# Configuração de compatibilidade FUSE
try:
    import fuse
    from fuse import FuseOSError, Operations
    FUSE_AVAILABLE = True
    FUSE_LIB = 'fuse'
    # Constantes para o módulo fuse
    S_IFDIR = stat.S_IFDIR
    S_IFREG = stat.S_IFREG
except ImportError:
    try:
        import fusepy as fuse
        from fusepy import FuseOSError, Operations
        FUSE_AVAILABLE = True
        FUSE_LIB = 'fusepy'
        # Constantes para o módulo fusepy (usando stat)
        S_IFDIR = stat.S_IFDIR
        S_IFREG = stat.S_IFREG
    except ImportError:
        FUSE_AVAILABLE = False
        FUSE_LIB = None
        print("Error: Neither 'fuse' nor 'fusepy' package is available.")
        print("Please install one of them:")
        print("  sudo apt-get install fuse python3-fuse")
        print("  OR")
        print("  pip install fusepy")
        sys.exit(1)

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Error: BeautifulSoup4 is required but not available.")
    print("Please install it:")
    print("  pip install beautifulsoup4")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('httpfs')


class EntryType(Enum):
    FILE = "file"
    DIRECTORY = "directory"


@dataclass
class CacheEntry:
    path: str
    entry_type: EntryType
    size: int
    mtime: float
    content_path: Optional[str] = None
    children: Optional[List[str]] = None
    last_verified: float = None
    downloaded: bool = False

    def __post_init__(self):
        if self.last_verified is None:
            self.last_verified = time.time()


class ParserType(Enum):
    APACHE = "apache"
    NGINX = "nginx"
    AUTO = "auto"


class HTTPDirectoryListing:
    pass


class HTTPDirectoryListing:
    """Parser for apache directory listing"""

    def parse_directory_listing(self, html_content: str) -> List[Tuple[str, EntryType, int]]:
        pass

    def parse_listing(parser_type: ParserType, server_header: str, html_content: str) -> List[Tuple[str, EntryType, int]]:
        if parser_type == ParserType.AUTO:
            parser_type = HTTPDirectoryListing._detect_parser_type(server_header, html_content)

        logger.debug(f"Detected parser type: {parser_type}")
        listing_parser = HTTPDirectoryListing._getListingParser(parser_type)
        return listing_parser.parse_directory_listing(html_content)

    def _getListingParser(parser_type: ParserType) -> HTTPDirectoryListing:
        if parser_type == ParserType.APACHE:
            return HTTPDirectoryListingApache()
        elif parser_type == ParserType.NGINX:
            return HTTPDirectoryListingNginx()

        return HTTPDirectoryListingNginx()  # Default fallback

    def _detect_parser_type(server_header: str, html_content: str) -> ParserType:
        """Auto-detect the directory listing format"""
        server_lower = server_header.lower()
        if 'apache' in server_lower:
            return ParserType.APACHE
        elif 'nginx' in server_lower:
            return ParserType.NGINX

        html_lower = html_content.lower()
        if 'apache' in html_lower or 'index of' in html_lower:
            return ParserType.APACHE
        elif 'nginx' in html_lower:
            return ParserType.NGINX

        return ParserType.NGINX

    def _parse_size(self, size_str: str) -> int:
        """Parse human-readable size string to bytes"""
        size_str = size_str.upper().strip()
        multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}

        # Remove any non-alphanumeric characters except decimal point
        size_str = re.sub(r'[^\d.KMGT]', '', size_str)

        if size_str.isdigit():
            return int(size_str)

        # Handle decimal numbers (e.g., 1.5M)
        for unit, multiplier in multipliers.items():
            if size_str.endswith(unit):
                num_str = size_str[:-1]
                try:
                    num = float(num_str)
                    return int(num * multiplier)
                except ValueError:
                    pass

        # If all else fails, try to extract just the number part
        match = re.search(r'(\d+(?:\.\d+)?)', size_str)
        if match:
            return int(float(match.group(1)))

        raise ValueError(f"Unable to parse size: {size_str}")


class HTTPDirectoryListingApache(HTTPDirectoryListing):
    """Parser for apache directory listing"""

    def parse_directory_listing(self, html_content: str) -> List[Tuple[str, EntryType, int]]:
        """Parse Apache-style directory listing"""
        entries = []
        soup = BeautifulSoup(html_content, 'html.parser')

        for link in soup.find_all('a'):
            href = link.get('href', '')
            text = link.get_text().strip()

            if not href or href == '../' or not text or text.lower() == 'parent directory' or link.find_parent('th'):
                continue

            if href.endswith('/'):
                entry_type = EntryType.DIRECTORY
                name = href.rstrip('/')
                size = 4096
            else:
                entry_type = EntryType.FILE
                name = href
                size = 0

            parent_row = link.find_parent('tr')
            if parent_row:
                cells = parent_row.find_all('td')
                if len(cells) >= 4:
                    size_str = cells[3].get_text().strip()
                    if size_str and size_str != '-' and size_str != '':
                        try:
                            size = self._parse_size(size_str)
                        except ValueError:
                            # If we can't parse size, keep the default
                            pass

            if name and name != '../':
                name = unquote(name)
                entries.append((name, entry_type, size))

        return entries


class HTTPDirectoryListingNginx(HTTPDirectoryListing):
    """Parser for nginx directory listing"""

    def parse_directory_listing(self, html_content: str) -> List[Tuple[str, EntryType, int]]:
        """Parse Nginx-style directory listing"""
        entries = []
        soup = BeautifulSoup(html_content, 'html.parser')

        for link in soup.find_all('a'):
            href = link.get('href', '')
            text = link.get_text().strip()

            if not href or href == '../' or not text or text.lower() == 'parent directory':
                continue

            if href.endswith('/'):
                entry_type = EntryType.DIRECTORY
                name = href.rstrip('/')
                size = 4096
            else:
                entry_type = EntryType.FILE
                name = href
                size = self._parse_size(link.next_sibling.split().pop())
                # logger.debug(f" ======= {size}")

            if name and name != '../':
                name = unquote(name)
                entries.append((name, entry_type, size))

        return entries


class HTTPDirectoryParser:
    """Parser for different directory listing formats"""

    def __init__(self, parser_type: ParserType = ParserType.AUTO):
        self.parser_type = parser_type

    def parse_directory_listing(self, server_header: str, html_content: str) -> List[Tuple[str, EntryType, int]]:
        """Parse directory listing and return entries (name, type, size)"""

        return HTTPDirectoryListing.parse_listing(self.parser_type, server_header, html_content)


class HTTPFSCache:
    """Cache management for HTTP filesystem"""

    def __init__(self, cache_dir: str, default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.content_dir = self.cache_dir / "content"
        self.metadata_dir = self.cache_dir / "metadata"
        self.default_ttl = default_ttl

        # Create cache directories
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.entries: Dict[str, CacheEntry] = {}

        logger.info(f"Cache initialized at {self.cache_dir}")
        logger.info(f"Using FUSE library: {FUSE_LIB}")

    def get_entry(self, path: str) -> Optional[CacheEntry]:
        """Get cache entry if it exists and is not expired"""
        entry = self.entries.get(path)
        if entry and time.time() - entry.last_verified < self.default_ttl:
            return entry
        return None

    def add_entry(self, path: str, entry_type: EntryType, size: int = 0,
                  content_path: Optional[str] = None, downloaded: bool = False) -> CacheEntry:
        """Add new cache entry"""
        entry = CacheEntry(
            path=path,
            entry_type=entry_type,
            size=size,
            mtime=time.time(),
            content_path=content_path,
            children=[] if entry_type == EntryType.DIRECTORY else None,
            last_verified=time.time(),
            downloaded=downloaded
        )
        self.entries[path] = entry
        logger.debug(f"Added cache entry: {path} ({entry_type.value})")
        return entry

    def update_entry_verification(self, path: str):
        """Update last verification time"""
        if path in self.entries:
            self.entries[path].last_verified = time.time()

    def get_expired_entry(self, path: str) -> Optional[CacheEntry]:
        """Get cache entry even if expired (for offline use)"""
        return self.entries.get(path)

    def mark_file_downloaded(self, path: str, content_path: str, size: int):
        """Mark a file as successfully downloaded"""
        if path in self.entries:
            self.entries[path].content_path = content_path
            self.entries[path].size = size
            self.entries[path].downloaded = True
            self.entries[path].last_verified = time.time()
            self.entries[path].mtime = time.time()
            logger.info(f"Marked file as downloaded: {path} ({size} bytes)")


class HTTPFileSystem(Operations):
    """HTTP Directory FUSE Filesystem"""

    def __init__(self, base_url: str, cache_dir: str = None,
                 parser_type: ParserType = ParserType.AUTO,
                 cache_ttl: int = 3600):
        self.base_url = base_url.rstrip('/') + '/'
        self.parser = HTTPDirectoryParser(parser_type)
        self.session = requests.Session()

        # Set user agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'HTTPFS/1.0'
        })

        # Set up cache
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="httpfs_")
        self.cache = HTTPFSCache(cache_dir, cache_ttl)

        # Add root directory to cache
        self.cache.add_entry('/', EntryType.DIRECTORY, 4096)

        logger.info(f"HTTPFS initialized with base URL: {self.base_url}")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"FUSE library: {FUSE_LIB}")

    def _fetch_url(self, url: str) -> Optional[requests.Response]:
        """Fetch URL with error handling"""
        try:
            logger.debug(f"Fetching URL: {url}")
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _download_file(self, path: str) -> bool:
        """Download and cache a file"""
        relative_path = path.lstrip('/')
        url = urljoin(self.base_url, relative_path)

        logger.info(f"Downloading file: {url}")
        response = self._fetch_url(url)
        if not response:
            logger.error(f"Failed to download file: {url}")
            return False

        # Create unique cache filename
        content_hash = hashlib.md5(path.encode()).hexdigest()
        content_path = self.cache.content_dir / content_hash

        try:
            # Write content to cache file
            with open(content_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = content_path.stat().st_size

            # Update cache entry
            self.cache.mark_file_downloaded(path, str(content_path), file_size)

            logger.info(f"Successfully downloaded and cached: {path} ({file_size} bytes)")
            return True

        except IOError as e:
            logger.error(f"Failed to write cache file {content_path}: {e}")
            # Clean up failed download
            if content_path.exists():
                content_path.unlink()
            return False

    def _get_directory_listing(self, path: str) -> List[CacheEntry]:
        """Get directory listing from server or cache"""
        # Build URL for this directory
        relative_path = path.lstrip('/')
        url = urljoin(self.base_url, relative_path)
        if not url.endswith('/'):
            url += '/'

        # Try to fetch fresh listing
        response = self._fetch_url(url)
        if response:
            entries_data = self.parser.parse_directory_listing(response.headers.get("Server"), response.text)
            cached_entries = []

            for name, entry_type, size in entries_data:
                entry_path = os.path.join(path, name).replace('//', '/')

                # Create or update cache entry
                if entry_path in self.cache.entries:
                    entry = self.cache.entries[entry_path]
                    if not entry.downloaded:
                        entry.size = size
                        entry.last_verified = time.time()
                else:
                    entry = self.cache.add_entry(entry_path, entry_type, size)

                cached_entries.append(entry)

            # Update parent directory's children list
            parent_entry = self.cache.entries.get(path)
            if parent_entry:
                parent_entry.children = [e.path for e in cached_entries]
                parent_entry.last_verified = time.time()

            logger.info(f"Refreshed directory listing for {path}: {len(cached_entries)} entries")
            return cached_entries
        else:
            # Server offline, try to use cached entries
            logger.warning(f"Server offline, using cached data for {path}")
            parent_entry = self.cache.get_expired_entry(path)
            if parent_entry and parent_entry.children:
                children_entries = []
                for child_path in parent_entry.children:
                    if child_path in self.cache.entries:
                        children_entries.append(self.cache.entries[child_path])
                return children_entries

            raise FuseOSError(errno.ENOENT)

    def getattr(self, path: str, fh: Any = None) -> Dict[str, Any]:
        """Get file attributes"""
        logger.debug(f"getattr: {path}")

        # Check cache first
        entry = self.cache.get_entry(path)
        if not entry:
            # Try expired cache for offline use
            entry = self.cache.get_expired_entry(path)
            if not entry:
                # If it's a directory that we haven't cached yet, try to fetch it
                if path != '/':
                    parent_dir = os.path.dirname(path)
                    if parent_dir == '':
                        parent_dir = '/'

                    # This will trigger a directory listing fetch
                    try:
                        self.readdir(parent_dir, None)
                        entry = self.cache.get_expired_entry(path)
                    except FuseOSError:
                        pass

                if not entry:
                    raise FuseOSError(errno.ENOENT)

        # Build stat result - USANDO CONSTANTES COMPATÍVEIS
        if entry.entry_type == EntryType.DIRECTORY:
            mode = 0o755 | S_IFDIR
            size = 4096
        else:
            mode = 0o644 | stat.S_IXUSR | S_IFREG
            size = entry.size

        return {
            'st_mode': mode,
            'st_size': size,
            'st_ctime': entry.mtime,
            'st_mtime': entry.mtime,
            'st_atime': time.time(),
            'st_nlink': 2 if entry.entry_type == EntryType.DIRECTORY else 1,
            'st_uid': os.getuid(),
            'st_gid': os.getgid()
        }

    def readdir(self, path: str, fh: Any) -> List[str]:
        """Read directory contents"""
        logger.debug(f"readdir: {path}")

        entries = ['.', '..']

        # Get directory listing
        cached_entries = self._get_directory_listing(path)

        # Add entry names to result
        for entry in cached_entries:
            entries.append(os.path.basename(entry.path))

        logger.debug(f"readdir {path}: returning {len(entries)} entries")
        return entries

    def open(self, path: str, flags: int) -> int:
        """Open file"""
        logger.debug(f"open: {path}, flags: {flags}")

        # Only support read operations
        if flags & os.O_WRONLY or flags & os.O_RDWR:
            raise FuseOSError(errno.EROFS)

        entry = self.cache.get_expired_entry(path)
        if not entry or entry.entry_type != EntryType.FILE:
            raise FuseOSError(errno.ENOENT)

        # If file content is not cached, download it
        if not entry.downloaded or not entry.content_path or not os.path.exists(entry.content_path):
            logger.info(f"File not cached, downloading: {path}")
            if not self._download_file(path):
                raise FuseOSError(errno.EIO)

        # Verify the cached file exists and is readable
        if not os.path.exists(entry.content_path):
            logger.error(f"Cached file missing: {entry.content_path}")
            # Try to re-download
            if not self._download_file(path):
                raise FuseOSError(errno.EIO)

        # Verify file permissions and readability
        try:
            with open(entry.content_path, 'rb') as f:
                # Just test if we can read the file
                f.read(1)
        except IOError as e:
            logger.error(f"Cannot read cached file {entry.content_path}: {e}")
            # Try to re-download
            if not self._download_file(path):
                raise FuseOSError(errno.EIO)

        return 0

    def read(self, path: str, size: int, offset: int, fh: Any) -> bytes:
        """Read file content"""
        logger.debug(f"read: {path}, size: {size}, offset: {offset}")

        entry = self.cache.get_expired_entry(path)
        if not entry or not entry.content_path:
            logger.error(f"No cache entry or content path for: {path}")
            raise FuseOSError(errno.ENOENT)

        # Verify file exists
        if not os.path.exists(entry.content_path):
            logger.error(f"Cached file not found: {entry.content_path}")
            raise FuseOSError(errno.ENOENT)

        try:
            with open(entry.content_path, 'rb') as f:
                f.seek(offset)
                data = f.read(size)
                bytes_read = len(data)
                logger.debug(f"Read {bytes_read} bytes from {path} (offset: {offset}, requested: {size})")
                return data
        except IOError as e:
            logger.error(f"Failed to read cached file {entry.content_path}: {e}")
            raise FuseOSError(errno.EIO)

    def statfs(self, path: str) -> Dict[str, Any]:
        """Get filesystem statistics"""
        return {
            'f_bsize': 4096,
            'f_frsize': 4096,
            'f_blocks': 1000000,
            'f_bfree': 500000,
            'f_bavail': 500000,
            'f_files': 100000,
            'f_ffree': 50000,
            'f_favail': 50000,
            'f_flag': 0,
            'f_namemax': 255
        }


def main():
    import argparse

    if not FUSE_AVAILABLE:
        print("FUSE not available. Please install fuse-python or fusepy.")
        sys.exit(1)

    if not BEAUTIFULSOUP_AVAILABLE:
        print("BeautifulSoup4 not available. Please install beautifulsoup4.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='HTTP Directory FUSE Filesystem')
    parser.add_argument('base_url', help='Base HTTP URL to mount')
    parser.add_argument('mount_point', help='Mount point directory')
    parser.add_argument('--cache-dir', help='Cache directory')
    parser.add_argument('--cache-ttl', type=int, default=3600,
                        help='Cache TTL in seconds (default: 3600)')
    parser.add_argument('--parser', choices=['auto', 'apache', 'nginx'],
                        default='auto', help='Directory listing parser type')
    parser.add_argument('--foreground', action='store_true',
                        help='Run in foreground')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('urllib3').setLevel(logging.INFO)

    # Validate URL
    if not args.base_url.startswith(('http://', 'https://')):
        print("Error: base_url must start with http:// or https://", file=sys.stderr)
        sys.exit(1)

    # Create filesystem instance
    parser_type = ParserType(args.parser)
    fs = HTTPFileSystem(
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        parser_type=parser_type,
        cache_ttl=args.cache_ttl
    )

    logger.info(f"Mounting {args.base_url} at {args.mount_point}")
    logger.info(f"Parser type: {parser_type}")

    try:
        # Mount the filesystem
        fuse.FUSE(
            fs,
            args.mount_point,
            nothreads=True,
            foreground=args.foreground,
            ro=True,
            fsname=f'httpfs::{args.base_url}'
        )
    except Exception as e:
        logger.error(f"Failed to mount filesystem: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
