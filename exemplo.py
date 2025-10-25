#!/usr/bin/env python3
"""
Example usage and configuration for HTTPFS

python httpfs.py https://cosmo.zip/pub/cosmos/bin/ /tmp/cosmos --cache-dir /tmp/cosmos_cache --foreground --debug
python httpfs.py https://cosmo.zip/pub/cosmos/bin/ /tmp/cosmos --cache-dir /tmp/cosmos_cache --parser nginx --foreground --debug
python httpfs.py [-h] [--cache-dir CACHE_DIR] [--cache-ttl CACHE_TTL] [--parser {auto,apache,nginx}] [--foreground] [--debug] base_url mount_point
sh -c "echo ':APE:M::MZqFpD::/tmp/cosmos/ape-x86_64.elf:' >/proc/sys/fs/binfmt_misc/register"
"""

import os
import tempfile

# Example configuration
CONFIG = {
    'base_url': 'https://cosmo.zip/pub/cosmos/bin/',
    'mount_point': '/tmp/cosmos',
    'cache_dir': tempfile.mkdtemp(prefix='httpfs_cosmos_'),
    'cache_ttl': 7200,  # 2 hours
    'parser_type': 'auto',
    'foreground': False,
    'debug': False
}

def mount_example():
    """Example mounting function"""
    from httpfs import HTTPFileSystem, ParserType
    import fuse
    
    fs = HTTPFileSystem(
        base_url=CONFIG['base_url'],
        cache_dir=CONFIG['cache_dir'],
        parser_type=ParserType(CONFIG['parser_type']),
        cache_ttl=CONFIG['cache_ttl']
    )
    
    # Ensure mount point exists
    os.makedirs(CONFIG['mount_point'], exist_ok=True)
    
    fuse.FUSE(fs, CONFIG['mount_point'], nothreads=True, 
              foreground=CONFIG['foreground'], ro=True, 
              fsname='httpfs_cosmos')

if __name__ == '__main__':
    print("HTTPFS Configuration Example")
    print(f"URL: {CONFIG['base_url']}")
    print(f"Mount: {CONFIG['mount_point']}")
    print(f"Cache: {CONFIG['cache_dir']}")
    
    # You would typically run: python -m httpfs URL MOUNT_POINT
