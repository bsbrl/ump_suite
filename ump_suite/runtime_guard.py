"""Prevent duplicate local rig processes from sharing devices or output files."""

import atexit
import fcntl
import os
from pathlib import Path


_LOCKS = {}


def acquire_process_lock(resource):
    """Hold an exclusive per-user resource lock until this process exits."""
    if not resource or any(c not in 'abcdefghijklmnopqrstuvwxyz_0123456789' for c in resource):
        raise ValueError(f'Invalid resource name: {resource!r}')
    if resource in _LOCKS:
        raise RuntimeError(f'{resource} already acquired by this process')
    directory = Path('/tmp') / f'ump_suite_{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise RuntimeError(f'Unsafe rig lock directory: {directory}')
    path = directory / f'{resource}.lock'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner = os.read(fd, 128).decode(errors='replace').strip()
        os.close(fd)
        raise RuntimeError(
            f'{resource} is already running ({owner or "another process"}). '
            'Close the existing rig session before launching another.'
        ) from None
    except BaseException:
        os.close(fd)
        raise
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f'pid={os.getpid()}\n'.encode())
    except BaseException:
        os.close(fd)
        raise
    _LOCKS[resource] = fd
    atexit.register(release_process_lock, resource)


def release_process_lock(resource):
    """Release a held lock; keep its inode so another opener cannot bypass it."""
    fd = _LOCKS.pop(resource, None)
    if fd is not None:
        os.close(fd)
