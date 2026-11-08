"""Shutdown handling shared by the startup scripts, so Ctrl+C, a GUI's Stop and a vanished GUI all unwind the same way."""

from __future__ import annotations

import _thread
import os
import signal
import sys
import threading
import time

STOP_SIGNALS = ("SIGTERM", "SIGHUP", "SIGBREAK")


def add_lifecycle_args(parser) -> None:
    parser.add_argument("--parent-pid", type=int, default=None, help="Stop when this process (e.g. the GUI) exits.")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Only stop processes a previous run of this config left behind, then exit.",
    )


def install_shutdown_handlers(parent_pid: int | None = None) -> None:
    """Turn stop signals into one KeyboardInterrupt, and raise one if parent_pid exits.

    The ProcessManager context then stops every child in order. Repeated signals for the
    same stop (a GUI and `uv run` forwarding its copy) are ignored, so they cannot
    interrupt that cleanup half way.
    """
    requested = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        if requested.is_set():
            return
        requested.set()
        raise KeyboardInterrupt

    handlers_installed = threading.current_thread() is threading.main_thread()
    if handlers_installed:
        for name in STOP_SIGNALS:
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), request_stop)

    if parent_pid:
        parent_gone = threading.Lock()

        def stop_for_parent() -> None:
            # The parent was reading our output; with it gone every print would raise
            # BrokenPipeError and abort the shutdown, so send output nowhere first.
            if requested.is_set() or not parent_gone.acquire(blocking=False):
                return
            detach_output()
            if handlers_installed and os.name != "nt":
                os.kill(os.getpid(), signal.SIGTERM)  # a real signal also wakes a blocking call
            else:
                requested.set()
                _thread.interrupt_main()

        def watch() -> None:
            while process_alive(parent_pid):
                time.sleep(1.0)
            stop_for_parent()

        sys.stdout = _PipeGuard(sys.stdout, stop_for_parent)
        sys.stderr = _PipeGuard(sys.stderr, stop_for_parent)
        threading.Thread(target=watch, name="parent-watchdog", daemon=True).start()


class _PipeGuard:
    """A stream whose writes never raise BrokenPipeError: a closed reader means the parent is gone."""

    def __init__(self, stream, on_broken):
        self._stream, self._on_broken = stream, on_broken

    def write(self, text):
        try:
            return self._stream.write(text)
        except BrokenPipeError:
            self._on_broken()
            return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except BrokenPipeError:
            self._on_broken()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def detach_output() -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    for fd in (1, 2):
        try:
            os.dup2(devnull, fd)
        except OSError:
            pass
    os.close(devnull)


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
