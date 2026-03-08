from __future__ import annotations

import logging
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def experiment_log_context(
    output_dir: Path,
    log_prefix: str,
    *,
    configure_root_logger: bool = False,
) -> Iterator[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{log_prefix}_{timestamp}.log"

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)

        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            if configure_root_logger:
                root_logger.handlers.clear()
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
                root_logger.addHandler(handler)
                root_logger.setLevel(logging.INFO)

            print(f"[log] Writing log to {log_path}")
            try:
                yield log_path
            finally:
                if configure_root_logger:
                    root_logger.handlers.clear()
                    root_logger.handlers.extend(original_handlers)
                    root_logger.setLevel(original_level)


def run_and_stream(command: list[str], *, env: dict[str, str] | None = None) -> None:
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    ) as process:
        assert process.stdout is not None
        for chunk in process.stdout:
            print(chunk, end="")

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
