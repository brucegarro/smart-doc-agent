"""Entrypoint for Redis-backed ingestion workers."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
from typing import Iterable

from redis import Redis
from rq import Connection, Worker

from agent.config import settings

LOGGER = logging.getLogger(__name__)


def _worker_process(name: str, queue_names: Iterable[str]) -> None:
    redis_conn = Redis.from_url(settings.redis_url)
    with Connection(redis_conn):
        worker = Worker(list(queue_names), name=name)
        LOGGER.info("Worker %s starting; queues=%s", name, queue_names)
        worker.work(with_scheduler=True)


def _spawn_worker_pool(concurrency: int, queue_names: Iterable[str]) -> None:
    processes: list[mp.Process] = []

    def _terminate_children(signum, frame):  # pragma: no cover - signal handler
        LOGGER.info("Received signal %s, terminating workers", signum)
        for proc in processes:
            if proc.is_alive():
                proc.terminate()

    signal.signal(signal.SIGTERM, _terminate_children)
    signal.signal(signal.SIGINT, _terminate_children)

    try:
        for idx in range(concurrency):
            name = f"ingest-{idx + 1}"
            proc = mp.Process(target=_worker_process, args=(name, tuple(queue_names)), daemon=False)
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()


def _configure_logging() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))


def main() -> None:
    _configure_logging()
    queue_names = [settings.redis_queue_ingest]
    concurrency = max(1, int(os.environ.get("WORKER_CONCURRENCY", "1")))

    LOGGER.info("Starting worker with concurrency=%s on queues=%s", concurrency, queue_names)

    if concurrency == 1:
        _worker_process("ingest-1", queue_names)
        return

    _spawn_worker_pool(concurrency, queue_names)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - graceful shutdown
        sys.exit(0)
