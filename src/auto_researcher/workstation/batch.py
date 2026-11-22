"""Batch mode: many memos at once through the Message Batches API, at half price.

A memo is a chain (plan, then the two filing reads, then drafts), so batching one memo
only adds waiting. Batching many works stage by stage: each company's workflow runs
unchanged in its own thread, and whenever every running workflow is waiting on Claude,
the ``Batcher`` sends all their pending calls as one batch, waits for it to end and
hands each reply back. Ten memos take the same four or five rounds as one.

Replies are cached on disk by a hash of the request (``ReplyCache``), and a batch's id
is recorded while it runs, so an interrupted pilot can be rerun: finished calls come
from the cache at no cost and a batch still in flight is collected, not resubmitted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .claude import ClaudeLLM, reply_of

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

BATCH_DISCOUNT = 0.5
CACHE = Path(__file__).resolve().parents[3] / "data" / "workstation" / "batch_cache"


def params_key(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:32]


class ReplyCache:
    """Replies by request hash, and the batches in flight (``inflight.json``)."""

    def __init__(self, root: Path = CACHE) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.root / f"{key}.json"
        reply: dict[str, Any] | None = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return reply

    def put(self, key: str, reply: dict[str, Any]) -> None:
        (self.root / f"{key}.json").write_text(json.dumps(reply), encoding="utf-8")

    def inflight(self) -> dict[str, dict[str, str]]:
        p = self.root / "inflight.json"
        got: dict[str, dict[str, str]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return got

    def set_inflight(self, batch_id: str, ids: dict[str, str] | None) -> None:
        with self._lock:
            cur = self.inflight()
            if ids is None:
                cur.pop(batch_id, None)
            else:
                cur[batch_id] = ids
            (self.root / "inflight.json").write_text(json.dumps(cur, indent=1), encoding="utf-8")


class Batcher:
    def __init__(self, client: Any, cache: ReplyCache, poll_seconds: float = 30.0) -> None:
        self.client, self.cache, self.poll = client, cache, poll_seconds
        self._cv = threading.Condition()
        self._pending: dict[str, tuple[dict[str, Any], Future, int]] = {}  # key -> params, future, waiters
        self._active = 0
        self._waiting = 0
        self.batches: list[dict[str, Any]] = []   # id, requests, seconds

    # -- called from workflow threads ------------------------------------------------------
    def submit(self, params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """(reply, from cache). Blocks until the batch holding this request has ended."""
        key = params_key(params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        with self._cv:
            if key in self._pending:
                p, fut, n = self._pending[key]
                self._pending[key] = (p, fut, n + 1)
            else:
                fut = Future()
                self._pending[key] = (params, fut, 1)
            self._waiting += 1
            self._cv.notify_all()
        # The coordinator releases this waiter (``flush``) before setting the result, so a
        # thread that has its reply is never counted as waiting.
        return fut.result(), False

    # -- the coordinator ------------------------------------------------------------------
    def collect(self, batch_id: str, ids: dict[str, str]) -> dict[str, dict[str, Any]]:
        """Wait for a batch to end; key -> reply (an ``error`` reply for failed requests)."""
        start = time.monotonic()
        while self.client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(self.poll)
        out: dict[str, dict[str, Any]] = {}
        for entry in self.client.messages.batches.results(batch_id):
            key = ids.get(entry.custom_id)
            if key is None:
                continue
            if entry.result.type == "succeeded":
                reply = reply_of(entry.result.message)
                reply["request_id"] = f"{batch_id}/{entry.custom_id}"
                self.cache.put(key, reply)
            else:
                err = getattr(entry.result, "error", None)
                reply = {"error": f"batch request {entry.result.type}: {err}"}
            out[key] = reply
        for key in set(ids.values()) - set(out):
            out[key] = {"error": "batch request missing from the results"}
        self.cache.set_inflight(batch_id, None)
        self.batches.append({"id": batch_id, "requests": len(ids), "seconds": round(time.monotonic() - start, 1)})
        logger.info("batch %s ended: %d requests in %.0f s", batch_id, len(ids), time.monotonic() - start)
        return out

    def resume(self) -> None:
        """Collect batches an earlier, interrupted run left in flight (fills the cache)."""
        for batch_id, ids in self.cache.inflight().items():
            logger.info("collecting batch %s left in flight (%d requests)", batch_id, len(ids))
            self.collect(batch_id, ids)

    def flush(self) -> None:
        with self._cv:
            pending, self._pending = self._pending, {}
        ids = {f"r{i}": key for i, key in enumerate(pending)}
        try:
            batch = self.client.messages.batches.create(
                requests=[{"custom_id": cid, "params": pending[key][0]} for cid, key in ids.items()])
            self.cache.set_inflight(batch.id, ids)
            logger.info("batch %s submitted: %d requests", batch.id, len(ids))
            replies: dict[str, dict[str, Any]] | None = self.collect(batch.id, ids)
            error: Exception | None = None
        except Exception as e:  # noqa: BLE001 - fail the waiting jobs instead of leaving them blocked
            logger.exception("batch failed")
            replies, error = None, e
        with self._cv:
            self._waiting -= sum(n for _, _, n in pending.values())
        for key, (_, fut, _) in pending.items():
            if replies is not None:
                fut.set_result(replies[key])
            else:
                fut.set_exception(error or RuntimeError("batch failed"))

    def run(self, jobs: list[Callable[[], Any]]) -> list[Any]:
        """Run each job in its own thread; submit a batch whenever all running jobs wait.
        Returns each job's result, or the exception it raised."""
        self.resume()
        results: list[Any] = [None] * len(jobs)

        def wrap(i: int, job: Callable[[], Any]) -> None:
            try:
                results[i] = job()
            except Exception as e:  # noqa: BLE001 - one failed memo must not stop the others
                logger.exception("job %d failed", i)
                results[i] = e
            finally:
                with self._cv:
                    self._active -= 1
                    self._cv.notify_all()

        with self._cv:
            self._active = len(jobs)
        threads = [threading.Thread(target=wrap, args=(i, j), daemon=True) for i, j in enumerate(jobs)]
        for t in threads:
            t.start()
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._active == 0 or (self._pending and self._waiting >= self._active))
                if self._active == 0 and not self._pending:
                    break
            self.flush()
        for t in threads:
            t.join()
        return results


class BatchLLM(ClaudeLLM):
    """The Claude client's roles, models and schemas, sent through a ``Batcher``. Costs are
    at batch prices; a reply served from the cache costs nothing again."""

    def __init__(self, batcher: Batcher, models: dict[str, str] | None = None, effort: str = "high",
                 max_tokens: int = 32000) -> None:
        super().__init__(client=None, models=models, effort=effort, max_tokens=max_tokens)
        self.batcher = batcher

    def complete_json(self, *, role: str, task: str, inputs: dict[str, Any], schema: type[T]) -> T:
        params = self.params(role, task, inputs, schema)
        reply, cached = self.batcher.submit(params)
        try:
            return self.finish(role, params, reply, schema, cost_factor=0.0 if cached else BATCH_DISCOUNT)
        finally:
            self.calls[-1]["cached"] = cached
