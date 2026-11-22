"""Batch mode with a fake Message Batches API: memos share batch rounds, replies are cached
and reused, a failed request fails only its own memo, and interrupted batches are collected."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from test_workstation import AS_OF, BEAR, FILING, LATE, MACRO, MDA, METRICS, PLAN, RISKS, BANDS, draft

from auto_researcher.workstation import Request, Workstation
from auto_researcher.workstation.agents import BEAR_TASK, FILING_TASK
from auto_researcher.workstation.batch import Batcher, BatchLLM, ReplyCache, params_key
from auto_researcher.workstation.claude import LLMError, call_cost
from auto_researcher.workstation.orchestrator import DRAFT_TASK, PLAN_TASK
from auto_researcher.workstation.tools.fixtures import FixtureContext, FixtureFilings, FixtureMetrics

REPLIES = {PLAN_TASK: PLAN, FILING_TASK: FILING, BEAR_TASK: BEAR, DRAFT_TASK: draft()}
USAGE = {"input_tokens": 1000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 200}


class FakeBatches:
    def __init__(self, fail_marker: str | None = None):
        self.created: list[tuple[str, list[dict]]] = []
        self.fail_marker = fail_marker

    def create(self, requests):
        bid = f"msgbatch_{len(self.created)}"
        self.created.append((bid, requests))
        return NS(id=bid)

    def retrieve(self, bid):
        return NS(processing_status="ended")

    def results(self, bid):
        for r in dict(self.created)[bid]:
            p = r["params"]
            system, user = p["system"][0]["text"], p["messages"][0]["content"]
            task = next(t for t in REPLIES if system.endswith(t))
            if self.fail_marker and self.fail_marker in user and task == BEAR_TASK:
                yield NS(custom_id=r["custom_id"], result=NS(type="errored", error="overloaded"))
                continue
            msg = NS(content=[NS(type="text", text=json.dumps(REPLIES[task]))], stop_reason="end_turn",
                     model=p["model"], usage=NS(**USAGE))
            yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=msg))


def _jobs(batcher, questions):
    def job(q):
        def run():
            ws = Workstation(BatchLLM(batcher), FixtureFilings([MDA, RISKS, LATE]), FixtureMetrics(METRICS),
                             FixtureContext([MACRO, BANDS]))
            return ws.run(Request(ticker="ACME", as_of=AS_OF, question=q))
        return run
    return [job(q) for q in questions]


def test_memos_share_batch_rounds_at_half_price_and_reruns_come_from_the_cache(tmp_path):
    fake = NS(messages=NS(batches=FakeBatches()))
    batcher = Batcher(fake, ReplyCache(tmp_path), poll_seconds=0)
    out = batcher.run(_jobs(batcher, ["Memo one.", "Memo two."]))
    created = fake.messages.batches.created
    assert [len(reqs) for _, reqs in created] == [2, 2, 2, 2]  # plan, filing, bear, draft: one round each
    for packet, state in out:
        assert [f.code for f in packet.flags] == [] and len(packet.paragraphs) == 4
        usage = next(e for e in state.audit if e["event"] == "llm_usage")
        assert all(c["cost"] == pytest.approx(call_cost(c["model"], USAGE) * 0.5) for c in usage["calls"])
    assert ReplyCache(tmp_path).inflight() == {}

    again = Batcher(fake, ReplyCache(tmp_path), poll_seconds=0)
    out2 = again.run(_jobs(again, ["Memo one.", "Memo two."]))
    assert len(fake.messages.batches.created) == 4  # nothing new submitted
    for (p1, _), (p2, s2) in zip(out, out2):
        assert p2.model_dump() == p1.model_dump()
        usage = next(e for e in s2.audit if e["event"] == "llm_usage")
        assert usage["cost"] == 0 and all(c["cached"] for c in usage["calls"])


def test_a_failed_request_fails_only_its_own_memo(tmp_path):
    fake = NS(messages=NS(batches=FakeBatches(fail_marker="FAILME")))
    batcher = Batcher(fake, ReplyCache(tmp_path), poll_seconds=0)
    ok, failed = batcher.run(_jobs(batcher, ["Fine memo.", "FAILME memo."]))
    assert ok[0].paragraphs and isinstance(failed, LLMError) and "errored" in str(failed)


def test_identical_requests_are_sent_once(tmp_path):
    fake = NS(messages=NS(batches=FakeBatches()))
    batcher = Batcher(fake, ReplyCache(tmp_path), poll_seconds=0)
    a, b = batcher.run(_jobs(batcher, ["Same memo.", "Same memo."]))
    assert [len(reqs) for _, reqs in fake.messages.batches.created] == [1, 1, 1, 1]
    assert a[0].model_dump() == b[0].model_dump()


def test_a_batch_left_in_flight_is_collected_not_resubmitted(tmp_path):
    fake = NS(messages=NS(batches=FakeBatches()))
    llm = BatchLLM(Batcher(fake, ReplyCache(tmp_path)))
    from auto_researcher.workstation.schemas import Plan
    params = llm.params("orchestrator_plan", PLAN_TASK, {"request": {"ticker": "ACME"}}, Plan)
    fake.messages.batches.created.append(("msgbatch_old", [{"custom_id": "r0", "params": params}]))
    cache = ReplyCache(tmp_path)
    cache.set_inflight("msgbatch_old", {"r0": params_key(params)})
    Batcher(fake, cache, poll_seconds=0).resume()
    assert cache.get(params_key(params))["text"] == json.dumps(PLAN) and cache.inflight() == {}
