"""Run storage: each run is written once, under data/workstation/runs/<run_id>/.

``state.json`` holds the whole workflow state, ``packet.json`` the review packet,
``audit.jsonl`` the event log; ``decision.json`` is added when a human reviews it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .packet import ReviewPacket
from .state import WorkflowState

RUNS = Path(__file__).resolve().parents[3] / "data" / "workstation" / "runs"


def run_id(state: WorkflowState) -> str:
    body = json.dumps(state.to_dict(), sort_keys=True, default=str).encode()
    r = state.request
    return f"{r.as_of.isoformat()}_{r.ticker}_{hashlib.sha256(body).hexdigest()[:10]}"


def save_run(state: WorkflowState, packet: ReviewPacket, root: Path = RUNS) -> Path:
    folder = root / run_id(state)
    folder.mkdir(parents=True, exist_ok=False)  # runs are never overwritten
    (folder / "state.json").write_text(json.dumps(state.to_dict(), indent=1, default=str), encoding="utf-8")
    (folder / "packet.json").write_text(packet.model_dump_json(indent=1), encoding="utf-8")
    with open(folder / "audit.jsonl", "w", encoding="utf-8") as fh:
        for e in state.audit:
            fh.write(json.dumps(e, default=str) + "\n")
    return folder


def record_decision(folder: Path, action: str, notes: str = "", reviewer: str = "user") -> dict:
    packet = ReviewPacket.model_validate_json((folder / "packet.json").read_text(encoding="utf-8"))
    if action not in packet.allowed_actions:
        raise ValueError(f"{action!r} is not allowed for this run (allowed: {packet.allowed_actions})")
    path = folder / "decision.json"
    if path.exists():
        raise FileExistsError(f"{folder.name} already has a decision; start a revision run instead")
    decision = {"action": action, "notes": notes, "reviewer": reviewer,
                "t": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(decision, indent=1), encoding="utf-8")
    return decision
