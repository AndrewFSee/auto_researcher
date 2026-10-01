"""
Phase 5 integration wire-ups: altdata feeds -> feature_pipeline + dump.

These tests pin the two integration points:

* ``_build_altdata_features`` turns a list of alt-data adapters into a
  properly-shaped ``(ticker, feature)`` MultiIndex-column DataFrame that
  slots into ``_merge_feature_matrices`` alongside technical / sentiment
  features.
* ``scripts.altdata_dump.dump_adapter_to_parquet`` writes the
  canonical long-format parquet that ``scripts/validate_signal_ic.py``
  expects — so callers can CPCV-validate a new adapter before enabling
  it in the composite, per the plan.

Both tests use stub adapters so the suite runs without network.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from auto_researcher.features.feature_pipeline import _build_altdata_features


@pytest.fixture
def repo_tmp_path():
    """Repo-local temp dir — the OS ``%TEMP%`` can be permissioned-out on
    this Windows dev box, which breaks pytest's built-in ``tmp_path``
    fixture. Uses the same workaround as ``test_alpha_agents.py``."""
    base = Path(__file__).parent.parent / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="altdata_integration_", dir=str(base)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _StubAdapter:
    """Matches the AltDataAdapter protocol with a deterministic payload."""

    def __init__(self, name: str, scores: dict[tuple[str, str], float]):
        self.name = name
        self._scores = scores
        self.fetch_calls: list[tuple] = []

    def fetch(self, tickers, start, end):
        self.fetch_calls.append((tuple(tickers), start, end))
        data = []
        for (date_str, ticker), val in self._scores.items():
            if ticker in tickers:
                data.append(((pd.Timestamp(date_str), ticker), val))
        if not data:
            return pd.Series(
                [],
                index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
                dtype=float,
            )
        idx = pd.MultiIndex.from_tuples([k for k, _ in data], names=["date", "ticker"])
        return pd.Series([v for _, v in data], index=idx, name=self.name)


class TestBuildAltdataFeatures:
    def test_empty_adapters_returns_empty_frame(self) -> None:
        price_index = pd.date_range("2024-01-02", periods=5, freq="B")
        out = _build_altdata_features(
            adapters=(),
            tickers=["AAPL", "MSFT"],
            price_index=price_index,
        )
        assert out.empty

    def test_single_adapter_multiindex_columns(self) -> None:
        price_index = pd.date_range("2024-01-02", periods=5, freq="B")
        adapter = _StubAdapter(
            name="wikipedia_pageviews",
            scores={
                ("2024-01-02", "AAPL"): 1.0,
                ("2024-01-02", "MSFT"): -1.0,
                ("2024-01-03", "AAPL"): 2.0,
                ("2024-01-03", "MSFT"): -2.0,
            },
        )
        out = _build_altdata_features(
            adapters=(adapter,),
            tickers=["AAPL", "MSFT"],
            price_index=price_index,
        )
        # Should have a MultiIndex column (ticker, altdata_<name>)
        assert isinstance(out.columns, pd.MultiIndex)
        assert set(out.columns.get_level_values("feature").unique()) == {
            "altdata_wikipedia_pageviews"
        }
        assert set(out.columns.get_level_values("ticker").unique()) == {"AAPL", "MSFT"}
        # Index must match price calendar.
        assert out.index.equals(price_index)

    def test_zscore_normalization_applied(self) -> None:
        """Each date should be z-scored across tickers before alignment."""
        price_index = pd.date_range("2024-01-02", periods=1, freq="B")
        adapter = _StubAdapter(
            name="test",
            # A mean-zero, unit-std sequence in raw space becomes z-scored.
            # Two-ticker dates produce symmetric ±1 (std) z-scores.
            scores={
                ("2024-01-02", "A"): 10.0,
                ("2024-01-02", "B"): 0.0,
            },
        )
        out = _build_altdata_features(
            adapters=(adapter,),
            tickers=["A", "B"],
            price_index=price_index,
        )
        # After z-scoring, A should be positive, B should be negative,
        # and they should be mirror images (sum to ~0).
        v_a = out[("A", "altdata_test")].iloc[0]
        v_b = out[("B", "altdata_test")].iloc[0]
        assert v_a > 0
        assert v_b < 0
        assert abs(v_a + v_b) < 1e-9

    def test_forward_fill_onto_price_calendar(self) -> None:
        """A single fetched date propagates forward via ffill."""
        price_index = pd.date_range("2024-01-02", periods=5, freq="B")
        adapter = _StubAdapter(
            name="episodic",
            scores={
                ("2024-01-02", "AAPL"): 1.0,
                ("2024-01-02", "MSFT"): -1.0,
                # Nothing else — later dates should inherit via ffill.
            },
        )
        out = _build_altdata_features(
            adapters=(adapter,),
            tickers=["AAPL", "MSFT"],
            price_index=price_index,
        )
        col_aapl = ("AAPL", "altdata_episodic")
        # All five business days should have a value after ffill.
        assert out[col_aapl].notna().all()
        # Later values should equal the first (ffill).
        first = out[col_aapl].iloc[0]
        for i in range(1, 5):
            assert out[col_aapl].iloc[i] == first

    def test_flaky_adapter_is_skipped_not_raised(self) -> None:
        """An adapter that raises shouldn't poison the whole matrix."""
        price_index = pd.date_range("2024-01-02", periods=3, freq="B")

        class _FlakyAdapter:
            name = "flaky"
            def fetch(self, tickers, start, end):
                raise RuntimeError("rate-limited")

        good = _StubAdapter(
            name="good",
            scores={
                ("2024-01-02", "AAPL"): 1.0,
                ("2024-01-02", "MSFT"): -1.0,
            },
        )
        out = _build_altdata_features(
            adapters=(_FlakyAdapter(), good),
            tickers=["AAPL", "MSFT"],
            price_index=price_index,
        )
        # Flaky adapter produced no columns; good adapter survived.
        features = set(out.columns.get_level_values("feature").unique())
        assert features == {"altdata_good"}

    def test_empty_price_index(self) -> None:
        """Empty price window returns a correctly-shaped empty DataFrame."""
        out = _build_altdata_features(
            adapters=(_StubAdapter("x", {}),),
            tickers=["A", "B"],
            price_index=pd.DatetimeIndex([]),
        )
        assert out.empty
        assert out.columns.names == ["ticker", "feature"]

    def test_all_adapters_empty_returns_empty_frame(self) -> None:
        """If every adapter returns empty, the result is an empty MultiIndex frame."""
        price_index = pd.date_range("2024-01-02", periods=3, freq="B")
        adapter = _StubAdapter(name="x", scores={})  # no data
        out = _build_altdata_features(
            adapters=(adapter,),
            tickers=["A"],
            price_index=price_index,
        )
        assert out.empty
        assert out.columns.names == ["ticker", "feature"]


class TestAltdataDumpScript:
    """Verify the long-format parquet layout validate_signal_ic.py expects."""

    def test_dump_writes_long_format_parquet(self, repo_tmp_path: Path) -> None:
        """Integration shape: date / ticker / score columns."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "altdata_dump",
            Path(__file__).parent.parent / "scripts" / "altdata_dump.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Monkey-patch _make_adapter to return our stub so we don't hit
        # the network or need a real EDGAR request.
        stub = _StubAdapter(
            name="wikipedia_pageviews",
            scores={
                ("2024-01-02", "AAPL"): 1.0,
                ("2024-01-03", "AAPL"): 2.0,
                ("2024-01-03", "MSFT"): 3.0,
            },
        )
        mod._make_adapter = lambda name, cache_dir: stub  # noqa: E501

        out_path = repo_tmp_path / "dump.parquet"
        mod.dump_adapter_to_parquet(
            adapter_name="wikipedia",
            tickers=["AAPL", "MSFT"],
            start="2024-01-01",
            end="2024-01-10",
            out=out_path,
            cache_dir=repo_tmp_path / "cache",
        )

        assert out_path.exists()
        df = pd.read_parquet(out_path)
        assert list(df.columns) == ["date", "ticker", "score"]
        assert len(df) == 3
        # Date column must be datetime, matching validate_signal_ic.py's
        # ``df["date"] = pd.to_datetime(df["date"])`` expectation.
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_dump_handles_empty_adapter_output(self, repo_tmp_path: Path) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "altdata_dump",
            Path(__file__).parent.parent / "scripts" / "altdata_dump.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        empty_stub = _StubAdapter(name="wikipedia_pageviews", scores={})
        mod._make_adapter = lambda name, cache_dir: empty_stub  # noqa: E501

        out_path = repo_tmp_path / "empty.parquet"
        mod.dump_adapter_to_parquet(
            adapter_name="wikipedia",
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-02",
            out=out_path,
            cache_dir=repo_tmp_path / "cache",
        )
        df = pd.read_parquet(out_path)
        assert list(df.columns) == ["date", "ticker", "score"]
        assert len(df) == 0
