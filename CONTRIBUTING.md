# Contributing to Auto Researcher

## Development setup

```bash
git clone https://github.com/AndrewFSee/auto_researcher.git
cd auto_researcher
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add extras (nlp, llm, dashboard, ...) as needed
```

## Checks

CI runs the same commands:

```bash
ruff check src tests scripts app.py      # correctness rules (pyflakes, bugbear)
pytest -q                                # tests needing optional deps skip themselves
mypy --ignore-missing-imports --follow-imports=silent \
    src/auto_researcher/validation/splits.py src/auto_researcher/validation/event_dates.py \
    src/auto_researcher/backtest/walk_forward.py src/auto_researcher/backtest/baselines.py \
    src/auto_researcher/features/alpha_factors.py src/auto_researcher/features/earnings_events.py \
    src/auto_researcher/composite.py src/auto_researcher/screening.py
```

The legacy code base has not been reformatted yet. Run `ruff format` on files
you create, but keep formatting-only changes to existing files in their own
commit so functional diffs stay reviewable.

## Adding a signal, model or agent

The project's earlier headline numbers were all produced by evaluation
mistakes (see [docs/AUDIT.md](docs/AUDIT.md)). New work follows these rules:

1. **Features must be causal.** A value dated *t* may use data up to the close
   of *t* only. Add your feature builder to the perturbation tests in
   `tests/test_feature_causality.py`.
2. **Evaluate with the shared tools, not a custom loop.**
   * Panel signals and models: `auto_researcher.backtest.walk_forward.run_walk_forward`
     (purged training windows, next-close execution, costs, baselines).
   * Event signals: key events on the date the information became public (an
     announcement or filing date, never a fiscal period end;
     `validation.event_dates.assert_announcement_dates` checks this), as in
     `scripts/pead_event_study.py`.
3. **Report the whole picture.** Mean IC with a Newey-West t-stat, the
   information ratio against the equal-weight universe, the percentile against
   random portfolios, and the number of variants you tried (`n_trials`) so the
   statistics can be deflated.
4. **Don't hardcode performance numbers** in code or docs. Generate a report under
   `docs/results/` and cite it.
5. **Feed the evidence to the composite.** Add the measured IC and its period
   count to `scripts/calibrate_ic_weights.py`; `auto_researcher.composite` turns
   it into a weight. Unmeasured agents get only a small prior weight.

## Pull requests

* Branch from `main`, keep commits focused, and use
  [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
  `docs:`, `test:`, `refactor:`, `chore:`).
* Add tests for new behavior and make sure the checks above pass.
