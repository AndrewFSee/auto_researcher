"""The calculator: derived numbers come from code, not from the model.

A draft may declare calculations (``schemas.Calculation``): a formula over

* metric ids (``m3``) and numeric context ids (``x1``): their values in natural units;
* numbers quoted in filings: ``f6["21,155"]`` is the number whose digits are ``21,155`` in
  one of claim f6's verified quotes, as written: ``21155`` (scale it yourself: a table
  "in millions" needs ``* 1e6``; "$132.4 billion" referenced as ``f10["132.4"]`` needs
  ``* 1e9``). Written with its unit (``f6["$4.6 billion"]``) it must match the quote's
  unit too and comes back in natural units (4.6e9);
* numeric constants (unit scales, counts such as 4 quarters), visible in the formula;

with ``+ - * /``, parentheses, unary minus and ``abs``, ``min``, ``max``. The formula is
parsed with ``ast`` and only those forms are evaluated; anything else is an error. Each
result becomes a ``Metric`` (id ``c1``, ...) that paragraphs cite like any other.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from datetime import date

from .gates import numbers_in
from .schemas import Calculation, Claim, ContextItem, Metric

FUNCTIONS: dict[str, Callable[..., float]] = {"abs": lambda x: abs(x), "min": lambda *a: min(a),
                                             "max": lambda *a: max(a)}


class CalcError(ValueError):
    pass


def evaluate(expr: str, metrics: dict[str, Metric], context: dict[str, ContextItem],
             claims: dict[str, Claim]) -> tuple[float, list[str]]:
    """(value, the ids used). Raises ``CalcError`` for anything not allowed or unknown."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"cannot parse {expr!r}") from e
    used: list[str] = []

    def quoted(cid: str, written: str) -> float:
        claim = claims.get(cid)
        if claim is None:
            raise CalcError(f"unknown claim {cid}")
        want = numbers_in(written)
        if not want:
            raise CalcError(f"{written!r} is not a number")
        w = want[0]
        bare = not w.suffix and "$" not in written
        for q in claim.evidence:
            for n in numbers_in(q.quote):
                if n.raw == w.raw and (bare or n.value == w.value):
                    used.append(f"{cid}[{written}]")
                    return n.raw if bare else n.value
        raise CalcError(f"{written!r} is not in claim {cid}'s quotes")

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in metrics:
                v = metrics[node.id].value
                if v is None:
                    raise CalcError(f"metric {node.id} has no value")
                used.append(node.id)
                return float(v)
            if node.id in context:
                cv = context[node.id].value
                if not isinstance(cv, (int, float)):
                    raise CalcError(f"context {node.id} is not numeric")
                used.append(node.id)
                return float(cv)
            raise CalcError(f"unknown id {node.id}")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return quoted(node.value.id, node.slice.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if b == 0:
                raise CalcError("division by zero")
            return a / b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS \
                and node.args and not node.keywords:
            return float(FUNCTIONS[node.func.id](*(ev(a) for a in node.args)))
        raise CalcError(f"not allowed in a calculation: {ast.dump(node)[:60]}")

    return ev(tree), used


def run_calculations(calcs: list[Calculation], metrics: list[Metric], context: list[ContextItem],
                     claims: dict[str, Claim], as_of: date) -> tuple[list[Metric], dict[str, str]]:
    """(calculated metrics, calculation id -> error). A calculation's data date is the
    latest of its inputs'; quoted filing numbers count as of the run's date.

    A calculation may use other calculations' results (``c8 = c7 / m3``), in any order:
    calculations are evaluated in passes until no more succeed, so one that rests on a
    failed or circular calculation fails too."""
    m_by_id, c_by_id = {m.id: m for m in metrics}, {c.id: c for c in context}
    done: dict[str, Metric] = {}
    errors: dict[str, str] = {}
    pending = list(calcs)
    while pending:
        waiting = []
        for calc in pending:
            known = {**m_by_id, **done}
            try:
                value, used = evaluate(calc.expression, known, c_by_id, claims)
            except CalcError as e:
                errors[calc.id] = str(e)
                waiting.append(calc)
                continue
            errors.pop(calc.id, None)
            dates = [known[u].as_of for u in used if u in known] + [c_by_id[u].as_of for u in used if u in c_by_id]
            public = [d for u in used if u in known and (d := known[u].available_date)]
            done[calc.id] = Metric(id=calc.id, name=calc.description, value=value, unit=calc.unit,
                                   as_of=max(dates) if dates else as_of,
                                   available_date=max(public) if public else None,
                                   source=f"calculation: {calc.expression}", snapshot_id="calc:" + ";".join(used),
                                   max_age_days=100_000)
        if len(waiting) == len(pending):
            break
        pending = waiting
    return [done[c.id] for c in calcs if c.id in done], errors
