"""Tests for the FinQA program executor and answer scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from auto_researcher.llm.finqa import (
    PROGRAM_PROMPT,
    FinQAExample,
    ProgramError,
    answers_match,
    build_messages,
    execute_program,
    extract_answer,
    extract_program,
    load_finqa,
    parse_program,
)

TABLE = (
    ("", "2017", "2016", "2015"),
    ("provision for income taxes", "$ 829", "$ 480", "$ 104"),
    ("net income ( loss )", "1,200", "( 300 )", "50"),
)


def test_arithmetic_and_step_references():
    assert execute_program("subtract(5829, 5735), divide(#0, 5735)") == round(94 / 5735, 5)
    assert execute_program("multiply(4.9, const_m1), subtract(#0, -3.7)") == pytest.approx(-1.2)
    assert execute_program("divide(1200, 8000)") == 0.15


def test_exp_is_power_and_greater_returns_yes_no():
    cagr = execute_program(
        "divide(208.14, const_100), divide(const_1, const_5), exp(#0, #1), subtract(#2, const_1)"
    )
    assert cagr == round(2.0814 ** 0.2 - 1, 5)
    assert execute_program("greater(286.61, 198.09)") == "yes"
    assert execute_program("greater(1, 2)") == "no"


def test_percent_and_dollar_arguments():
    assert execute_program("add(5%, 10%)") == 0.15
    assert execute_program("add($ 1,000, 1)") == 1001
    assert execute_program("add(5,3)") == 8


def test_table_operations_match_row_names_loosely():
    assert execute_program("table_average(provision for income taxes, none)", TABLE) == 471.0
    assert execute_program("table_max( Provision  for income taxes , none)", TABLE) == 829.0
    with pytest.raises(ProgramError, match="no table row"):
        execute_program("table_sum(revenue, none)", TABLE)


def test_row_names_with_parentheses_parse_but_bad_cells_fail():
    assert parse_program("table_sum(net income ( loss ), none), add(#0, 1)") == [
        ("table_sum", ["net income ( loss )"]),
        ("add", ["#0", "1"]),
    ]
    with pytest.raises(ProgramError, match="non-numeric"):
        execute_program("table_sum(net income ( loss ), none)", TABLE)


@pytest.mark.parametrize(
    "program",
    ["", "add(1)", "add(1, 2", "foo(1, 2)", "add(#3, 1)", "divide(1, 0)", "add(x, 1)",
     "greater(1, 2), add(#0, 1)", "exp(-8, 0.5)"],
)
def test_bad_programs_raise_program_error(program):
    with pytest.raises(ProgramError):
        execute_program(program)


def test_extract_program_takes_last_program_line():
    reply = "<think>hmm</think>We need growth.\nProgram: add(1, 2)\nProgram: `subtract(5, 3)`"
    assert extract_program(reply) == "subtract(5, 3)"
    assert extract_program("divide(1, 2)") == "divide(1, 2)"
    assert extract_program("I am not sure.") is None


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("so... Answer: 14.5%", 14.5), ("Answer: $1,234.5 million", 1234.5),
     ("Answer: -3.2", -3.2), ("Answer: (12)", -12.0), ("Answer: Yes, it did", "yes"),
     ("no final line", None)],
)
def test_extract_answer(reply, expected):
    assert extract_answer(reply) == expected


def test_answers_match_modes():
    assert answers_match(0.14464, 0.14464)
    assert answers_match(0.144641, 0.14464)  # rounded to 5 decimals first
    assert not answers_match(14.464, 0.14464)
    assert answers_match(14.464, 0.14464, "scale")
    assert not answers_match(14.5, 0.14464, "scale")
    assert answers_match(14.5, 0.14464, "tolerant")
    assert answers_match("YES", "yes")
    assert answers_match(0.0, 0.0) and not answers_match(None, 1.0)
    assert not answers_match(float("nan"), 1.0, "tolerant")


def test_program_prompt_is_frozen():
    # The baseline run and the fine-tuning data (notebooks/finqa_qlora_colab.ipynb)
    # used exactly this text; a fine-tuned model is only comparable under it.
    # Changing it means re-running the baseline and retraining.
    digest = hashlib.sha256(PROGRAM_PROMPT.encode()).hexdigest()
    assert digest == "71e761a062a815845bb39e8f18b006fbc9b807f22a0c3f0d3b4c00b6ff588edb"


def test_build_messages_puts_page_and_question_in_user_turn():
    ex = FinQAExample(id="X/2020/page_1.pdf-1", question="what is it?", pre_text=("before",),
                      post_text=("after",), table=TABLE, program="add(1, 2)", answer=3.0)
    system, user = build_messages(ex)
    assert system == {"role": "system", "content": PROGRAM_PROMPT}
    assert user["content"].startswith("Report page:\nbefore\n\nTable:\n | 2017 | 2016 | 2015\n")
    assert user["content"].endswith("after\n\nQuestion: what is it?")


DATA = Path(__file__).resolve().parent.parent / "data" / "finqa" / "test.json"


@pytest.mark.skipif(not DATA.exists(), reason="FinQA test split not downloaded")
def test_gold_programs_reproduce_gold_answers():
    examples = load_finqa(DATA)
    assert len(examples) == 1147
    assert all(answers_match(execute_program(e.program, e.table), e.answer) for e in examples)
