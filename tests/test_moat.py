"""Tests for the moat tagger's text handling: name hiding, chunking, quote checks, prompts."""

from __future__ import annotations

from auto_researcher.llm.moat import (
    anonymize,
    check_rating,
    chunk_lines,
    collect_evidence,
    name_variants,
    number_quotes,
    one_call_params,
    rate_messages,
    self_aliases,
    short_name,
    verify_evidence,
    verify_quote,
)


def test_anonymize_hides_names_short_names_and_tickers_but_not_prose():
    assert name_variants("MICRON TECHNOLOGY INC") == ["MICRON TECHNOLOGY INC", "MICRON TECHNOLOGY"]
    assert short_name("MICRON TECHNOLOGY INC") == "Micron" and short_name("AMERICAN EXPRESS CO") is None
    text = "Micron Technology, Inc. (MU) makes memory. MICRON's fabs are large. Micron-scale parts."
    assert anonymize(text, "MICRON TECHNOLOGY INC", ["MU"]) == (
        "the Company. (the Company) makes memory. the Company fabs are large. the Company-scale parts.")
    jj = anonymize("Johnson & Johnson sells to ALL hospitals.", "JOHNSON & JOHNSON", ["JNJ", "ALL"])
    assert jj == "the Company sells to the Company hospitals."  # tickers are upper-case words
    assert anonymize("we serve all regions", "REGIONS FINANCIAL CORP", ["RF"]) == "we serve all regions"
    assert anonymize("Starbucks Corporation (together with", "STARBUCKS CORP", ["SBUX"]) == "the Company (together with"
    text = ("Overview C.H. Robinson Worldwide, Inc. (“C.H. Robinson,” “we,” or “us”) uses "
            "the Robinson Operating Model (“ROM”). See https://www.chrobinson.com/about. C.H. Robinson's "
            "network.")
    assert self_aliases(text) == ["C.H. Robinson"]
    assert anonymize(text, "C. H. ROBINSON WORLDWIDE, INC.", ["CHRW"]) == (
        "Overview the Company (“the Company,” “we,” or “us”) uses the Robinson Operating "
        "Model (“ROM”). See [website] the Company network.")


def test_chunk_lines_packs_paragraphs_and_cuts_long_lines_at_sentences():
    text = "\n".join(["a" * 40, "b" * 40, "c" * 40])
    assert chunk_lines(text, max_chars=90) == ["a" * 40 + "\n" + "b" * 40, "c" * 40]
    long = "First sentence here. " * 10
    parts = chunk_lines(long, max_chars=50)
    assert all(len(p) <= 50 for p in parts) and all(p.endswith(".") for p in parts)
    assert " ".join(parts) == long.strip()


def test_verify_quote_is_word_for_word_up_to_case_spacing_and_typography():
    text = "Our network connects   150 million merchants—more than any other “processor”."
    assert verify_quote("our network connects 150 million merchants-more than any other \"processor\".", text)
    assert not verify_quote("our network connects 150 million merchants worldwide", text)
    assert not verify_quote("150 million", text)  # too short to be evidence
    html = "Tinder ® was launched in 2012 . PSE&G —A New Jersey utility. Biktarvy 2033 2033"
    assert verify_quote("Tinder® was launched in 2012.", html)  # spacing around signs from HTML
    assert verify_quote("PSE&G—A New Jersey utility", html)
    assert verify_quote("Biktarvy | 2033 | 2033", html)  # a table row quoted with bars
    assert not verify_quote("Tinder® was launched in 2013", html)


def test_collect_evidence_keeps_verified_unique_quotes_and_counts_the_rest():
    chunk = "We hold exclusive rights to the only pipeline into the region. Prices are set by the market."
    reply = {"evidence": [
        {"kind": "efficient_scale", "quote": "We hold exclusive rights to the only pipeline into the region."},
        {"kind": "against", "quote": "Prices are set by the market"},
        {"kind": "intangibles", "quote": "We own the strongest brand in the world."},  # not in the text
        {"kind": "efficient_scale", "quote": "we hold exclusive rights to the only pipeline into the region"},
        {"kind": "moat", "quote": "Prices are set by the market."}]}  # unknown kind
    by_kind, counts = collect_evidence([(chunk, reply)])
    assert by_kind["efficient_scale"] == ["We hold exclusive rights to the only pipeline into the region."]
    assert by_kind["against"] == ["Prices are set by the market"]
    assert counts == {"kept": 2, "not_in_text": 2, "duplicate": 1, "over_cap": 0}


def test_rate_prompt_numbers_quotes_and_check_rating_enforces_citations():
    numbered = number_quotes({"intangibles": ["patent A expires in 2035"], "against": ["prices are set by the market"]})
    assert numbered == [(1, "intangibles", "patent A expires in 2035"), (2, "against", "prices are set by the market")]
    user = rate_messages("The Company makes drugs.", numbered)[1]["content"]
    assert "intangibles:\n[1] patent A expires in 2035" in user and "network_effects:\n(none)" in user
    assert "against:\n[2] prices are set by the market" in user
    zero = {"rating": 0, "quotes": [], "reason": ""}
    rating = {"intangibles": {"rating": 2, "quotes": [1, 7], "reason": "patents"},
              "switching_costs": {"rating": 1, "quotes": [2], "reason": "cites an against quote"},
              "network_effects": zero, "cost_advantage": {"rating": 1, "quotes": [1], "reason": "any group"},
              "efficient_scale": zero, "overall": {"moat": "narrow", "reason": ""}}
    out = check_rating(rating, numbered)
    assert out["intangibles"] == {"rating": 2, "quotes": [1], "reason": "patents"}
    assert out["cost_advantage"]["rating"] == 1  # may cite a quote filed under another source
    assert out["switching_costs"]["rating"] == 0 and len(out["fixes"]) == 3


def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for v in schema.values():
            yield from _objects(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _objects(v)


def test_one_call_request_is_strict_cached_and_has_no_sampling_parameters():
    params = one_call_params("claude-haiku-5-5", "Item 1 text", effort="high")
    schema = params["output_config"]["format"]["schema"]
    assert all(o.get("additionalProperties") is False for o in _objects(schema))
    assert schema["required"][0] == "evidence" and "overall" in schema["required"]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["thinking"] == {"type": "adaptive"} and "temperature" not in params
    assert params["messages"][0]["content"].endswith("Item 1 text")


def test_verify_evidence_keeps_the_reply_numbering_for_citations():
    text = "We are the sole supplier of engines for this aircraft. Prices are set by the market."
    evidence = [{"kind": "intangibles", "quote": "We invented the wheel and nobody else did"},
                {"kind": "switching_costs", "quote": "We are the sole supplier of engines for this aircraft."},
                {"kind": "against", "quote": "Prices are set by the market"}]
    numbered, counts = verify_evidence(evidence, text)
    assert numbered == [(2, "switching_costs", "We are the sole supplier of engines for this aircraft."),
                        (3, "against", "Prices are set by the market")]
    assert counts == {"kept": 2, "not_in_text": 1, "duplicate": 0, "over_cap": 0}
    zero = {"rating": 0, "quotes": [], "reason": ""}
    rating = {"intangibles": {"rating": 2, "quotes": [1], "reason": "invented quote"},
              "switching_costs": {"rating": 2, "quotes": [2], "reason": "sole supplier"},
              "network_effects": zero, "cost_advantage": zero, "efficient_scale": zero,
              "overall": {"moat": "wide", "reason": ""}}
    out = check_rating(rating, numbered)
    assert out["intangibles"]["rating"] == 0 and out["switching_costs"]["rating"] == 2
