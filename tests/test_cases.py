"""Checks for deterministic fixture loading and labeled record integrity."""

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from cerberus.redteam.cases import CASESET_VERSION, Case, Category, _CasesFile, load_cases


def test_corpus_diversity_and_stability() -> None:
    """The wheel corpus covers every declared category including benign examples."""
    cases = load_cases()
    assert load_cases() is cases
    assert {case.category for case in cases} == set(get_args(Category))
    assert {case.expected_block for case in cases} == {True, False}
    assert any(case.documents for case in cases) and any(case.tool_outputs for case in cases)
    assert all(case.canary for case in cases if case.expected_block)


def test_invalid_markers_and_controls() -> None:
    """Reject accidental copyable canaries and contradictory control labels."""
    case = load_cases()[0].model_dump()
    for changes in (
        {"text": case["canary"]},
        {"tool_outputs": (case["canary"],)},
        {"category": "benign_control"},
        {"canary": "private raw field"},
        {"severity": "unrecognized"},
        {"expected_block": "false"},
    ):
        with pytest.raises(ValidationError):
            Case.model_validate(case | changes)


def test_duplicate_records_and_versions_are_rejected() -> None:
    """IDs and markers remain reliable joins across runs and recordings."""
    case = load_cases()[0].model_dump(mode="json")
    with pytest.raises(ValidationError, match="Duplicate"):
        _CasesFile.model_validate_json(
            json.dumps({"version": CASESET_VERSION, "cases": [case, case | {"id": "new-record"}]})
        )
    with pytest.raises(ValidationError, match="Duplicate"):
        _CasesFile.model_validate_json(
            json.dumps({"version": CASESET_VERSION, "cases": [case, case | {"canary": None}]})
        )
    with pytest.raises(ValidationError):
        _CasesFile.model_validate_json(json.dumps({"version": "9", "cases": [case]}))
