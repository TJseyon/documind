from app.generation.grounding import check_groundedness


CONTEXT = [
    "Employees receive 20 vacation days and 10 sick days per calendar year.",
    "The SR-40 drone has a maximum payload of 2.5 kilograms and 28 minutes of flight time.",
]


def test_answer_with_supported_numbers_and_citation_is_grounded():
    answer = "Employees get 20 vacation days per year [chunk_1]."
    result = check_groundedness(answer, CONTEXT, require_citation=True)
    assert result.grounded is True
    assert result.unsupported_claims == []


def test_answer_with_fabricated_number_is_flagged_unsupported():
    answer = "Employees get 45 vacation days per year [chunk_1]."
    result = check_groundedness(answer, CONTEXT, require_citation=True)
    assert result.grounded is False
    assert "45" in result.unsupported_claims


def test_answer_without_any_citation_is_treated_as_unsupported():
    answer = "Employees get 20 vacation days per year."
    result = check_groundedness(answer, CONTEXT, require_citation=True)
    assert result.has_citation is False
    assert result.grounded is False


def test_correct_decline_is_treated_as_grounded():
    answer = "I don't have enough information in the provided documents to answer this."
    result = check_groundedness(answer, CONTEXT, require_citation=True)
    assert result.grounded is True


def test_small_trivial_numbers_do_not_trigger_false_positive():
    # "1" and "2" are common as list markers and shouldn't require exact context match
    answer = "There are 2 main policies described [chunk_1]."
    result = check_groundedness(answer, CONTEXT, require_citation=True)
    assert result.unsupported_claims == []


def test_citation_not_required_when_flag_disabled():
    answer = "Employees get 20 vacation days per year."
    result = check_groundedness(answer, CONTEXT, require_citation=False)
    assert result.grounded is True
