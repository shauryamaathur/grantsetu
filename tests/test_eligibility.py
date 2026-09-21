from src.ranking.eligibility_rules import (
    EligibilityResult,
    check_eligibility,
    eligibility_score,
)


def test_nonprofit_keyword_marks_compatible():
    check = check_eligibility("trust", "nonprofits with 501(c)(3) status")
    assert check.result == EligibilityResult.COMPATIBLE


def test_state_government_only_marks_incompatible():
    check = check_eligibility("trust", "state governments")
    assert check.result == EligibilityResult.INCOMPATIBLE


def test_missing_eligibility_text_is_unknown():
    check = check_eligibility("trust", None)
    assert check.result == EligibilityResult.UNKNOWN


def test_missing_registration_type_is_unknown_when_text_is_ambiguous():
    check = check_eligibility(None, "for-profit small businesses only")
    assert check.result == EligibilityResult.UNKNOWN


def test_eligibility_score_ordering():
    compatible = eligibility_score(check_eligibility("trust", "nonprofits"))
    unknown = eligibility_score(check_eligibility("trust", None))
    incompatible = eligibility_score(check_eligibility("trust", "state governments"))
    assert compatible > unknown > incompatible
