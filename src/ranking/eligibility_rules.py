"""Rule-based eligibility compatibility between an NGO profile and an opportunity.

Eligibility is treated as a hard-ish constraint rather than a similarity score
(per blueprint Section D.1/F.1). Because eligible_applicants/_type are
US-specific categories (Section C.5) that do not map cleanly onto Indian legal
structures, this module implements a *best-effort compatibility* check with
three explicit outcomes -- COMPATIBLE, INCOMPATIBLE, UNKNOWN -- rather than
pretending precision that the data cannot support. UNKNOWN is treated as a
soft pass (opportunity is shown, but not asserted as eligible).
"""
from dataclasses import dataclass
from enum import Enum


class EligibilityResult(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


# NGO registration types (Indian context) mapped to keyword signals that
# commonly appear in the US eligible_applicants/_type free text. This is a
# deliberately coarse mapping -- see docs/data_audit_report.md for why a
# precise US<->India eligibility taxonomy does not exist yet.
NGO_TYPE_KEYWORDS = {
    "trust": ["nonprofit", "non-profit", "non-governmental", "nongovernmental", "others", "unrestricted"],
    "society": ["nonprofit", "non-profit", "non-governmental", "nongovernmental", "others", "unrestricted"],
    "section_8_company": ["nonprofit", "non-profit", "private", "others", "unrestricted"],
    "trust_or_society_or_section_8": ["nonprofit", "non-profit", "non-governmental", "nongovernmental", "others", "unrestricted"],
}

# Keywords that signal an opportunity is restricted to entities an Indian NGO
# structurally cannot be (US state/local/tribal government, individuals, etc.).
HARD_EXCLUSION_KEYWORDS = [
    "state governments",
    "county governments",
    "city or township governments",
    "special district governments",
    "independent school districts",
    "public and state controlled institutions of higher education",
    "native american tribal",
    "individuals",
    "universities only",
    "accredited universities only",
    "higher education institutions only",
    "academic institutions only",
    "for-profit organizations only",
    "for profit organizations only",
]


@dataclass
class EligibilityCheck:
    result: EligibilityResult
    reason: str


def check_eligibility(ngo_registration_type: str | None, eligibility_text: str | None) -> EligibilityCheck:
    """Evaluate whether an NGO of the given registration type is plausibly
    eligible for an opportunity, based on its (lowercased) eligibility_text."""
    if not eligibility_text or not eligibility_text.strip():
        return EligibilityCheck(EligibilityResult.UNKNOWN, "No eligibility text available on this opportunity")

    text = eligibility_text.lower()

    for excluded in HARD_EXCLUSION_KEYWORDS:
        if excluded in text and "nonprofit" not in text and "non-profit" not in text:
            return EligibilityCheck(
                EligibilityResult.INCOMPATIBLE,
                f"Eligibility text restricts applicants to '{excluded}', which does not match an Indian NGO structure",
            )

    if not ngo_registration_type:
        return EligibilityCheck(EligibilityResult.UNKNOWN, "NGO registration type not provided")

    keywords = NGO_TYPE_KEYWORDS.get(
        ngo_registration_type.lower().replace(" ", "_"), NGO_TYPE_KEYWORDS["trust_or_society_or_section_8"]
    )
    if any(kw in text for kw in keywords):
        matched = next(kw for kw in keywords if kw in text)
        return EligibilityCheck(
            EligibilityResult.COMPATIBLE,
            f"Eligibility text includes '{matched}', broadly compatible with a registered NGO",
        )

    return EligibilityCheck(
        EligibilityResult.UNKNOWN,
        "Eligibility text does not clearly confirm or exclude an NGO of this type",
    )


def eligibility_score(check: EligibilityCheck) -> float:
    """Map an eligibility check to a numeric multiplier used in the weighted
    ranking score (score.py). INCOMPATIBLE opportunities are down-weighted
    heavily rather than hard-removed, so they can still surface if nothing
    else matches (avoids an empty result set from an overzealous filter).

    UNKNOWN is deliberately close to 1.0 (a mild, not heavy, discount): the
    source simply didn't state eligibility clearly, which is "insufficient
    data" -- not evidence the NGO is ineligible -- and must not be scored
    as if it were a genuine mismatch (interview.txt: "unknown eligibility
    should still be surfaced, not treated as mismatch")."""
    return {
        EligibilityResult.COMPATIBLE: 1.0,
        EligibilityResult.UNKNOWN: 0.85,
        EligibilityResult.INCOMPATIBLE: 0.1,
    }[check.result]
