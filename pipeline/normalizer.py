import re

# Expanding these here, before translation, means the model never has to guess
# what an abbreviation stands for. Only unambiguous ones belong in this table —
# things like "St." (Saint or Street?) and "No." (which might just be the word
# "No.") caused more trouble than they saved, so they're deliberately left out.
_ABBREVIATIONS = {
    "ETA": "estimated time of arrival",
    "ETD": "estimated time of departure",
    "ASAP": "as soon as possible",
    "DIY": "do it yourself",
    "FAQ": "frequently asked questions",
    "IMO": "in my opinion",
    "RSVP": "please respond",
    "TBD": "to be decided",
    "TBA": "to be announced",
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Prof.": "Professor",
    "vs.": "versus",
    "approx.": "approximately",
}

# A trailing \b doesn't work after a literal "." (no word boundary between two
# non-word characters), so the dotted entries need a lookahead instead.
_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _ABBREVIATIONS) + r")(?=\W|$)"
)

_AM_PM_RE = re.compile(r"(\d)\s*([ap])\.?m\.?\b", re.IGNORECASE)


def _expand_abbreviations(text: str) -> str:
    return _ABBREV_RE.sub(lambda m: _ABBREVIATIONS[m.group(0)], text)


def _normalize_numbers_and_times(text: str) -> str:
    # Whisper writes times a few different ways (5pm, 5 p.m., 10:30am) —
    # settle on one shape so the translator sees consistent input.
    return _AM_PM_RE.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}M", text)


def normalize(text: str) -> str:
    """Expand abbreviations and clean up time tokens."""
    if not text:
        return text
    text = _normalize_numbers_and_times(text)
    text = _expand_abbreviations(text)
    return re.sub(r"\s+", " ", text).strip()
