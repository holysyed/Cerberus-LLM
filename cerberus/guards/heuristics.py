"""Bounded, multilingual prompt-injection detection for untrusted text."""

import base64
import binascii
import re
import unicodedata

MAX_SOURCE_CHARS = 32_768

# Normalize common lookalikes only for inspection. Never rewrite the user's input.
_LOOKALIKES = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "у": "y",
        "х": "x",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "һ": "h",
        "к": "k",
        "м": "m",
        "т": "t",
        "α": "a",
        "ο": "o",
        "ι": "i",
        "κ": "k",
        "ρ": "p",
        "τ": "t",
        "χ": "x",
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
    }
)
_HIDDEN_FORMATTING = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")

_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\s+"
    r"(?:(?:all|any|the|your|my|given)\s+){0,3}"
    r"(?:previous|prior|earlier|above|preceding|system|safety|developer)\s+"
    r"(?:instructions?|rules?|prompts?|guidelines?|messages?|polic(?:y|ies))\b"
    r"|(?:पिछल[ेी]|पूर्व|पहले के|ऊपर दिए गए)\s+(?:(?:के|सभी|सारे)\s+){0,2}"
    r"(?:निर्देशों?|निर्देश|नियमों?)\s+(?:को\s+)?(?:अनदेखा|नज़रअंदाज़|नजरअंदाज|भूल)"
    r"|(?:మునుపటి|పూర్వపు|గత|ముందరి)\s+(?:అన్ని\s+)?"
    r"(?:సూచనలను|నిర్దేశాలను|నియమాలను)\s+(?:విస్మరించ|పట్టించుకోకు|మరచిపో)"
    r"|(?:پچھل[یے]|سابقہ|پہلی)\s+(?:(?:کی|تمام|ساری)\s+){0,2}"
    r"(?:ہدایات|قواعد)\s+(?:کو\s+)?(?:نظر\s*انداز|بھول)"
)
_ROLE = re.compile(
    r"\b(?:you are now|act as|pretend to be|assume the role of|switch to)\s+"
    r"(?:(?:a|an|the)\s+)?(?:unrestricted|unfiltered|uncensored|developer mode|dan\b)"
    r"|\b(?:enable|activate|enter)\s+(?:developer|unrestricted|jailbreak)\s+mode\b"
    r"|(?:अब तुम|अब आप)\s+(?:एक\s+)?(?:अप्रतिबंधित|बिना नियमों)"
    r"|(?:ఇప్పుడు|ఇకపై)\s+నువ్వు\s+(?:పరిమితులు లేని|నియమాలు లేని)"
    r"|(?:اب تم|اب آپ)\s+(?:ایک\s+)?(?:غیر محدود|بلا پابندی)"
)
_EXTRACTION = re.compile(
    r"\b(?:print|reveal|show|repeat|disclose|return|quote|expose|display|output)\s+"
    r"(?:(?:me|the|your|entire|original|full|hidden|initial|complete|exact)\s+){0,5}"
    r"(?:system\s+(?:prompt|message|instructions?)|developer\s+(?:message|instructions?)"
    r"|hidden\s+instructions?)\b"
    r"|(?:सिस्टम|सिस्ट[मॅ])\s+(?:प्रॉम्प्ट|निर्देश)\s+(?:को\s+)?(?:दिखा|बता|लिख)"
    r"|(?:సిస్టమ్|సిస్టం)\s+(?:ప్రాంప్ట్|సూచనల)(?:ను|ని)?\s+(?:చూప|తెలుప|వెల్లడించ)"
    r"|سسٹم\s+(?:پرامپٹ|ہدایات)\s+(?:کو\s+)?(?:دکھا|بتا|ظاہر)"
)
_PRIVILEGED_DELIMITER = re.compile(
    r"<\|(?:im_start|start_header_id)\|>\s*(?:system|developer)\b"
    r"|(?:\A|\n)[ \t]*(?:\[(?:system|/inst)\]|<<sys>>)\s*\S"
    r"|(?:\A|\n)#{2,4}\s*(?:system|developer)\s*:\s*\S"
)
_RULES = (
    ("instruction_override", 0.92, _OVERRIDE),
    ("role_override", 0.89, _ROLE),
    ("prompt_extraction", 0.92, _EXTRACTION),
    ("privileged_delimiter", 0.88, _PRIVILEGED_DELIMITER),
)

# Match entire runs to avoid repeatedly decoding overlapping substrings.
_TOKEN = re.compile(r"(?<![A-Za-z0-9_+/-])[A-Za-z0-9_+/-]{12,}={0,2}")
_HEX = re.compile(r"[0-9A-Fa-f]+")


def _normalize(text: str) -> str:
    """Compatibility-normalize text and fold formatting and common lookalikes."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _HIDDEN_FORMATTING.sub("", folded).translate(_LOOKALIKES)


def _inspect(text: str) -> tuple[float, tuple[str, ...]]:
    """Return the largest signal and all matching rule identifiers."""
    score = 0.0
    reasons: list[str] = []
    normalized = _normalize(text)
    for reason, confidence, pattern in _RULES:
        if pattern.search(normalized):
            reasons.append(reason)
            score = max(score, confidence)
    return score, tuple(reasons)


def _decode(token: str) -> tuple[str, str] | None:
    """Decode one token only when it represents UTF-8 text."""
    try:
        hex_candidate = token[2:] if token.startswith(("0x", "0X")) else token
        if len(hex_candidate) % 2 == 0 and _HEX.fullmatch(hex_candidate):
            return "hex", bytes.fromhex(hex_candidate).decode()
        padded = token + "=" * (-len(token) % 4)
        return "base64", base64.b64decode(padded, altchars=b"-_", validate=True).decode()
    except (ValueError, UnicodeError, binascii.Error):
        return None


def heuristic_score(text: str) -> tuple[float, tuple[str, ...]]:
    """Score one source, inspecting literal text and one level of encoded text."""
    if len(text) > MAX_SOURCE_CHARS:
        return 1.0, ("input_length_limit",)
    score, matches = _inspect(text)
    reasons = list(matches)
    for match in _TOKEN.finditer(text):
        decoded = _decode(match.group())
        if decoded is None:
            continue
        encoding, candidate = decoded
        decoded_score, decoded_reasons = _inspect(candidate)
        if decoded_reasons:
            score = max(score, decoded_score)
            reasons.extend((*decoded_reasons, f"encoded_{encoding}"))
    return score, tuple(dict.fromkeys(reasons))
