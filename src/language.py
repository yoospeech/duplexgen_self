"""Output-language instructions shared by dialogue generation stages."""

SUPPORTED_LANGUAGES = ("en", "id")
LANGUAGE = "en"


def set_language(language: str) -> None:
    global LANGUAGE
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    LANGUAGE = language


def instruction() -> str:
    if LANGUAGE == "en":
        return ""
    return """

### Bahasa Indonesia requirements
- Write every spoken utterance in natural Bahasa Indonesia, even when the source is English.
- Treat the source dialogue as the only source of facts. Do not add names, dates, prices,
  policies, promises, or background details that are absent from it.
- Preserve each speaker's role, intent, constraints, and all factual details.
- Keep JSON keys, role values, [TAKE_FLOOR], and [BACKCHANNEL] exactly unchanged.
- Use conversational Indonesian with a consistent level of formality and consistent
  pronouns. Prefer neutral Indonesian unless the scenario clearly calls for another register.
- Use Indonesian fillers and acknowledgements, such as "hmm", "iya", "baik", and
  "oh begitu", only where they sound natural. Do not imitate the English examples.
- Express numbers, dates, currencies, and abbreviations in a form that Indonesian speakers
  would naturally say aloud, without changing their values.
"""
