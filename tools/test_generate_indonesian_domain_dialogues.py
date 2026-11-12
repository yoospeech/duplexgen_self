import pytest

from tools.generate_indonesian_domain_dialogues import (
    balanced_domain_plan,
    parse_dialogue_response,
)


def test_balanced_domain_plan_is_reproducible_and_balanced():
    domains = ["healthcare", "travel", "education"]
    first = balanced_domain_plan(domains, 8, seed=17)
    second = balanced_domain_plan(domains, 8, seed=17)

    assert first == second
    counts = [first.count(domain) for domain in domains]
    assert max(counts) - min(counts) <= 1


def test_parse_dialogue_response_validates_and_normalizes():
    raw = """{
      "topic": "mengubah jadwal dokter",
      "scenario": "pasien mengubah jadwal pemeriksaan",
      "history": [
        {"role": "user", "content": " Saya ingin mengubah jadwal. "},
        {"role": "assistant", "content": "Baik, jadwal mana yang Anda inginkan?"}
      ]
    }"""

    result = parse_dialogue_response(raw, "healthcare", 2, 4)

    assert result["history"][0]["content"] == "Saya ingin mengubah jadwal."
    assert result["meta"]["domain"] == "healthcare"
    assert result["meta"]["synthetic"] is True


def test_parse_dialogue_response_rejects_wrong_role_order():
    raw = """{
      "topic": "hotel",
      "scenario": "memesan kamar",
      "history": [
        {"role": "assistant", "content": "Halo"},
        {"role": "user", "content": "Halo"}
      ]
    }"""

    with pytest.raises(ValueError, match="role=user"):
        parse_dialogue_response(raw, "travel", 2, 4)
