import pytest
from pydantic_ai.usage import RunUsage

from knowledge_graph.energy import (
    EnergyReport,
    to_ecologits_model,
    track_energy,
    track_llm,
)


@pytest.mark.parametrize(
    "model_name, expected",
    [
        (
            "openrouter:google/gemini-3-flash-preview",
            ("google_genai", "gemini-3-flash-preview"),
        ),
        ("openrouter:openai/gpt-5", ("openai", "gpt-5")),
        ("openai:gpt-5", ("openai", "gpt-5")),
        ("openrouter:some-unknown-vendor/model", None),
        ("llama3", None),
    ],
)
def test_to_ecologits_model(model_name, expected):
    assert to_ecologits_model(model_name) == expected


def test_track_llm_is_a_no_op_without_tracker():
    track_llm("openrouter:openai/gpt-5", RunUsage(requests=1, output_tokens=100))


def test_reports_add_up():
    a = EnergyReport(local_kwh=1.0, llm_kwh_min=0.5, llm_kwh_max=1.5, llm_requests=2)
    b = EnergyReport(local_kwh=2.0, llm_requests=1)
    total = sum([a, b], EnergyReport())
    assert total.local_kwh == 3.0
    assert total.llm_requests == 3
    assert total.total_kwh == 3.0 + 1.0


def test_track_energy_measures_local_and_estimates_llm():
    with track_energy() as report:
        sum(i * i for i in range(1_000_000))  # some local work
        track_llm(
            "openrouter:google/gemini-3-flash-preview",
            RunUsage(requests=1, output_tokens=500),
        )
        track_llm("openrouter:unknown/model", RunUsage(requests=1, output_tokens=500))

    assert report.local_kwh > 0
    assert 0 < report.llm_kwh_min <= report.llm_kwh_max
    assert report.llm_requests == 2
    assert report.llm_requests_not_estimated == 1


def test_unknown_model_warns_once(caplog):
    with track_energy():
        for _ in range(3):
            track_llm(
                "openrouter:mystery/model-x", RunUsage(requests=1, output_tokens=10)
            )

    warnings = [
        r for r in caplog.records if "No EcoLogits energy estimate" in r.message
    ]
    assert len(warnings) == 1


def test_falls_back_to_offline_tracker_when_location_lookup_fails(monkeypatch):
    from codecarbon import OfflineEmissionsTracker

    from knowledge_graph import energy

    monkeypatch.setattr(energy, "_location_lookup_check", lambda: False)
    assert isinstance(energy._make_energy_tracker(), OfflineEmissionsTracker)
