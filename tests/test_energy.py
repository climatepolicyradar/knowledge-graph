from unittest.mock import patch

import pytest
from pydantic_ai.usage import RunUsage

from knowledge_graph.energy import (
    EnergyReport,
    to_ecologits_model,
    track_compute_energy_usage,
    track_llm_energy_usage,
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


def test_track_llm_energy_usage_is_a_no_op_without_tracker():
    """Calling it outside a tracked block must not raise or estimate anything."""
    with patch("knowledge_graph.energy.llm_impacts") as mock_llm_impacts:
        track_llm_energy_usage(
            "openrouter:openai/gpt-5", RunUsage(requests=1, output_tokens=100)
        )
    mock_llm_impacts.assert_not_called()


def test_reports_add_up():
    a = EnergyReport(
        local_kwh=1.0,
        llm_kwh_min=0.5,
        llm_kwh_max=1.5,
        llm_requests=2,
        local_kgco2e=0.2,
        llm_kgco2e_min=0.1,
        llm_kgco2e_max=0.3,
        local_water_l=0.5,
        llm_water_l_min=1.0,
        llm_water_l_max=3.0,
    )
    b = EnergyReport(local_kwh=2.0, llm_requests=1, local_kgco2e=0.4, local_water_l=0.5)
    total = sum([a, b], EnergyReport())
    assert total.local_kwh == 3.0
    assert total.llm_requests == 3
    assert total.total_kwh == 3.0 + 1.0
    assert total.total_kgco2e == pytest.approx(0.6 + 0.2)
    assert total.total_water_l == pytest.approx(1.0 + 2.0)


def test_track_compute_energy_usage_usage_measures_local_and_estimates_llm():
    with track_compute_energy_usage() as report:
        sum(i * i for i in range(1_000_000))  # some local work
        track_llm_energy_usage(
            "openrouter:google/gemini-3-flash-preview",
            RunUsage(requests=1, output_tokens=500),
        )
        track_llm_energy_usage(
            "openrouter:unknown/model", RunUsage(requests=1, output_tokens=500)
        )

    assert report.local_kwh > 0
    assert 0 < report.llm_kwh_min <= report.llm_kwh_max
    assert report.llm_requests == 2
    assert report.llm_requests_not_estimated == 1
    assert report.local_kgco2e > 0
    assert 0 < report.llm_kgco2e_min <= report.llm_kgco2e_max
    assert report.total_kgco2e == report.local_kgco2e + report.llm_kgco2e
    assert report.local_water_l > 0
    assert 0 < report.llm_water_l_min <= report.llm_water_l_max
    assert report.total_water_l == report.local_water_l + report.llm_water_l


def test_unknown_model_warns_once():
    with patch("knowledge_graph.energy.get_logger") as mock_get_logger:
        with track_compute_energy_usage():
            for _ in range(3):
                track_llm_energy_usage(
                    "openrouter:mystery/model-x", RunUsage(requests=1, output_tokens=10)
                )

    warnings = [
        str(call.args[0])
        for call in mock_get_logger().warning.call_args_list
        if "No EcoLogits energy estimate" in str(call.args[0])
    ]
    assert len(warnings) == 1


def test_falls_back_to_offline_tracker_when_location_lookup_fails(monkeypatch):
    from codecarbon import OfflineEmissionsTracker

    from knowledge_graph import energy

    monkeypatch.setattr(energy, "_location_lookup_check", lambda: False)
    assert isinstance(energy._make_energy_tracker(), OfflineEmissionsTracker)
