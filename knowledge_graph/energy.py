"""
Energy cost estimation for classifier training.

- Local compute (CPU / GPU / RAM on this machine) is estimated with CodeCarbon. See https://docs.codecarbon.io/latest/explanation/methodology/ for methodology.
- Remote LLM calls (via OpenRouter) are estimated with EcoLogits, from the
  model name and the number of output tokens. See https://ecologits.ai/latest/methodology/ for methodology.
"""

import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

import httpx
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from codecarbon.emissions_tracker import BaseEmissionsTracker
from codecarbon.input import DataSource
from ecologits.tracers.utils import llm_impacts
from ecologits.utils.range_value import RangeValue
from pydantic_ai.usage import RunUsage
from wandb.sdk.wandb_run import Run

from knowledge_graph.utils import get_logger

# Model-name prefixes we use (OpenRouter vendor, or pydantic-ai provider) the
# provider name EcoLogits uses in its model repository https://ecologits.ai/latest/tutorial/providers/.
_ECOLOGITS_PROVIDERS = {
    # Note that OpenAI does not disclose any information about model architecture and inference
    # infrastructure, so Ecologits estimates the environmental impacts with a very low
    # precision (https://ecologits.ai/latest/tutorial/providers/openai/).
    "openai": "openai",
    "google": "google_genai",
    "google-gla": "google_genai",
    "google-vertex": "google_genai",
    "anthropic": "anthropic",
    "mistralai": "mistralai",
    "mistral": "mistralai",
}
# Models we've already warned about, so each one is only warned about once per process
_warned_unestimated_models: set[str] = set()


def to_ecologits_model(model_name: str) -> tuple[str, str] | None:
    """
    Map one model name to an EcoLogits (provider, model) pair.

    e.g. "openrouter:google/gemini-3-flash-preview" -> ("google_genai", "gemini-3-flash-preview")

    Returns None if the provider isn't one EcoLogits knows about.
    """
    name = model_name.removeprefix("openrouter:")
    for separator in ("/", ":"):
        if separator in name:
            vendor, model = name.split(separator, 1)
            provider = _ECOLOGITS_PROVIDERS.get(vendor)
            return (provider, model) if provider else None
    return None


@dataclass
class EnergyReport:
    """
    Energy and emissions of one tracked block of work.

    Energy values are in kWh; emissions are in kgCO2e.
    """

    local_kwh: float = 0.0
    local_cpu_kwh: float = 0.0
    local_gpu_kwh: float = 0.0
    local_ram_kwh: float = 0.0
    duration_s: float = 0.0
    local_kgco2e: float = 0.0
    # EcoLogits gives a range, as model sizes / hardware of closed models are uncertain
    llm_kwh_min: float = 0.0
    llm_kwh_max: float = 0.0
    llm_kgco2e_min: float = 0.0
    llm_kgco2e_max: float = 0.0
    llm_requests: int = 0
    llm_output_tokens: int = 0
    # LLM requests whose model EcoLogits doesn't know, so aren't in llm_kwh
    llm_requests_not_estimated: int = 0

    @property
    def llm_kwh(self) -> float:
        """Midpoint of the EcoLogits estimate range"""
        return (self.llm_kwh_min + self.llm_kwh_max) / 2

    @property
    def total_kwh(self) -> float:
        """Measured local energy plus estimated remote LLM energy"""
        return self.local_kwh + self.llm_kwh

    @property
    def llm_kgco2e(self) -> float:
        """Midpoint of the EcoLogits emissions estimate range"""
        return (self.llm_kgco2e_min + self.llm_kgco2e_max) / 2

    @property
    def total_kgco2e(self) -> float:
        """Measured local emissions plus estimated remote LLM emissions"""
        return self.local_kgco2e + self.llm_kgco2e

    def __add__(self, other: "EnergyReport") -> "EnergyReport":
        """Add reports field by field"""
        return EnergyReport(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def to_dict(self) -> dict[str, float]:
        """All fields plus the derived totals"""
        return asdict(self) | {
            "llm_kwh": self.llm_kwh,
            "total_kwh": self.total_kwh,
            "llm_kgco2e": self.llm_kgco2e,
            "total_kgco2e": self.total_kgco2e,
        }

    def log(self, run: Run | None, phase: str, n_passages: int | None = None) -> None:
        """Log to the console, and to the W&B run summary if a run is given."""
        metrics = self.to_dict()
        if n_passages:
            metrics["total_kwh_per_passage"] = self.total_kwh / n_passages
            metrics["total_kgco2e_per_passage"] = self.total_kgco2e / n_passages

        get_logger().info(
            f"⚡ {phase} energy: {self.total_kwh:.6f} kWh, "
            f"{self.total_kgco2e:.6f} kgCO2e "
            f"(local {self.local_kwh:.6f} kWh / {self.local_kgco2e:.6f} kgCO2e "
            f"over {self.duration_s:.1f}s, "
            f"LLM {self.llm_kwh:.6f} kWh / {self.llm_kgco2e:.6f} kgCO2e "
            f"from {self.llm_requests} requests, "
            f"{self.llm_requests_not_estimated} not estimated)"
        )
        if run is not None:
            run.summary.update(
                {f"energy/{phase}/{key}": value for key, value in metrics.items()}
            )


# Country of the default AWS region our pipelines run in (eu-west-1 = Dublin), used when
# CodeCarbon can't look up location i.e. OfflineEmissionsTracker.
AWS_REGION_COUNTRY_ISO_CODE = "IRL"


_active_report: EnergyReport | None = None


def _location_lookup_check() -> bool:
    """Whether CodeCarbon's online IP-geolocation service is reachable."""
    try:
        httpx.get(DataSource().geo_js_url, timeout=2).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _make_energy_tracker() -> BaseEmissionsTracker:
    """
    Online tracker, otherwise offline tracker with the country of default AWS region.

    The online tracker doesn't raise if the lookup fails, it silently assumes
    Canada, so we check reachability up front. Location only affects CodeCarbon's
    CO2 estimate, not the energy (kWh) it measures.
    """
    settings = {"save_to_file": False, "log_level": "error"}
    if _location_lookup_check():
        try:
            return EmissionsTracker(**settings)
        except Exception as e:
            get_logger().warning(
                f"Online CodeCarbon tracker failed, using offline: {e}"
            )
    else:
        get_logger().info(
            "Location lookup unavailable, using offline CodeCarbon tracker "
            f"with country {AWS_REGION_COUNTRY_ISO_CODE}"
        )
    return OfflineEmissionsTracker(
        country_iso_code=AWS_REGION_COUNTRY_ISO_CODE, **settings
    )


@contextmanager
def track_compute_energy_usage() -> Generator[EnergyReport, None, None]:
    """
    Measure the energy used by the code inside the `with` block.

    Local compute (and GPU) is estimated by CodeCarbon; LLM calls made inside the block are
    added by `track_llm_energy_usage`. The report is filled in when the block exits.

    Example::

        with track_compute_energy_usage() as train_energy:
            classifier.fit(...)  # any track_llm_energy_usage() calls in here are included
        train_energy.log(wandb_run, phase="train")
    """
    global _active_report
    if _active_report is not None:
        raise RuntimeError("track_energy blocks can't be nested")

    report = EnergyReport()
    codecarbon: BaseEmissionsTracker | None = None
    try:
        codecarbon = _make_energy_tracker()
        codecarbon.start()
    except Exception as exc:  # energy tracking should never break a training run
        get_logger().warning(f"CodeCarbon failed to start: {exc}")
        codecarbon = None

    _active_report = report
    try:
        yield report
    finally:
        _active_report = None
        if codecarbon is not None:
            try:
                codecarbon.stop()
                data = codecarbon.final_emissions_data
                if data is not None:
                    report.local_kwh = data.energy_consumed
                    report.local_cpu_kwh = data.cpu_energy
                    report.local_gpu_kwh = data.gpu_energy
                    report.local_ram_kwh = data.ram_energy
                    report.duration_s = data.duration
                    report.local_kgco2e = data.emissions
            except Exception as exc:
                get_logger().warning(f"CodeCarbon failed to stop: {exc}")


def track_llm_energy_usage(model_name: str, usage: RunUsage) -> None:
    """
    Add the estimated energy of a pydantic-ai agent run to the active tracker.

    Does nothing outside a `track_compute_energy_usage`, so it's safe to call from
    classifier code that also runs outside of training.

    EcoLogits' automatic patching (EcoLogits.init(providers=["openai"])) does not work for us.
    pydantic-ai sends OpenRouter calls through the OpenAI SDK. EcoLogits therefore looks up the
    model as provider openai and model e.g. gemini-3-flsh-preview and returns `model-not-registered`.
    pydantic-ai also drops the extra .impacts attribute when it builds its own response.
    Therefore we call ecologits.tracers.utils.llm_impacts() and pass the mapped provider and model with
    the output-token count from AgentRunResult.usage().

    :param model_name: Our model name, e.g. "openrouter:google/gemini-3-flash-preview"
    :param usage: `result.usage()` from a pydantic-ai AgentRunResult
    """
    report = _active_report
    if report is None:
        return

    report.llm_requests += usage.requests
    report.llm_output_tokens += usage.output_tokens

    model = to_ecologits_model(model_name)
    impacts = (
        llm_impacts(
            provider=model[0],
            model_name=model[1],
            output_token_count=usage.output_tokens,
            # Use Ecologits per-token estimate.
            # (I think concurrent batch calls would
            # make a measured value misleading anyway?)
            request_latency=math.inf,
        )
        if model
        else None
    )
    if impacts is None or impacts.energy is None:
        report.llm_requests_not_estimated += usage.requests
        if model_name not in _warned_unestimated_models:
            _warned_unestimated_models.add(model_name)
            get_logger().warning(
                f"⚠️ No EcoLogits energy estimate for {model_name}: its energy is "
                "excluded from total_kwh (see llm_requests_not_estimated)."
            )
        return

    energy = impacts.energy.value
    if isinstance(energy, RangeValue):
        report.llm_kwh_min += energy.min
        report.llm_kwh_max += energy.max
    else:
        report.llm_kwh_min += energy
        report.llm_kwh_max += energy
    if impacts.usage is not None and impacts.usage.gwp is not None:
        gwp = impacts.usage.gwp.value
        if isinstance(gwp, RangeValue):
            report.llm_kgco2e_min += gwp.min
            report.llm_kgco2e_max += gwp.max
        else:
            report.llm_kgco2e_min += gwp
            report.llm_kgco2e_max += gwp
