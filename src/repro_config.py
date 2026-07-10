"""Shared configuration for model-scale replications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelProfile:
    """A model identity plus the repository paths used for its artifacts."""

    slug: str
    hf_name: str
    tl_name: str
    label: str
    checkpoint_names: tuple[str, ...]

    @property
    def experiments_dir(self) -> Path:
        return ROOT / "experiments" / self.slug

    @property
    def tables_dir(self) -> Path:
        return ROOT / "results" / self.slug / "tables"

    @property
    def processed_dir(self) -> Path:
        return ROOT / "results" / self.slug / "processed"

    @property
    def figures_dir(self) -> Path:
        return ROOT / "paper" / "figures" / self.slug

    @property
    def report_dir(self) -> Path:
        return ROOT / "reports" / self.slug


PROFILES: dict[str, ModelProfile] = {
    "gpt2_small": ModelProfile(
        slug="gpt2_small",
        hf_name="openai-community/gpt2",
        tl_name="gpt2-small",
        label="GPT-2 Small",
        checkpoint_names=("checkpoint-100", "final_model"),
    ),
    "gpt2_medium": ModelProfile(
        slug="gpt2_medium",
        hf_name="openai-community/gpt2-medium",
        tl_name="gpt2-medium",
        label="GPT-2 Medium",
        checkpoint_names=("checkpoint-100", "checkpoint-200", "checkpoint-220", "final_model"),
    ),
    "gpt2_large": ModelProfile(
        slug="gpt2_large",
        hf_name="openai-community/gpt2-large",
        tl_name="gpt2-large",
        label="GPT-2 Large",
        checkpoint_names=("checkpoint-100", "final_model"),
    ),
}


def profile_choices() -> list[str]:
    """Return supported model-profile slugs for command-line interfaces."""

    return sorted(PROFILES)


def get_profile(slug: str) -> ModelProfile:
    try:
        return PROFILES[slug]
    except KeyError as exc:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown model profile '{slug}'. Valid profiles: {valid}") from exc


def ensure_profile_dirs(profile: ModelProfile) -> None:
    for path in [
        profile.experiments_dir,
        profile.tables_dir,
        profile.processed_dir,
        profile.figures_dir,
        profile.report_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
