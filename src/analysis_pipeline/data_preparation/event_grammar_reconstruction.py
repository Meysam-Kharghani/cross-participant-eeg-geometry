#!/usr/bin/env python3
# Public portability snapshot.
# Analytic event-pairing and README-sequence logic is preserved from the frozen source.
# Machine-specific defaults and historical execution identifiers were normalized for public release.
# Original frozen source SHA-256: 3d065b93307cfcea3682e7f7fdd87d85c24ac5418e0ff9948c04cafc4cca605c
"""
Event-grammar reconstruction and direct README sequence audit.

Read-only objectives
--------------------
1. Confirm the source-code defect that assigned context from behavioral row
   number (1–21 imagery, 22–42 video).
2. Parse stimulus sequences directly from the dataset README, rather than
   inheriting the superseded canonical builder.
3. Learn robust stimulus-to-rating timing priors from structurally clean
   discovery recordings.
4. Reconstruct true non-overlapping stimulus→rating trial pairs from EEG
   annotations with a capacity-constrained dynamic program.
5. Remove duplicated/chattering triggers while preserving the documented
   number of imagery/video behavioral trials.
6. Validate the reconstruction against metadata in unaffected participants.
7. Map reconstructed event trials back to raw behavioral trial_number.
8. Recompute context_trial_index in acquisition order.
9. Attach a stimulus candidate only when README sequence alignment is
   structurally identifiable.
10. Write a physiology sample-clock contract from the already-audited timing
    evidence.
11. Leave every corrected table explicitly non-authoritative.

No EEG/GSR/PPG amplitudes are read. Validation and confirmation outcomes and
signals remain closed.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import json
import math
import re
import statistics
import time
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(
    "<EMOEEG_DATASET_ROOT>"
)
DEFAULT_PROJECT = Path(
    "<ANALYSIS_ROOT>"
)

DISCOVERY = "discovery"

BEHAVIOR_REL = Path(
    "outputs/tables/behavior_discovery_rows.csv"
)
METADATA_REL = Path(
    "outputs/tables/metadata_schedule_discovery.csv"
)
GLOBAL_EVENTS_REL = Path(
    "outputs/tables/eeg_events_global_time.csv"
)
METADATA_INDEX_REL = Path(
    "outputs/tables/metadata_context_index_audit.csv"
)
METADATA_AUDIT_SUMMARY_REL = Path(
    "outputs/qc/canonical_builder_audit_summary.json"
)
PHYSIO_CLOCK_REL = Path(
    "outputs/tables/physio_clock_qc.csv"
)
BUILDER_REL = Path(
    "<CANONICAL_BUILDER_SOURCE>"
)
COMBINED_METADATA_REL = Path(
    "derivatives/metadata/trials_metadata_basic_sub01-05.csv"
)

CONTEXTS = [
    "imagery",
    "video",
]

VALID_EVENT_METHODS = {
    "readme_exact_type_id",
    "readme_modulo_10",
}

CANONICAL_STIMULI = {
    "sad4",
    "sad5",
    "sad8",
    "dis4",
    "dis5",
    "dis8",
    "fear4",
    "fear5",
    "fear8",
    "neu4",
    "neu5",
    "neu8",
    "joy4",
    "joy5",
    "joy8",
    "ten4",
    "ten5",
    "ten8",
    "ins4",
    "ins5",
    "ins8",
}

CATEGORY_MAP = {
    "sad": "sadness",
    "dis": "disgust",
    "fear": "fear",
    "neu": "neutral",
    "joy": "joy",
    "ten": "tenderness",
    "ins": "inspiration",
}

VARIANTS = {
    "primary": {
        "maximum_delay_sec": 650.0,
        "ratings_per_stimulus": 6,
        "intervening_stimulus_penalty": 0.4,
        "intervening_rating_penalty": 0.6,
    },
    "tight": {
        "maximum_delay_sec": 500.0,
        "ratings_per_stimulus": 5,
        "intervening_stimulus_penalty": 0.5,
        "intervening_rating_penalty": 0.8,
    },
    "wide": {
        "maximum_delay_sec": 700.0,
        "ratings_per_stimulus": 8,
        "intervening_stimulus_penalty": 0.3,
        "intervening_rating_penalty": 0.5,
    },
}

MINIMUM_STIMULUS_TO_RATING_SEC = 30.0


@dataclass
class SubjectSequence:
    imagery: list[str]
    video: list[str]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only event-grammar reconstruction and direct README "
            "sequence audit."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
    )
    parser.add_argument(
        "--metadata-match-tolerance-sec",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S"
    )


def normalize_subject(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(
        r"(?:sub[-_ ]?)?(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return f"sub-{int(match.group(1)):02d}"

    return text


def normalize_context(value: object) -> str:
    text = str(value).strip().lower()

    if text in {
        "ima",
        "imagery",
        "ses-ima",
    }:
        return "imagery"

    if text in {
        "vid",
        "video",
        "ses-vid",
    }:
        return "video"

    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require_files(paths: Iterable[Path]) -> None:
    missing = [
        path
        for path in paths
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Required files are missing:\n  - "
            + "\n  - ".join(
                str(path)
                for path in missing
            )
        )


def stimulus_category(
    stimulus_id: object,
) -> str | None:
    if pd.isna(
        stimulus_id
    ):
        return None

    text = str(
        stimulus_id
    )

    for prefix, category in (
        CATEGORY_MAP.items()
    ):
        if text.startswith(
            prefix
        ):
            return category

    return "unknown"


def literal_list_from_line(
    line: str,
) -> list[str] | None:
    match = re.search(
        r"`(\[.*\])`",
        line,
    )

    if not match:
        return None

    try:
        value = ast.literal_eval(
            match.group(
                1
            )
        )
    except Exception:
        return None

    if not isinstance(
        value,
        list,
    ):
        return None

    return [
        str(
            item
        )
        for item in value
    ]


def parse_readme_sequences(
    readme_path: Path,
) -> dict[
    str,
    SubjectSequence,
]:
    text = readme_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    lines = text.splitlines()
    sequences: dict[
        str,
        SubjectSequence,
    ] = {}

    for line in lines:
        match = re.match(
            r"^\s*(\d{1,2})\.\s+`(\[.*\])`\s*$",
            line,
        )

        if not match:
            continue

        number = int(
            match.group(
                1
            )
        )

        if not (
            1
            <= number
            <= 53
        ):
            continue

        try:
            sequence = [
                str(
                    item
                )
                for item
                in ast.literal_eval(
                    match.group(
                        2
                    )
                )
            ]
        except Exception:
            continue

        subject = (
            f"sub-{number:02d}"
        )
        sequences[
            subject
        ] = SubjectSequence(
            sequence,
            sequence.copy(),
            (
                "README full participant "
                "sequence"
            ),
        )

    for index, line in enumerate(
        lines
    ):
        if (
            "sub54 Imagery sequence"
            in line
        ):
            sequence = None

            for probe in range(
                index,
                min(
                    index + 5,
                    len(
                        lines
                    ),
                ),
            ):
                sequence = literal_list_from_line(
                    lines[
                        probe
                    ]
                )

                if sequence:
                    break

            if sequence:
                existing = sequences.get(
                    "sub-54",
                    SubjectSequence(
                        [],
                        [],
                        "",
                    ),
                )
                existing.imagery = sequence
                existing.source = (
                    "README sub54 "
                    "context-specific sequence"
                )
                sequences[
                    "sub-54"
                ] = existing

        if (
            "sub54 Video sequence"
            in line
        ):
            sequence = None

            for probe in range(
                index,
                min(
                    index + 5,
                    len(
                        lines
                    ),
                ),
            ):
                sequence = literal_list_from_line(
                    lines[
                        probe
                    ]
                )

                if sequence:
                    break

            if sequence:
                existing = sequences.get(
                    "sub-54",
                    SubjectSequence(
                        [],
                        [],
                        "",
                    ),
                )
                existing.video = sequence
                existing.source = (
                    "README sub54 "
                    "context-specific sequence"
                )
                sequences[
                    "sub-54"
                ] = existing

    for index, line in enumerate(
        lines
    ):
        if re.match(
            r"^\s*-\s+\*\*sub55\*\*:",
            line,
        ):
            sequence = None

            for probe in range(
                index,
                min(
                    index + 6,
                    len(
                        lines
                    ),
                ),
            ):
                sequence = literal_list_from_line(
                    lines[
                        probe
                    ]
                )

                if sequence:
                    break

            if sequence:
                sequences[
                    "sub-55"
                ] = SubjectSequence(
                    sequence,
                    sequence.copy(),
                    (
                        "README sub55 "
                        "available-trial sequence"
                    ),
                )

    for subject_number in range(
        56,
        61,
    ):
        subject = (
            f"sub-{subject_number:02d}"
        )
        block_starts = [
            index
            for index, line
            in enumerate(
                lines
            )
            if re.match(
                (
                    r"^\s*-\s+\*\*sub"
                    f"{subject_number}"
                    r"\*\*:"
                ),
                line,
            )
        ]
        imagery = None
        video = None

        for block_start in (
            block_starts
        ):
            block_end = min(
                block_start + 40,
                len(
                    lines
                ),
            )

            for probe in range(
                block_start + 1,
                block_end,
            ):
                if re.match(
                    r"^\s*-\s+\*\*sub\d+\*\*:",
                    lines[
                        probe
                    ],
                ):
                    block_end = probe
                    break

            local_imagery = None
            local_video = None

            for probe in range(
                block_start,
                block_end,
            ):
                line = lines[
                    probe
                ]

                if (
                    "Imagery sequence"
                    in line
                ):
                    for candidate in range(
                        probe,
                        min(
                            probe + 6,
                            block_end,
                        ),
                    ):
                        sequence = literal_list_from_line(
                            lines[
                                candidate
                            ]
                        )

                        if sequence:
                            local_imagery = sequence
                            break

                if (
                    "Video sequence"
                    in line
                ):
                    for candidate in range(
                        probe,
                        min(
                            probe + 6,
                            block_end,
                        ),
                    ):
                        sequence = literal_list_from_line(
                            lines[
                                candidate
                            ]
                        )

                        if sequence:
                            local_video = sequence
                            break

            if (
                local_imagery
                or local_video
            ):
                imagery = local_imagery
                video = local_video
                break

        if (
            imagery
            or video
        ):
            sequences[
                subject
            ] = SubjectSequence(
                imagery
                or [],
                video
                or [],
                (
                    f"README {subject} "
                    "available-trial "
                    "context-specific sequence"
                ),
            )

    for subject, sequence in (
        sequences.items()
    ):
        for context_values in [
            sequence.imagery,
            sequence.video,
        ]:
            unknown = set(
                context_values
            ) - CANONICAL_STIMULI

            if unknown:
                raise RuntimeError(
                    f"{subject}: README sequence "
                    f"contains unknown stimuli: "
                    f"{sorted(unknown)}"
                )

    return sequences


def build_readme_manifest(
    sequences: dict[
        str,
        SubjectSequence,
    ],
) -> pd.DataFrame:
    rows = []

    for subject in sorted(
        sequences
    ):
        sequence = sequences[
            subject
        ]

        for context in CONTEXTS:
            values = getattr(
                sequence,
                context,
            )

            for index, stimulus_id in enumerate(
                values,
                start=1,
            ):
                rows.append(
                    {
                        "subject_id": subject,
                        "context": context,
                        "available_sequence_index": (
                            index
                        ),
                        "stimulus_id": (
                            stimulus_id
                        ),
                        "emotion_category": (
                            stimulus_category(
                                stimulus_id
                            )
                        ),
                        "sequence_source": (
                            sequence.source
                        ),
                        "available_sequence_length": (
                            len(
                                values
                            )
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def load_inputs(
    dataset: Path,
    project: Path,
) -> dict[str, Any]:
    paths = {
        "behavior": (
            project / BEHAVIOR_REL
        ),
        "metadata": (
            project / METADATA_REL
        ),
        "events": (
            project / GLOBAL_EVENTS_REL
        ),
        "metadata_index": (
            project / METADATA_INDEX_REL
        ),
        "metadata_audit_summary": (
            project
            / METADATA_AUDIT_SUMMARY_REL
        ),
        "physio_clock": (
            project / PHYSIO_CLOCK_REL
        ),
        "readme": (
            dataset / "README.md"
        ),
        "builder": (
            dataset / BUILDER_REL
        ),
    }
    require_files(
        paths.values()
    )
    summary = json.loads(
        paths[
            "metadata_audit_summary"
        ].read_text(
            encoding="utf-8"
        )
    )

    if summary.get(
        "status"
    ) != (
        "context_order_defect_confirmed_"
        "correction_candidate_not_frozen"
    ):
        raise RuntimeError(
            "Unexpected event-grammar prerequisite status."
        )

    behavior = pd.read_csv(
        paths[
            "behavior"
        ],
        low_memory=False,
    )
    metadata = pd.read_csv(
        paths[
            "metadata"
        ],
        low_memory=False,
    )
    events = pd.read_csv(
        paths[
            "events"
        ],
        low_memory=False,
    )
    metadata_index = pd.read_csv(
        paths[
            "metadata_index"
        ],
        low_memory=False,
    )
    physio_clock = pd.read_csv(
        paths[
            "physio_clock"
        ],
        low_memory=False,
    )

    for frame in [
        behavior,
        metadata,
        events,
        metadata_index,
        physio_clock,
    ]:
        if (
            "subject_id"
            in frame.columns
        ):
            frame[
                "subject_id"
            ] = frame[
                "subject_id"
            ].map(
                normalize_subject
            )

    metadata[
        "context"
    ] = metadata[
        "context"
    ].map(
        normalize_context
    )
    metadata_index[
        "context"
    ] = metadata_index[
        "context"
    ].map(
        normalize_context
    )

    return {
        "paths": paths,
        "behavior": behavior,
        "metadata": metadata,
        "events": events,
        "metadata_index": (
            metadata_index
        ),
        "physio_clock": (
            physio_clock
        ),
        "prior_summary": summary,
    }


def verify_builder_defect(
    builder_path: Path,
) -> dict[str, Any]:
    text = builder_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    patterns = {
        "assign_context_definition": (
            r"def\s+assign_context"
            r"\s*\(\s*trial_number"
        ),
        "first_21_imagery_rule": (
            r"if\s+trial_number\s*<=\s*21"
        ),
        "imagery_return": (
            r"return\s+[\"']imagery[\"']"
        ),
        "video_return": (
            r"return\s+[\"']video[\"']"
        ),
        "behavior_row_application": (
            r"context\s*,\s*"
            r"context_trial_index\s*="
            r"\s*assign_context"
            r"\s*\(\s*trial_number"
        ),
    }
    matches = {
        name: bool(
            re.search(
                pattern,
                text,
                flags=re.MULTILINE,
            )
        )
        for name, pattern
        in patterns.items()
    }

    if not all(
        matches.values()
    ):
        raise RuntimeError(
            "The expected canonical builder context-order defect "
            "was not found exactly as audited."
        )

    lines = text.splitlines()
    relevant = []

    for index, line in enumerate(
        lines,
        start=1,
    ):
        if (
            "def assign_context"
            in line
            or "trial_number <= 21"
            in line
            or (
                "assign_context("
                in line
                and "def "
                not in line
            )
        ):
            start = max(
                1,
                index - 2,
            )
            end = min(
                len(
                    lines
                ),
                index + 4,
            )
            relevant.append(
                {
                    "matched_line": index,
                    "excerpt": "\n".join(
                        f"{line_number}: "
                        f"{lines[line_number - 1]}"
                        for line_number
                        in range(
                            start,
                            end + 1,
                        )
                    ),
                }
            )

    return {
        "builder_path": str(
            builder_path
        ),
        "builder_sha256": (
            sha256_file(
                builder_path
            )
        ),
        "matches": matches,
        "source_defect_confirmed": True,
        "excerpts": relevant,
        "interpretation": (
            "Behavioral acquisition-order rows were "
            "assigned to context by row number rather "
            "than event-derived context."
        ),
    }


def target_context_counts(
    behavior_rows: int,
    subject: str,
    sequences: dict[
        str,
        SubjectSequence,
    ],
) -> tuple[
    int,
    int,
]:
    if behavior_rows == 42:
        return (
            21,
            21,
        )

    sequence = sequences.get(
        subject
    )

    if sequence is not None:
        imagery = len(
            sequence.imagery
        )
        video = len(
            sequence.video
        )

        if (
            imagery
            + video
            == behavior_rows
        ):
            return (
                imagery,
                video,
            )

    raise RuntimeError(
        f"{subject}: cannot determine target "
        f"context counts for {behavior_rows} "
        "behavior rows."
    )


def deduplicate_events(
    events: pd.DataFrame,
    event_classes: Iterable[str],
) -> pd.DataFrame:
    frame = events.loc[
        events[
            "event_class"
        ].isin(
            list(
                event_classes
            )
        )
        & events[
            "classification_method"
        ].isin(
            VALID_EVENT_METHODS
        )
        & events[
            "global_onset_sec"
        ].notna()
    ].copy()

    if frame.empty:
        return frame

    method_rank = {
        "readme_exact_type_id": 0,
        "readme_modulo_10": 1,
    }
    frame[
        "method_rank"
    ] = frame[
        "classification_method"
    ].map(
        method_rank
    ).fillna(
        9
    )
    frame = frame.sort_values(
        [
            "global_onset_sec",
            "event_class",
            "method_rank",
            "annotation_index",
        ]
    )
    frame = frame.drop_duplicates(
        [
            "global_onset_sec",
            "event_class",
        ],
        keep="first",
    )
    return frame.sort_values(
        "global_onset_sec"
    ).reset_index(
        drop=True
    )


def metadata_windows(
    metadata: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    stimulus_events = deduplicate_events(
        events,
        CONTEXTS,
    )
    rows = []

    for subject, group in metadata.groupby(
        "subject_id",
        sort=True,
    ):
        ordered = group.sort_values(
            [
                "rating_time_sec",
                "trial_number",
            ]
        ).reset_index(
            drop=True
        )
        subject_stimulus = (
            stimulus_events.loc[
                stimulus_events[
                    "subject_id"
                ].eq(
                    subject
                )
            ]
        )
        previous_rating = -np.inf

        for _, row in ordered.iterrows():
            rating_time = float(
                row[
                    "rating_time_sec"
                ]
            )
            candidates = (
                subject_stimulus.loc[
                    subject_stimulus[
                        "global_onset_sec"
                    ].gt(
                        previous_rating
                    )
                    & subject_stimulus[
                        "global_onset_sec"
                    ].lt(
                        rating_time
                    )
                ]
            )
            candidate_records = [
                {
                    "context": str(
                        candidate[
                            "event_class"
                        ]
                    ),
                    "onset": float(
                        candidate[
                            "global_onset_sec"
                        ]
                    ),
                    "delay": float(
                        rating_time
                        - candidate[
                            "global_onset_sec"
                        ]
                    ),
                    "method": str(
                        candidate[
                            "classification_method"
                        ]
                    ),
                }
                for _, candidate
                in candidates.iterrows()
            ]
            rows.append(
                {
                    "subject_id": subject,
                    "trial_number": int(
                        row[
                            "trial_number"
                        ]
                    ),
                    "metadata_context": (
                        normalize_context(
                            row[
                                "context"
                            ]
                        )
                    ),
                    "rating_time_sec": (
                        rating_time
                    ),
                    "candidate_count": int(
                        len(
                            candidates
                        )
                    ),
                    "candidate_records": (
                        candidate_records
                    ),
                }
            )
            previous_rating = rating_time

    return pd.DataFrame(
        rows
    )


def learn_duration_priors(
    behavior: pd.DataFrame,
    metadata: pd.DataFrame,
    events: pd.DataFrame,
    metadata_index: pd.DataFrame,
    sequences: dict[
        str,
        SubjectSequence,
    ],
) -> tuple[
    dict[
        str,
        dict[str, float],
    ],
    pd.DataFrame,
]:
    mismatch_subjects = set(
        metadata_index.loc[
            metadata_index[
                "stored_equals_recomputed"
            ].eq(
                False
            ),
            "subject_id",
        ]
    )
    windows = metadata_windows(
        metadata,
        events,
    )
    behavior_counts = (
        behavior.groupby(
            "subject_id"
        ).size().to_dict()
    )
    event_counts = (
        deduplicate_events(
            events,
            CONTEXTS,
        )
        .groupby(
            "subject_id"
        )
        .size()
        .to_dict()
    )
    rows = []

    for _, row in windows.iterrows():
        subject = row[
            "subject_id"
        ]
        behavior_rows = int(
            behavior_counts.get(
                subject,
                0,
            )
        )

        if (
            subject
            in mismatch_subjects
            or behavior_rows
            != 42
            or event_counts.get(
                subject,
                0,
            )
            != 42
            or row[
                "candidate_count"
            ]
            != 1
        ):
            continue

        candidate = row[
            "candidate_records"
        ][
            0
        ]
        delay = float(
            candidate[
                "delay"
            ]
        )
        context = str(
            candidate[
                "context"
            ]
        )

        if (
            context
            != row[
                "metadata_context"
            ]
            or delay
            < MINIMUM_STIMULUS_TO_RATING_SEC
            or delay
            > 650
        ):
            continue

        rows.append(
            {
                "subject_id": subject,
                "context": context,
                "delay_sec": delay,
                "log_delay": math.log(
                    delay
                ),
            }
        )

    reference = pd.DataFrame(
        rows
    )
    priors = {}

    for context in CONTEXTS:
        values = reference.loc[
            reference[
                "context"
            ].eq(
                context
            ),
            "log_delay",
        ].to_numpy(
            dtype=float
        )

        if len(
            values
        ) < 100:
            raise RuntimeError(
                f"Too few clean {context} timing "
                f"observations: {len(values)}"
            )

        median = float(
            np.median(
                values
            )
        )
        mad = float(
            np.median(
                np.abs(
                    values
                    - median
                )
            )
        )
        robust_scale = (
            mad
            * 1.4826
        )

        if robust_scale <= 0:
            raise RuntimeError(
                f"{context}: zero robust timing scale."
            )

        priors[
            context
        ] = {
            "n": int(
                len(
                    values
                )
            ),
            "median_log_delay": (
                median
            ),
            "robust_scale_log_delay": (
                robust_scale
            ),
            "median_delay_sec": float(
                math.exp(
                    median
                )
            ),
        }

    return (
        priors,
        reference,
    )


def generate_pair_candidates(
    subject: str,
    events: pd.DataFrame,
    priors: dict[
        str,
        dict[str, float],
    ],
    variant_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    parameters = VARIANTS[
        variant_name
    ]
    subject_events = events.loc[
        events[
            "subject_id"
        ].eq(
            subject
        )
    ]
    stimulus = deduplicate_events(
        subject_events,
        CONTEXTS,
    )
    rating = deduplicate_events(
        subject_events,
        [
            "rating"
        ],
    )
    columns = [
        "subject_id",
        "variant",
        "context",
        "stimulus_onset_global_sec",
        "rating_onset_global_sec",
        "stimulus_to_rating_sec",
        "pair_cost",
        "stimulus_method",
        "rating_method",
        "intervening_stimulus_events",
        "intervening_rating_events",
        "stimulus_annotation_index",
        "rating_annotation_index",
    ]

    if (
        stimulus.empty
        or rating.empty
    ):
        return (
            pd.DataFrame(
                columns=columns
            ),
            stimulus,
            rating,
        )

    rating_times = rating[
        "global_onset_sec"
    ].to_numpy(
        dtype=float
    )
    rows = []

    for _, stimulus_row in (
        stimulus.iterrows()
    ):
        stimulus_time = float(
            stimulus_row[
                "global_onset_sec"
            ]
        )
        first = int(
            np.searchsorted(
                rating_times,
                (
                    stimulus_time
                    + MINIMUM_STIMULUS_TO_RATING_SEC
                ),
                side="left",
            )
        )
        last = min(
            len(
                rating
            ),
            first
            + int(
                parameters[
                    "ratings_per_stimulus"
                ]
            ),
        )

        for rating_index in range(
            first,
            last,
        ):
            rating_row = rating.iloc[
                rating_index
            ]
            rating_time = float(
                rating_row[
                    "global_onset_sec"
                ]
            )
            delay = (
                rating_time
                - stimulus_time
            )

            if delay > float(
                parameters[
                    "maximum_delay_sec"
                ]
            ):
                break

            context = str(
                stimulus_row[
                    "event_class"
                ]
            )
            prior = priors[
                context
            ]
            timing_z = abs(
                math.log(
                    delay
                )
                - prior[
                    "median_log_delay"
                ]
            ) / prior[
                "robust_scale_log_delay"
            ]
            intervening_stimulus = int(
                (
                    stimulus[
                        "global_onset_sec"
                    ].gt(
                        stimulus_time
                    )
                    & stimulus[
                        "global_onset_sec"
                    ].lt(
                        rating_time
                    )
                ).sum()
            )
            intervening_rating = int(
                (
                    rating[
                        "global_onset_sec"
                    ].gt(
                        stimulus_time
                    )
                    & rating[
                        "global_onset_sec"
                    ].lt(
                        rating_time
                    )
                ).sum()
            )
            cost = (
                min(
                    timing_z,
                    20.0,
                )
                + float(
                    parameters[
                        "intervening_stimulus_penalty"
                    ]
                )
                * intervening_stimulus
                + float(
                    parameters[
                        "intervening_rating_penalty"
                    ]
                )
                * intervening_rating
            )

            if (
                stimulus_row[
                    "classification_method"
                ]
                == "readme_modulo_10"
            ):
                cost += 0.2

            if (
                rating_row[
                    "classification_method"
                ]
                == "readme_modulo_10"
            ):
                cost += 0.2

            rows.append(
                {
                    "subject_id": subject,
                    "variant": variant_name,
                    "context": context,
                    "stimulus_onset_global_sec": (
                        stimulus_time
                    ),
                    "rating_onset_global_sec": (
                        rating_time
                    ),
                    "stimulus_to_rating_sec": (
                        delay
                    ),
                    "pair_cost": float(
                        cost
                    ),
                    "stimulus_method": (
                        stimulus_row[
                            "classification_method"
                        ]
                    ),
                    "rating_method": (
                        rating_row[
                            "classification_method"
                        ]
                    ),
                    "intervening_stimulus_events": (
                        intervening_stimulus
                    ),
                    "intervening_rating_events": (
                        intervening_rating
                    ),
                    "stimulus_annotation_index": int(
                        stimulus_row[
                            "annotation_index"
                        ]
                    ),
                    "rating_annotation_index": int(
                        rating_row[
                            "annotation_index"
                        ]
                    ),
                }
            )

    return (
        pd.DataFrame(
            rows,
            columns=columns,
        ),
        stimulus,
        rating,
    )


def select_nonoverlapping_pairs(
    candidates: pd.DataFrame,
    target_imagery: int,
    target_video: int,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    if candidates.empty:
        return (
            candidates.copy(),
            {
                "selected_imagery": 0,
                "selected_video": 0,
                "selected_total": 0,
                "target_imagery": int(
                    target_imagery
                ),
                "target_video": int(
                    target_video
                ),
                "target_total": int(
                    target_imagery
                    + target_video
                ),
                "total_cost": None,
                "candidate_pairs": 0,
            },
        )

    frame = candidates.sort_values(
        [
            "rating_onset_global_sec",
            "stimulus_onset_global_sec",
            "pair_cost",
        ]
    ).reset_index(
        drop=True
    )
    rating_times = frame[
        "rating_onset_global_sec"
    ].to_numpy(
        dtype=float
    )
    previous_indices = [
        (
            bisect.bisect_left(
                rating_times,
                float(
                    row[
                        "stimulus_onset_global_sec"
                    ]
                ),
            )
            - 1
        )
        for _, row
        in frame.iterrows()
    ]
    candidate_count = len(
        frame
    )
    infinity = 1e15
    dp = np.full(
        (
            candidate_count + 1,
            target_imagery + 1,
            target_video + 1,
        ),
        infinity,
        dtype=np.float64,
    )
    take = np.zeros(
        dp.shape,
        dtype=np.int8,
    )
    dp[
        0,
        0,
        0,
    ] = 0.0

    for candidate_index in range(
        1,
        candidate_count + 1,
    ):
        dp[
            candidate_index
        ] = dp[
            candidate_index - 1
        ]
        row = frame.iloc[
            candidate_index - 1
        ]
        previous_dp_index = (
            previous_indices[
                candidate_index - 1
            ]
            + 1
        )
        add_imagery = int(
            row[
                "context"
            ]
            == "imagery"
        )
        add_video = int(
            row[
                "context"
            ]
            == "video"
        )

        for imagery_count in range(
            add_imagery,
            target_imagery + 1,
        ):
            for video_count in range(
                add_video,
                target_video + 1,
            ):
                value = (
                    dp[
                        previous_dp_index,
                        (
                            imagery_count
                            - add_imagery
                        ),
                        (
                            video_count
                            - add_video
                        ),
                    ]
                    + float(
                        row[
                            "pair_cost"
                        ]
                    )
                )

                if value < dp[
                    candidate_index,
                    imagery_count,
                    video_count,
                ]:
                    dp[
                        candidate_index,
                        imagery_count,
                        video_count,
                    ] = value
                    take[
                        candidate_index,
                        imagery_count,
                        video_count,
                    ] = 1

    feasible = []

    for imagery_count in range(
        target_imagery + 1
    ):
        for video_count in range(
            target_video + 1
        ):
            value = dp[
                candidate_count,
                imagery_count,
                video_count,
            ]

            if value < infinity:
                feasible.append(
                    (
                        imagery_count
                        + video_count,
                        float(
                            value
                        ),
                        imagery_count,
                        video_count,
                    )
                )

    _, total_cost, imagery_count, video_count = max(
        feasible,
        key=lambda item: (
            item[
                0
            ],
            -item[
                1
            ],
        ),
    )
    selected_indices = []
    candidate_index = (
        candidate_count
    )
    current_imagery = (
        imagery_count
    )
    current_video = (
        video_count
    )

    while candidate_index > 0:
        if take[
            candidate_index,
            current_imagery,
            current_video,
        ] == 1:
            frame_index = (
                candidate_index
                - 1
            )
            row = frame.iloc[
                frame_index
            ]
            selected_indices.append(
                frame_index
            )

            if (
                row[
                    "context"
                ]
                == "imagery"
            ):
                current_imagery -= 1
            else:
                current_video -= 1

            candidate_index = (
                previous_indices[
                    frame_index
                ]
                + 1
            )
        else:
            candidate_index -= 1

    selected = frame.iloc[
        selected_indices[
            ::-1
        ]
    ].copy()
    selected[
        "selection_rank"
    ] = np.arange(
        1,
        len(
            selected
        )
        + 1,
    )
    summary = {
        "selected_imagery": int(
            imagery_count
        ),
        "selected_video": int(
            video_count
        ),
        "selected_total": int(
            imagery_count
            + video_count
        ),
        "target_imagery": int(
            target_imagery
        ),
        "target_video": int(
            target_video
        ),
        "target_total": int(
            target_imagery
            + target_video
        ),
        "total_cost": float(
            total_cost
        ),
        "candidate_pairs": int(
            candidate_count
        ),
    }
    return (
        selected,
        summary,
    )


def pair_signature(
    frame: pd.DataFrame,
) -> set[
    tuple[
        float,
        float,
        str,
    ]
]:
    return {
        (
            round(
                float(
                    row[
                        "stimulus_onset_global_sec"
                    ]
                ),
                4,
            ),
            round(
                float(
                    row[
                        "rating_onset_global_sec"
                    ]
                ),
                4,
            ),
            str(
                row[
                    "context"
                ]
            ),
        )
        for _, row
        in frame.iterrows()
    }


def reconstruct_all_subjects(
    behavior: pd.DataFrame,
    events: pd.DataFrame,
    priors: dict[
        str,
        dict[str, float],
    ],
    sequences: dict[
        str,
        SubjectSequence,
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    selected_rows = []
    candidate_rows = []
    subject_rows = []

    for subject, behavior_group in (
        behavior.groupby(
            "subject_id",
            sort=True,
        )
    ):
        behavior_rows = len(
            behavior_group
        )
        (
            target_imagery,
            target_video,
        ) = target_context_counts(
            behavior_rows,
            subject,
            sequences,
        )
        variant_selected = {}
        variant_summaries = {}
        stimulus_event_count = None
        rating_event_count = None

        for variant_name in VARIANTS:
            (
                candidates,
                stimulus,
                rating,
            ) = generate_pair_candidates(
                subject,
                events,
                priors,
                variant_name,
            )
            candidates[
                "target_imagery"
            ] = target_imagery
            candidates[
                "target_video"
            ] = target_video
            candidate_rows.append(
                candidates
            )
            selected, selection_summary = (
                select_nonoverlapping_pairs(
                    candidates,
                    target_imagery,
                    target_video,
                )
            )
            variant_selected[
                variant_name
            ] = selected
            variant_summaries[
                variant_name
            ] = selection_summary

            if (
                variant_name
                == "primary"
            ):
                stimulus_event_count = int(
                    len(
                        stimulus
                    )
                )
                rating_event_count = int(
                    len(
                        rating
                    )
                )

        signatures = {
            name: pair_signature(
                selected
            )
            for name, selected
            in variant_selected.items()
        }
        stable_all = set.intersection(
            *signatures.values()
        ) if signatures else set()
        union_all = set.union(
            *signatures.values()
        ) if signatures else set()
        primary = variant_selected[
            "primary"
        ].copy()
        primary[
            "selected_in_tight"
        ] = [
            signature in signatures[
                "tight"
            ]
            for signature in [
                (
                    round(
                        float(
                            row[
                                "stimulus_onset_global_sec"
                            ]
                        ),
                        4,
                    ),
                    round(
                        float(
                            row[
                                "rating_onset_global_sec"
                            ]
                        ),
                        4,
                    ),
                    str(
                        row[
                            "context"
                        ]
                    ),
                )
                for _, row
                in primary.iterrows()
            ]
        ]
        primary[
            "selected_in_wide"
        ] = [
            signature in signatures[
                "wide"
            ]
            for signature in [
                (
                    round(
                        float(
                            row[
                                "stimulus_onset_global_sec"
                            ]
                        ),
                        4,
                    ),
                    round(
                        float(
                            row[
                                "rating_onset_global_sec"
                            ]
                        ),
                        4,
                    ),
                    str(
                        row[
                            "context"
                        ]
                    ),
                )
                for _, row
                in primary.iterrows()
            ]
        ]
        primary[
            "selection_stability_count"
        ] = (
            1
            + primary[
                "selected_in_tight"
            ].astype(
                int
            )
            + primary[
                "selected_in_wide"
            ].astype(
                int
            )
        )
        primary[
            "target_imagery"
        ] = target_imagery
        primary[
            "target_video"
        ] = target_video
        selected_rows.append(
            primary
        )
        primary_summary = (
            variant_summaries[
                "primary"
            ]
        )
        subject_rows.append(
            {
                "subject_id": subject,
                "behavior_rows": int(
                    behavior_rows
                ),
                "target_imagery": int(
                    target_imagery
                ),
                "target_video": int(
                    target_video
                ),
                "deduplicated_stimulus_events": (
                    stimulus_event_count
                ),
                "deduplicated_rating_events": (
                    rating_event_count
                ),
                "selected_trial_pairs": int(
                    primary_summary[
                        "selected_total"
                    ]
                ),
                "selected_imagery": int(
                    primary_summary[
                        "selected_imagery"
                    ]
                ),
                "selected_video": int(
                    primary_summary[
                        "selected_video"
                    ]
                ),
                "target_reached": bool(
                    primary_summary[
                        "selected_total"
                    ]
                    == (
                        target_imagery
                        + target_video
                    )
                ),
                "stable_all_variants_pairs": int(
                    len(
                        stable_all
                    )
                ),
                "union_all_variants_pairs": int(
                    len(
                        union_all
                    )
                ),
                "primary_pair_stability_fraction": (
                    float(
                        len(
                            stable_all
                        )
                        / len(
                            signatures[
                                "primary"
                            ]
                        )
                    )
                    if signatures[
                        "primary"
                    ]
                    else None
                ),
                "primary_total_cost": (
                    primary_summary[
                        "total_cost"
                    ]
                ),
            }
        )

    all_selected = pd.concat(
        selected_rows,
        ignore_index=True,
        sort=False,
    ) if selected_rows else pd.DataFrame()
    all_candidates = pd.concat(
        candidate_rows,
        ignore_index=True,
        sort=False,
    ) if candidate_rows else pd.DataFrame()
    subject_qc = pd.DataFrame(
        subject_rows
    )
    return (
        all_selected,
        all_candidates,
        subject_qc,
    )


def metadata_rating_lookup(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    return metadata[
        [
            "subject_id",
            "trial_number",
            "rating_time_sec",
            "context",
        ]
    ].copy()


def map_selected_pairs_to_behavior(
    behavior: pd.DataFrame,
    metadata: pd.DataFrame,
    selected: pd.DataFrame,
    subject_qc: pd.DataFrame,
    metadata_index: pd.DataFrame,
    tolerance: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rows = []
    mapping_qc_rows = []
    stable_metadata_subjects = set(
        metadata_index.groupby(
            "subject_id"
        )[
            "stored_equals_recomputed"
        ].all().loc[
            lambda values: values
        ].index
    )

    for subject, behavior_group in (
        behavior.groupby(
            "subject_id",
            sort=True,
        )
    ):
        behavior_ordered = (
            behavior_group.sort_values(
                "trial_number"
            ).copy()
        )
        selected_subject = (
            selected.loc[
                selected[
                    "subject_id"
                ].eq(
                    subject
                )
            ].sort_values(
                "rating_onset_global_sec"
            ).reset_index(
                drop=True
            )
        )
        metadata_subject = (
            metadata.loc[
                metadata[
                    "subject_id"
                ].eq(
                    subject
                )
            ].sort_values(
                "trial_number"
            ).copy()
        )
        subject_qc_row = (
            subject_qc.loc[
                subject_qc[
                    "subject_id"
                ].eq(
                    subject
                )
            ].iloc[
                0
            ]
        )
        behavior_count = len(
            behavior_ordered
        )
        selected_count = len(
            selected_subject
        )
        assignments = []

        if selected_count == behavior_count:
            for index, pair in (
                selected_subject.iterrows()
            ):
                assignments.append(
                    {
                        "trial_number": int(
                            behavior_ordered.iloc[
                                index
                            ][
                                "trial_number"
                            ]
                        ),
                        **pair.to_dict(),
                        "behavior_event_mapping_method": (
                            "complete_event_grammar_sequence"
                        ),
                    }
                )
        elif selected_count > 0:
            metadata_times = (
                metadata_subject[
                    "rating_time_sec"
                ].to_numpy(
                    dtype=float
                )
            )
            metadata_trials = (
                metadata_subject[
                    "trial_number"
                ].to_numpy()
            )
            matched_trials = []
            unmatched_pairs = []

            for _, pair in (
                selected_subject.iterrows()
            ):
                rating_time = float(
                    pair[
                        "rating_onset_global_sec"
                    ]
                )

                if len(
                    metadata_times
                ) == 0:
                    unmatched_pairs.append(
                        pair
                    )
                    continue

                differences = np.abs(
                    metadata_times
                    - rating_time
                )
                candidates = np.where(
                    differences
                    <= tolerance
                )[
                    0
                ]

                if len(
                    candidates
                ) == 1:
                    metadata_index_value = int(
                        candidates[
                            0
                        ]
                    )
                    trial_number = int(
                        metadata_trials[
                            metadata_index_value
                        ]
                    )
                    matched_trials.append(
                        trial_number
                    )
                    assignments.append(
                        {
                            "trial_number": (
                                trial_number
                            ),
                            **pair.to_dict(),
                            "behavior_event_mapping_method": (
                                "exact_metadata_rating_anchor"
                            ),
                        }
                    )
                else:
                    unmatched_pairs.append(
                        pair
                    )

            matched_trials_sorted = sorted(
                set(
                    matched_trials
                )
            )
            prefix_length = 0

            while (
                prefix_length + 1
                in matched_trials_sorted
            ):
                prefix_length += 1

            unmatched_pairs = sorted(
                unmatched_pairs,
                key=lambda row: float(
                    row[
                        "rating_onset_global_sec"
                    ]
                ),
            )
            after_prefix = [
                pair
                for pair in unmatched_pairs
                if (
                    not assignments
                    or float(
                        pair[
                            "rating_onset_global_sec"
                        ]
                    )
                    > max(
                        float(
                            assignment[
                                "rating_onset_global_sec"
                            ]
                        )
                        for assignment
                        in assignments
                    )
                )
            ]

            if (
                len(
                    after_prefix
                )
                == len(
                    unmatched_pairs
                )
                and prefix_length
                + len(
                    unmatched_pairs
                )
                <= behavior_count
            ):
                for offset, pair in enumerate(
                    unmatched_pairs,
                    start=1,
                ):
                    assignments.append(
                        {
                            "trial_number": (
                                prefix_length
                                + offset
                            ),
                            **pair.to_dict(),
                            "behavior_event_mapping_method": (
                                "ordered_event_pair_after_"
                                "metadata_prefix"
                            ),
                        }
                    )
        elif (
            subject
            in stable_metadata_subjects
            and len(
                metadata_subject
            )
            == behavior_count
        ):
            for _, metadata_row in (
                metadata_subject.iterrows()
            ):
                assignments.append(
                    {
                        "trial_number": int(
                            metadata_row[
                                "trial_number"
                            ]
                        ),
                        "subject_id": subject,
                        "context": (
                            normalize_context(
                                metadata_row[
                                    "context"
                                ]
                            )
                        ),
                        "stimulus_onset_global_sec": (
                            np.nan
                        ),
                        "rating_onset_global_sec": (
                            float(
                                metadata_row[
                                    "rating_time_sec"
                                ]
                            )
                        ),
                        "stimulus_to_rating_sec": (
                            np.nan
                        ),
                        "pair_cost": np.nan,
                        "selected_in_tight": (
                            False
                        ),
                        "selected_in_wide": (
                            False
                        ),
                        "selection_stability_count": (
                            0
                        ),
                        "behavior_event_mapping_method": (
                            "stable_metadata_fallback_"
                            "raw_eeg_unavailable"
                        ),
                    }
                )

        assignment_frame = pd.DataFrame(
            assignments
        )

        # A participant can legitimately have no usable event or metadata
        # mapping source (notably sub-01). A DataFrame created from an empty
        # list has no columns, which previously caused the left merge below
        # to fail with KeyError: 'subject_id'. Preserve all behavioral rows
        # as explicitly unmapped instead of failing or guessing.
        required_assignment_dtypes = {
            "subject_id": "object",
            "trial_number": "int64",
            "context": "object",
            "stimulus_onset_global_sec": "float64",
            "rating_onset_global_sec": "float64",
            "stimulus_to_rating_sec": "float64",
            "pair_cost": "float64",
            "selected_in_tight": "boolean",
            "selected_in_wide": "boolean",
            "selection_stability_count": "float64",
            "behavior_event_mapping_method": "object",
        }

        for (
            required_column,
            required_dtype,
        ) in required_assignment_dtypes.items():
            if required_column not in assignment_frame.columns:
                assignment_frame[
                    required_column
                ] = pd.Series(
                    dtype=required_dtype
                )

        assignment_frame = assignment_frame[
            list(
                dict.fromkeys(
                    list(
                        required_assignment_dtypes
                    )
                    + list(
                        assignment_frame.columns
                    )
                )
            )
        ]

        if (
            not assignment_frame.empty
            and assignment_frame.duplicated(
                "trial_number"
            ).any()
        ):
            raise RuntimeError(
                f"{subject}: event mapping assigned "
                "a behavior trial more than once."
            )

        merged = behavior_ordered.merge(
            assignment_frame,
            on=[
                "subject_id",
                "trial_number",
            ],
            how="left",
            validate="one_to_one",
            suffixes=(
                "",
                "_event",
            ),
        )
        rows.append(
            merged
        )
        mapping_qc_rows.append(
            {
                "subject_id": subject,
                "behavior_rows": int(
                    behavior_count
                ),
                "selected_event_pairs": int(
                    selected_count
                ),
                "mapped_behavior_rows": int(
                    merged[
                        "behavior_event_mapping_method"
                    ].notna().sum()
                ),
                "unmapped_behavior_rows": int(
                    merged[
                        "behavior_event_mapping_method"
                    ].isna().sum()
                ),
                "target_reached": bool(
                    subject_qc_row[
                        "target_reached"
                    ]
                ),
                "event_mapping_complete": bool(
                    merged[
                        "behavior_event_mapping_method"
                    ].notna().all()
                ),
                "mapping_source_status": (
                    "no_event_or_metadata_mapping_available"
                    if assignment_frame.empty
                    else (
                        "complete_mapping"
                        if merged[
                            "behavior_event_mapping_method"
                        ].notna().all()
                        else "partial_mapping"
                    )
                ),
            }
        )

    mapped = pd.concat(
        rows,
        ignore_index=True,
        sort=False,
    )
    mapped = mapped.sort_values(
        [
            "subject_id",
            "trial_number",
        ]
    ).reset_index(
        drop=True
    )
    mapped[
        "context_candidate"
    ] = mapped[
        "context"
    ].map(
        lambda value: (
            normalize_context(
                value
            )
            if pd.notna(
                value
            )
            else np.nan
        )
    )
    valid_context = mapped[
        "context_candidate"
    ].isin(
        CONTEXTS
    )
    mapped[
        "recomputed_context_trial_index"
    ] = np.nan
    mapped.loc[
        valid_context,
        "recomputed_context_trial_index",
    ] = (
        mapped.loc[
            valid_context
        ]
        .groupby(
            [
                "subject_id",
                "context_candidate",
            ]
        )
        .cumcount()
        + 1
    )
    return (
        mapped,
        pd.DataFrame(
            mapping_qc_rows
        ),
    )


def validate_reference_subjects(
    behavior: pd.DataFrame,
    metadata: pd.DataFrame,
    events: pd.DataFrame,
    selected: pd.DataFrame,
    metadata_index: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    mismatch_subjects = set(
        metadata_index.loc[
            metadata_index[
                "stored_equals_recomputed"
            ].eq(
                False
            ),
            "subject_id",
        ]
    )
    stimulus_counts = (
        deduplicate_events(
            events,
            CONTEXTS,
        )
        .groupby(
            "subject_id"
        )
        .size()
        .to_dict()
    )
    rows = []

    for subject, behavior_group in (
        behavior.groupby(
            "subject_id",
            sort=True,
        )
    ):
        behavior_count = len(
            behavior_group
        )
        metadata_subject = (
            metadata.loc[
                metadata[
                    "subject_id"
                ].eq(
                    subject
                )
            ].sort_values(
                "trial_number"
            )
        )
        selected_subject = (
            selected.loc[
                selected[
                    "subject_id"
                ].eq(
                    subject
                )
            ].sort_values(
                "rating_onset_global_sec"
            )
        )
        is_reference = bool(
            behavior_count
            == 42
            and len(
                metadata_subject
            )
            == 42
            and stimulus_counts.get(
                subject,
                0,
            )
            == 42
            and subject
            not in mismatch_subjects
        )
        exact_matches = 0
        maximum_difference = None

        if (
            len(
                selected_subject
            )
            > 0
            and len(
                metadata_subject
            )
            > 0
        ):
            count = min(
                len(
                    selected_subject
                ),
                len(
                    metadata_subject
                ),
            )
            differences = np.abs(
                selected_subject[
                    "rating_onset_global_sec"
                ].iloc[
                    :count
                ].to_numpy(
                    dtype=float
                )
                - metadata_subject[
                    "rating_time_sec"
                ].iloc[
                    :count
                ].to_numpy(
                    dtype=float
                )
            )
            exact_matches = int(
                (
                    differences
                    <= tolerance
                ).sum()
            )
            maximum_difference = float(
                differences.max()
            )

        rows.append(
            {
                "subject_id": subject,
                "is_structural_reference_subject": (
                    is_reference
                ),
                "behavior_rows": int(
                    behavior_count
                ),
                "metadata_rows": int(
                    len(
                        metadata_subject
                    )
                ),
                "deduplicated_stimulus_events": int(
                    stimulus_counts.get(
                        subject,
                        0,
                    )
                ),
                "selected_event_pairs": int(
                    len(
                        selected_subject
                    )
                ),
                "sequential_rating_matches": int(
                    exact_matches
                ),
                "maximum_sequential_rating_difference_sec": (
                    maximum_difference
                ),
                "reference_validation_passed": bool(
                    (
                        exact_matches
                        == behavior_count
                    )
                    if is_reference
                    else False
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    )
    reference = frame.loc[
        frame[
            "is_structural_reference_subject"
        ]
    ]

    if (
        not reference.empty
        and not reference[
            "reference_validation_passed"
        ].all()
    ):
        failed = reference.loc[
            ~reference[
                "reference_validation_passed"
            ]
        ]
        raise RuntimeError(
            "Event grammar failed one or more "
            "structural reference subjects:\n"
            + failed.to_string(
                index=False
            )
        )

    return frame


def attach_readme_stimulus_candidates(
    mapped: pd.DataFrame,
    manifest: pd.DataFrame,
    behavior: pd.DataFrame,
) -> pd.DataFrame:
    output = mapped.copy()
    manifest_lookup = manifest.rename(
        columns={
            "context": (
                "manifest_context"
            ),
            "available_sequence_index": (
                "manifest_sequence_index"
            ),
            "stimulus_id": (
                "stimulus_id_candidate"
            ),
            "emotion_category": (
                "emotion_category_candidate"
            ),
            "sequence_source": (
                "stimulus_sequence_source"
            ),
            "available_sequence_length": (
                "available_sequence_length"
            ),
        }
    )
    output = output.merge(
        manifest_lookup[
            [
                "subject_id",
                "manifest_context",
                "manifest_sequence_index",
                "stimulus_id_candidate",
                "emotion_category_candidate",
                "stimulus_sequence_source",
                "available_sequence_length",
            ]
        ],
        left_on=[
            "subject_id",
            "context_candidate",
            "recomputed_context_trial_index",
        ],
        right_on=[
            "subject_id",
            "manifest_context",
            "manifest_sequence_index",
        ],
        how="left",
        validate="many_to_one",
    )
    subject_context_counts = (
        output.loc[
            output[
                "context_candidate"
            ].isin(
                CONTEXTS
            )
        ]
        .groupby(
            [
                "subject_id",
                "context_candidate",
            ]
        )
        .size()
        .rename(
            "mapped_context_count"
        )
        .reset_index()
    )
    output = output.merge(
        subject_context_counts,
        on=[
            "subject_id",
            "context_candidate",
        ],
        how="left",
        validate="many_to_one",
    )
    output[
        "stimulus_mapping_identifiability"
    ] = np.select(
        [
            output[
                "context_candidate"
            ].isna(),
            output[
                "subject_id"
            ].eq(
                "sub-58"
            )
            & output[
                "context_candidate"
            ].eq(
                "imagery"
            ),
            output[
                "stimulus_id_candidate"
            ].notna(),
        ],
        [
            "context_unresolved",
            (
                "requires_waveform_fingerprint_"
                "three_missing_imagery_trials"
            ),
            "sequence_position_candidate",
        ],
        default=(
            "no_available_sequence_item"
        ),
    )
    sub58_imagery = (
        output[
            "subject_id"
        ].eq(
            "sub-58"
        )
        & output[
            "context_candidate"
        ].eq(
            "imagery"
        )
    )
    output.loc[
        sub58_imagery,
        [
            "stimulus_id_candidate",
            "emotion_category_candidate",
        ],
    ] = np.nan
    output[
        "authoritative_for_analysis"
    ] = False
    output[
        "candidate_only_do_not_replace_contract"
    ] = True
    return output


def probe_combined_metadata(
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    if not path.is_file():
        return (
            pd.DataFrame(),
            {
                "path": str(
                    path
                ),
                "exists": False,
            },
        )

    header = pd.read_csv(
        path,
        nrows=0,
        low_memory=False,
    )
    columns = list(
        header.columns
    )
    safe_patterns = (
        "sub",
        "subject",
        "session",
        "context",
        "trial",
        "code",
        "emotion",
        "intensity",
        "start",
        "end",
        "time",
    )
    safe_columns = [
        column
        for column in columns
        if any(
            token
            in str(
                column
            ).lower()
            for token
            in safe_patterns
        )
        and not re.search(
            r"(score|valence|liking|joy|sadness|fear|"
            r"disgust|arousal|familiarity|tenderness|"
            r"inspiration)",
            str(
                column
            ),
            flags=re.IGNORECASE,
        )
    ]
    sample = (
        pd.read_csv(
            path,
            usecols=safe_columns,
            nrows=20,
            low_memory=False,
        )
        if safe_columns
        else pd.DataFrame()
    )
    summary = {
        "path": str(
            path
        ),
        "exists": True,
        "sha256": sha256_file(
            path
        ),
        "all_columns": columns,
        "safe_columns_read": safe_columns,
        "sample_rows": int(
            len(
                sample
            )
        ),
        "contains_acquisition_order_candidate": bool(
            any(
                re.search(
                    r"(global|acquisition|rating)",
                    column,
                    flags=re.IGNORECASE,
                )
                for column
                in safe_columns
            )
        ),
        "interpretation": (
            "This source is only useful for acquisition "
            "alignment if it contains an acquisition-order "
            "or rating-time field."
        ),
    }
    return (
        sample,
        summary,
    )


def physiology_contract(
    physio_clock: pd.DataFrame,
) -> dict[str, Any]:
    parsed = (
        physio_clock[
            "timestamp_audit_status"
        ].fillna("").str.startswith(
            "parsed"
        )
    )
    one_second = (
        physio_clock[
            "timestamp_resolution_is_one_second"
        ].eq(
            True
        )
    )
    jumps = (
        physio_clock[
            "wall_clock_jump_flag"
        ].eq(
            True
        )
    )
    return {
        "status": (
            "sample_clock_policy_frozen_"
            "absolute_crossmodal_offset_unresolved"
        ),
        "timestamp_streams": int(
            len(
                physio_clock
            )
        ),
        "parsed_timestamp_streams": int(
            parsed.sum()
        ),
        "one_second_resolution_streams": int(
            one_second.sum()
        ),
        "wall_clock_jump_streams": int(
            jumps.sum()
        ),
        "wall_clock_jump_participants": sorted(
            physio_clock.loc[
                jumps,
                "subject_id",
            ].unique().tolist()
        ),
        "sample_time_formula": (
            "sample_index_zero_based / "
            "sampling_frequency"
        ),
        "timestamp_string_role": (
            "coarse_start_and_discontinuity_audit_only"
        ),
        "absolute_eeg_physio_offset_status": (
            "unresolved"
        ),
        "amplitude_values_used": False,
    }


def write_report(
    path: Path,
    summary: dict[str, Any],
    subject_qc: pd.DataFrame,
    reference_qc: pd.DataFrame,
    mapping_qc: pd.DataFrame,
) -> None:
    reconstruction = summary[
        "event_reconstruction"
    ]
    lines = [
        "# Event-grammar reconstruction — Event-grammar reconstruction",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Status: **`{summary['status']}`**",
        "- Source dataset modified: **no**",
        "- Authoritative canonical modified: **no**",
        "- Validation outcomes read: **no**",
        "- Confirmation outcomes read: **no**",
        "- EEG/GSR/PPG amplitudes read: **no**",
        "",
        "## Source-code defect",
        "",
        "The original canonical builder builder assigned behavioral rows 1–21 to "
        "imagery and rows 22–42 to video. This is incompatible with the "
        "interleaved acquisition order and is the direct source of the "
        "current context/stimulus attachment defect.",
        "",
        "## Event-grammar reconstruction",
        "",
        (
            "- Discovery participants: "
            f"**{reconstruction['participants']}**"
        ),
        (
            "- Behavior rows: "
            f"**{reconstruction['behavior_rows']}**"
        ),
        (
            "- Selected stimulus→rating pairs: "
            f"**{reconstruction['selected_pairs']}**"
        ),
        (
            "- Behavior rows mapped to an event/context source: "
            f"**{reconstruction['mapped_behavior_rows']}**"
        ),
        (
            "- Event selections stable in all three specifications: "
            f"**{reconstruction['stable_all_variant_pairs']}**"
        ),
        "",
        "The optimizer selects chronologically non-overlapping "
        "stimulus→rating pairs, enforces documented imagery/video capacity, "
        "and penalizes implausible timing and intervening trigger chatter.",
        "",
        "## Structural reference validation",
        "",
        (
            "- Reference participants: "
            f"**{summary['reference_validation']['reference_participants']}**"
        ),
        (
            "- Reference participants passing exact sequential rating "
            "alignment: "
            f"**{summary['reference_validation']['passed_participants']}**"
        ),
        "",
        "Reference participants have stable metadata indices, exactly the "
        "expected number of stimulus triggers, complete behavior, and no "
        "known trigger pathology.",
        "",
        "## Remaining unresolved cases",
        "",
        (
            "- Completely or partially unmapped behavior rows: "
            f"**{reconstruction['unmapped_behavior_rows']}**"
        ),
        (
            "- Primary event pairs not stable in every timing "
            "specification: "
            f"**{reconstruction['unstable_primary_pairs']}**"
        ),
        "",
        "Unstable pairs and derivative-missing sequence positions must be "
        "resolved with waveform fingerprints before a corrected multimodal "
        "master can be frozen.",
        "",
        "## Participant reconstruction QC",
        "",
        "| Subject | Behavior | Selected pairs | Target reached | "
        "Stable fraction | Mapped rows |",
        "|---|---:|---:|---|---:|---:|",
    ]
    mapping_lookup = mapping_qc.set_index(
        "subject_id"
    )

    for _, row in subject_qc.iterrows():
        mapped_rows = (
            int(
                mapping_lookup.loc[
                    row[
                        "subject_id"
                    ],
                    "mapped_behavior_rows",
                ]
            )
            if row[
                "subject_id"
            ] in mapping_lookup.index
            else 0
        )
        fraction = row[
            "primary_pair_stability_fraction"
        ]
        lines.append(
            f"| `{row['subject_id']}` | "
            f"{int(row['behavior_rows'])} | "
            f"{int(row['selected_trial_pairs'])} | "
            f"{bool(row['target_reached'])} | "
            f"{fraction if pd.notna(fraction) else ''} | "
            f"{mapped_rows} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "**The reconstructed table remains a candidate and must not yet "
            "replace the frozen canonical contract.**",
            "",
            "The next phase is a targeted waveform-fingerprint audit for "
            "unstable event pairs and participants with derivative gaps. "
            "Rows lacking any acquisition evidence remain explicitly "
            "unresolved rather than being guessed.",
            "",
        ]
    )
    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def make_zip(
    project: Path,
    files: Iterable[Path],
) -> Path:
    output = (
        project
        / "release_bundle"
        / (
            "event_grammar_reconstruction_event_grammar_reconstruction_"
            f"{timestamp_tag()}.zip"
        )
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    root = output.stem
    seen = set()

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for file_path in files:
            if not file_path.is_file():
                continue

            resolved = str(
                file_path.resolve()
            )

            if resolved in seen:
                continue

            seen.add(
                resolved
            )

            try:
                relative = file_path.relative_to(
                    project
                )
            except ValueError:
                relative = (
                    Path(
                        "external"
                    )
                    / file_path.name
                )

            archive.write(
                file_path,
                arcname=(
                    f"{root}/{relative}"
                ),
            )

    return output


def main() -> int:
    args = parse_args()
    started = time.time()
    dataset = args.dataset.expanduser().resolve()
    project = args.project.expanduser().resolve()
    inputs = load_inputs(
        dataset,
        project,
    )
    sequences = parse_readme_sequences(
        inputs[
            "paths"
        ][
            "readme"
        ]
    )
    discovery_subjects = set(
        inputs[
            "behavior"
        ][
            "subject_id"
        ].unique()
    )
    missing_sequence_subjects = (
        discovery_subjects
        - set(
            sequences
        )
    )

    if missing_sequence_subjects:
        raise RuntimeError(
            "README sequence parser missed discovery subjects: "
            f"{sorted(missing_sequence_subjects)}"
        )

    tables = (
        project
        / "outputs/tables"
    )
    qc = (
        project
        / "outputs/qc"
    )
    reports = (
        project
        / "outputs/reports"
    )
    logs = (
        project
        / "outputs/logs"
    )

    for directory in [
        tables,
        qc,
        reports,
        logs,
        project / "release_bundle",
        project / "config",
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    outputs = {
        "readme_manifest": (
            tables
            / "event_grammar_reconstruction_direct_readme_sequence_manifest.csv"
        ),
        "timing_reference": (
            tables
            / "event_grammar_reconstruction_timing_reference_observations.csv"
        ),
        "pair_candidates": (
            tables
            / "event_grammar_reconstruction_event_pair_candidates.csv"
        ),
        "selected_pairs": (
            tables
            / "event_grammar_reconstruction_selected_event_trial_pairs.csv"
        ),
        "subject_qc": (
            qc
            / "event_grammar_reconstruction_event_reconstruction_subject_qc.csv"
        ),
        "reference_qc": (
            qc
            / "event_grammar_reconstruction_structural_reference_validation.csv"
        ),
        "behavior_event_candidate": (
            tables
            / "event_grammar_reconstruction_behavior_event_alignment_candidate.csv"
        ),
        "mapping_qc": (
            qc
            / "event_grammar_reconstruction_behavior_event_mapping_subject_qc.csv"
        ),
        "comprehensive_candidate": (
            tables
            / "event_grammar_reconstruction_non_authoritative_discovery_lineage.csv"
        ),
        "combined_metadata_sample": (
            tables
            / "event_grammar_reconstruction_combined_metadata_safe_probe.csv"
        ),
        "builder_contract": (
            project
            / "config/"
            "event_grammar_reconstruction_builder_audit.json"
        ),
        "physio_contract": (
            project
            / "config/"
            "event_grammar_reconstruction_physio_sample_clock_contract.json"
        ),
        "summary": (
            qc
            / "event_grammar_reconstruction_event_grammar_summary.json"
        ),
        "report": (
            reports
            / "event_grammar_reconstruction_event_grammar_report.md"
        ),
        "log": (
            logs
            / "event_grammar_reconstruction_event_grammar.log"
        ),
    }

    existing = [
        path
        for path in outputs.values()
        if path.exists()
    ]

    if (
        existing
        and not args.force
    ):
        raise RuntimeError(
            "Outputs already exist. Use --force only after review:\n  - "
            + "\n  - ".join(
                str(
                    path
                )
                for path
                in existing
            )
        )

    with outputs[
        "log"
    ].open(
        "w",
        encoding="utf-8",
    ) as log:
        def note(message: str) -> None:
            line = (
                f"[{now_iso()}] "
                f"{message}"
            )
            print(
                line,
                flush=True,
            )
            log.write(
                line
                + "\n"
            )
            log.flush()

        note(
            "Confirming the canonical builder source-code context-order defect."
        )
        builder_contract = (
            verify_builder_defect(
                inputs[
                    "paths"
                ][
                    "builder"
                ]
            )
        )
        outputs[
            "builder_contract"
        ].write_text(
            json.dumps(
                builder_contract,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        note(
            "Parsing subject/context stimulus sequences directly from README."
        )
        readme_manifest = (
            build_readme_manifest(
                sequences
            )
        )
        readme_manifest.to_csv(
            outputs[
                "readme_manifest"
            ],
            index=False,
        )

        note(
            "Learning robust stimulus-to-rating timing priors from clean structural references."
        )
        (
            priors,
            timing_reference,
        ) = learn_duration_priors(
            inputs[
                "behavior"
            ],
            inputs[
                "metadata"
            ],
            inputs[
                "events"
            ],
            inputs[
                "metadata_index"
            ],
            sequences,
        )
        timing_reference.to_csv(
            outputs[
                "timing_reference"
            ],
            index=False,
        )

        note(
            "Reconstructing non-overlapping trial pairs under three timing specifications."
        )
        (
            selected_pairs,
            pair_candidates,
            subject_qc,
        ) = reconstruct_all_subjects(
            inputs[
                "behavior"
            ],
            inputs[
                "events"
            ],
            priors,
            sequences,
        )
        pair_candidates.to_csv(
            outputs[
                "pair_candidates"
            ],
            index=False,
        )
        selected_pairs.to_csv(
            outputs[
                "selected_pairs"
            ],
            index=False,
        )
        subject_qc.to_csv(
            outputs[
                "subject_qc"
            ],
            index=False,
        )

        note(
            "Validating event reconstruction against unaffected structural reference participants."
        )
        reference_qc = (
            validate_reference_subjects(
                inputs[
                    "behavior"
                ],
                inputs[
                    "metadata"
                ],
                inputs[
                    "events"
                ],
                selected_pairs,
                inputs[
                    "metadata_index"
                ],
                args.metadata_match_tolerance_sec,
            )
        )
        reference_qc.to_csv(
            outputs[
                "reference_qc"
            ],
            index=False,
        )

        note(
            "Mapping reconstructed event trials to raw behavioral acquisition order."
        )
        (
            behavior_event_candidate,
            mapping_qc,
        ) = map_selected_pairs_to_behavior(
            inputs[
                "behavior"
            ],
            inputs[
                "metadata"
            ],
            selected_pairs,
            subject_qc,
            inputs[
                "metadata_index"
            ],
            args.metadata_match_tolerance_sec,
        )
        behavior_event_candidate.to_csv(
            outputs[
                "behavior_event_candidate"
            ],
            index=False,
        )
        mapping_qc.to_csv(
            outputs[
                "mapping_qc"
            ],
            index=False,
        )

        note(
            "Attaching README stimulus candidates only where the sequence position is identifiable."
        )
        comprehensive_candidate = (
            attach_readme_stimulus_candidates(
                behavior_event_candidate,
                readme_manifest,
                inputs[
                    "behavior"
                ],
            )
        )
        comprehensive_candidate.to_csv(
            outputs[
                "comprehensive_candidate"
            ],
            index=False,
        )

        note(
            "Probing the combined sub01-05 metadata schema without reading report outcomes."
        )
        (
            combined_metadata_sample,
            combined_metadata_summary,
        ) = probe_combined_metadata(
            dataset
            / COMBINED_METADATA_REL
        )
        combined_metadata_sample.to_csv(
            outputs[
                "combined_metadata_sample"
            ],
            index=False,
        )

        note(
            "Writing the physiology sample-clock candidate contract."
        )
        physio_contract_data = (
            physiology_contract(
                inputs[
                    "physio_clock"
                ]
            )
        )
        outputs[
            "physio_contract"
        ].write_text(
            json.dumps(
                physio_contract_data,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        reference = reference_qc.loc[
            reference_qc[
                "is_structural_reference_subject"
            ]
        ]
        stable_pairs = int(
            selected_pairs[
                "selection_stability_count"
            ].eq(
                3
            ).sum()
        )
        total_selected_pairs = int(
            len(
                selected_pairs
            )
        )
        mapped_rows = int(
            behavior_event_candidate[
                "behavior_event_mapping_method"
            ].notna().sum()
        )
        total_behavior_rows = int(
            len(
                behavior_event_candidate
            )
        )
        event_reconstruction = {
            "participants": int(
                inputs[
                    "behavior"
                ][
                    "subject_id"
                ].nunique()
            ),
            "behavior_rows": (
                total_behavior_rows
            ),
            "selected_pairs": (
                total_selected_pairs
            ),
            "mapped_behavior_rows": (
                mapped_rows
            ),
            "unmapped_behavior_rows": int(
                total_behavior_rows
                - mapped_rows
            ),
            "stable_all_variant_pairs": (
                stable_pairs
            ),
            "unstable_primary_pairs": int(
                total_selected_pairs
                - stable_pairs
            ),
            "subjects_reaching_target_trial_count": int(
                subject_qc[
                    "target_reached"
                ].sum()
            ),
            "subjects_not_reaching_target_trial_count": int(
                (
                    ~subject_qc[
                        "target_reached"
                    ]
                ).sum()
            ),
        }
        summary = {
            "generated_at": now_iso(),
            "phase": "event_grammar_reconstruction",
            "status": (
                "event_grammar_reconstruction_"
                "candidate_not_frozen"
            ),
            "source_dataset_modified": False,
            "authoritative_canonical_modified": False,
            "lockbox": {
                "validation_outcomes_read": False,
                "confirmation_outcomes_read": False,
                "validation_signal_values_read": False,
                "confirmation_signal_values_read": False,
                "raw_eeg_amplitudes_read": False,
                "physiology_amplitudes_read": False,
            },
            "builder_defect": (
                builder_contract
            ),
            "duration_priors": (
                priors
            ),
            "event_reconstruction": (
                event_reconstruction
            ),
            "reference_validation": {
                "reference_participants": int(
                    len(
                        reference
                    )
                ),
                "passed_participants": int(
                    reference[
                        "reference_validation_passed"
                    ].sum()
                ),
                "all_reference_participants_passed": bool(
                    reference[
                        "reference_validation_passed"
                    ].all()
                )
                if not reference.empty
                else False,
            },
            "readme_sequence_manifest": {
                "rows": int(
                    len(
                        readme_manifest
                    )
                ),
                "participants": int(
                    readme_manifest[
                        "subject_id"
                    ].nunique()
                ),
                "direct_from_readme": True,
            },
            "combined_metadata_probe": (
                combined_metadata_summary
            ),
            "physiology_clock": (
                physio_contract_data
            ),
            "decision": {
                "corrected_discovery_master_ready": False,
                "candidate_table_authoritative": False,
                "next_phase": (
                    "targeted waveform-fingerprint "
                    "resolution and versioned lineage freeze"
                ),
                "required_waveform_targets": [
                    (
                        "primary event pairs with "
                        "selection_stability_count < 3"
                    ),
                    (
                        "sub-58 imagery derivative-gap "
                        "alignment"
                    ),
                ],
                "structurally_unrecoverable_without_new_source": [
                    (
                        "sub-01 acquisition context order "
                        "unless a new event/log source is found"
                    ),
                    (
                        "behavior rows beyond available "
                        "continuous-event evidence in truncated "
                        "recordings"
                    ),
                ],
            },
            "input_hashes": {
                name: sha256_file(
                    path
                )
                for name, path
                in inputs[
                    "paths"
                ].items()
            },
            "runtime_seconds": round(
                time.time()
                - started,
                2,
            ),
        }
        outputs[
            "summary"
        ].write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_report(
            outputs[
                "report"
            ],
            summary,
            subject_qc,
            reference_qc,
            mapping_qc,
        )
        note(
            "Event-grammar reconstruction completed. All reconstructed mappings remain non-authoritative."
        )

    release_bundle = make_zip(
        project,
        outputs.values(),
    )
    release_bundle_sha = release_bundle.with_suffix(
        ".zip.sha256"
    )
    release_bundle_sha.write_text(
        f"{sha256_file(release_bundle)}  "
        f"{release_bundle.name}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "EVENT-GRAMMAR RECONSTRUCTION COMPLETED"
    )
    print("=" * 100)
    print(
        f"Status:                         {summary['status']}"
    )
    print(
        "Behavior rows:                  "
        f"{event_reconstruction['behavior_rows']}"
    )
    print(
        "Selected stimulus-rating pairs: "
        f"{event_reconstruction['selected_pairs']}"
    )
    print(
        "Mapped behavior rows:            "
        f"{event_reconstruction['mapped_behavior_rows']}"
    )
    print(
        "Stable selected pairs:           "
        f"{event_reconstruction['stable_all_variant_pairs']}"
    )
    print(
        "Unstable selected pairs:         "
        f"{event_reconstruction['unstable_primary_pairs']}"
    )
    print(
        "Reference participants passed:   "
        f"{summary['reference_validation']['passed_participants']}"
        f"/{summary['reference_validation']['reference_participants']}"
    )
    print(
        "Authoritative canonical changed: False"
    )
    print(
        "Validation outcomes opened:      False"
    )
    print(
        "Confirmation outcomes opened:    False"
    )
    print(
        f"Report:                          {outputs['report']}"
    )
    print(
        f"Results ZIP:                     {release_bundle}"
    )
    print(
        f"Runtime:                         {time.time() - started:.1f} seconds"
    )
    return 0


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    raise SystemExit(
        main()
    )
