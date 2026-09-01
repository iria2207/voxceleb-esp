#!/usr/bin/env python3
#Generate the speaker-disjoint training list and verification trial lists.

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import pandas as pd


def canonical_path(row: pd.Series) -> str:
    speaker = str(row["speaker_id"]).strip()
    video = str(row["video_id"]).strip()
    utterance = str(row.get("utterance_id", "")).strip()
    if utterance:
        suffix = utterance if utterance.endswith(".wav") else f"{utterance}.wav"
        return f"wav/{speaker}/{video}/{suffix}"

    for column in ("export_path", "audio_path"):
        value = str(row.get(column, "")).strip().replace("\\", "/")
        if not value:
            continue
        if "/dev/wav/" in value:
            return "wav/" + value.split("/dev/wav/", 1)[1]
        if value.startswith("dev/wav/"):
            return value[4:]
        if value.startswith("wav/"):
            return value
    raise ValueError("Cannot obtain a VoxCeleb-style path from the metadata row")


def sample_pairs(pairs: list[tuple[str, str]], count: int, rng: random.Random) -> list[tuple[str, str]]:
    unique = sorted({tuple(sorted(pair)) for pair in pairs})
    if len(unique) < count:
        raise ValueError(f"Only {len(unique)} eligible target pairs are available; {count} are required")
    return rng.sample(unique, count)


def target_pairs(rows: pd.DataFrame, protocol: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for _, speaker_rows in rows.groupby("speaker_id"):
        records = speaker_rows[["video_id", "trial_path"]].to_dict("records")
        for first, second in itertools.combinations(records, 2):
            same_video = str(first["video_id"]) == str(second["video_id"])
            if (protocol == "A" and same_video) or (protocol == "B" and not same_video):
                pairs.append((first["trial_path"], second["trial_path"]))
    return pairs


def non_target_pairs(rows: pd.DataFrame, count: int, rng: random.Random) -> list[tuple[str, str]]:
    by_gender: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for (gender, speaker), group in rows.groupby(["gender", "speaker_id"]):
        by_gender.setdefault(str(gender), {})[str(speaker)] = list(
            group[["video_id", "trial_path"]].itertuples(index=False, name=None)
        )

    valid_genders = [gender for gender, speakers in by_gender.items() if len(speakers) >= 2]
    if not valid_genders:
        raise ValueError("At least two speakers with the same gender label are required")

    selected: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(100_000, count * 200)
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        gender = rng.choice(valid_genders)
        speakers = list(by_gender[gender])
        first_speaker, second_speaker = rng.sample(speakers, 2)
        first_video, first_path = rng.choice(by_gender[gender][first_speaker])
        second_video, second_path = rng.choice(by_gender[gender][second_speaker])
        if first_video == second_video:
            continue
        selected.add(tuple(sorted((first_path, second_path))))

    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} unique non-target pairs could be generated")
    return sorted(selected)


def write_trials(path: Path, targets: list[tuple[str, str]], non_targets: list[tuple[str, str]], rng: random.Random) -> None:
    trials = [(1, *pair) for pair in targets] + [(0, *pair) for pair in non_targets]
    rng.shuffle(trials)
    path.write_text("".join(f"{label} {first} {second}\n" for label, first, second in trials), encoding="utf-8")


def write_gender_subsets(path: Path, rows: pd.DataFrame) -> None:
    gender_by_path = rows.set_index("trial_path")["gender"].astype(str).to_dict()
    parsed = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    for gender in ("female", "male"):
        subset = [parts for parts in parsed if gender_by_path.get(parts[1]) == gender_by_path.get(parts[2]) == gender]
        output = path.with_name(f"{path.stem}_{gender}{path.suffix}")
        output.write_text("".join(" ".join(parts) + "\n" for parts in subset), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Exported metadata.csv with 25,000 utterances")
    parser.add_argument("--split", default="evaluation/split_speakers.csv")
    parser.add_argument("--output-dir", default="workspace/evaluation")
    parser.add_argument("--target-trials", type=int, default=5_000)
    parser.add_argument("--non-target-trials", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, low_memory=False)
    split = pd.read_csv(args.split)
    required_metadata = {"speaker_id", "video_id"}
    required_split = {"speaker_id", "gender", "split"}
    if missing := required_metadata - set(metadata.columns):
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if missing := required_split - set(split.columns):
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    metadata = metadata.drop(columns=[column for column in ("gender", "split") if column in metadata.columns])
    rows = metadata.merge(split[["speaker_id", "gender", "split"]], on="speaker_id", how="inner")
    rows["trial_path"] = rows.apply(canonical_path, axis=1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = rows[rows["split"] == "train"].sort_values(["speaker_id", "trial_path"])
    (output_dir / "train_list.txt").write_text(
        "".join(f"{row.speaker_id} {row.trial_path}\n" for row in train.itertuples()),
        encoding="utf-8",
    )

    for split_name in ("validation", "test"):
        split_rows = rows[rows["split"] == split_name].copy()
        for protocol in ("A", "B"):
            rng = random.Random(f"{args.seed}-{split_name}-{protocol}")
            targets = sample_pairs(target_pairs(split_rows, protocol), args.target_trials, rng)
            non_targets = non_target_pairs(split_rows, args.non_target_trials, rng)
            prefix = "val" if split_name == "validation" else "test"
            output = output_dir / f"{prefix}_trials_{protocol}.txt"
            write_trials(output, targets, non_targets, rng)
            if split_name == "test":
                write_gender_subsets(output, split_rows)
            print(f"{output}: {len(targets)} target + {len(non_targets)} non-target trials")


if __name__ == "__main__":
    main()
