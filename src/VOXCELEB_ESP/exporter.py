#Exportación a formato VoxCeleb/VoxCeleb-ESP.

from __future__ import annotations

import itertools
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


class VoxCelebExporter:
    def __init__(self, config):
        self.output_base = Path(config.output_base)
        self.dev_dir = self.output_base / "dev"
        self.wav_dir = self.dev_dir / "wav"

        self.meta_path = self.output_base / "voxceleb_esp_meta.csv"
        self.utt_meta_path = self.output_base / "voxceleb_esp_utterances.csv"
        self.trials_a_path = self.output_base / "trials_A.txt"
        self.trials_b_path = self.output_base / "trials_B.txt"

        self.wav_dir.mkdir(parents=True, exist_ok=True)
        print("Exporter inicializado")

    def export_from_approved(self, approved_csv: Path, celebrities_config: List) -> pd.DataFrame:
        approved_csv = Path(approved_csv)
        if not approved_csv.exists():
            raise FileNotFoundError(f"No existe approved_segments.csv: {approved_csv}")

        df = pd.read_csv(approved_csv)
        if "approved" in df.columns:
            approved = pd.to_numeric(df["approved"], errors="coerce").fillna(0).astype(int)
            df = df.loc[approved == 1].copy()
        if df.empty:
            raise ValueError("No hay segmentos aprobados para exportar")

        required = {"speaker_id", "video_id", "candidate_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en approved_segments.csv: {sorted(missing)}")

        names, genders = self._celeb_maps(celebrities_config)
        exported_rows = []
        selected_destinations = set()
        for ordinal, (_, row) in enumerate(df.iterrows()):
            speaker_id = str(row["speaker_id"])
            video_id = str(row["video_id"])
            source = Path(str(row["candidate_path"]))
            if not source.exists():
                print(f"No encontrado, se omite: {source}")
                continue

            approved_index = row.get("approved_index", ordinal)
            try:
                approved_index = int(approved_index)
            except (TypeError, ValueError):
                approved_index = ordinal
            utterance_id = f"{video_id}_{approved_index:05d}"
            destination = self.wav_dir / speaker_id / video_id / f"{utterance_id}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            selected_destinations.add(destination.resolve())

            output_row = row.to_dict()
            output_row.update({
                "utterance_id": utterance_id,
                "speaker_name": names.get(speaker_id, output_row.get("speaker_name", speaker_id)),
                "gender": genders.get(speaker_id, output_row.get("gender", "unknown")),
                "set": "dev",
                "audio_path": str(destination),
                "export_path": self._rel_trial_path(destination, self.dev_dir),
            })
            exported_rows.append(output_row)

       
        removed = 0
        for speaker_id in sorted(df["speaker_id"].astype(str).unique()):
            speaker_dir = self.wav_dir / speaker_id
            if not speaker_dir.exists():
                continue
            for old_wav in speaker_dir.rglob("*.wav"):
                if old_wav.resolve() not in selected_destinations:
                    old_wav.unlink()
                    removed += 1
        if removed:
            print(f"🧹 WAV antiguos retirados de la exportación: {removed}")

        utterances = pd.DataFrame(exported_rows)
        if utterances.empty:
            raise RuntimeError("No se pudo exportar ningún audio aprobado")
        utterances.to_csv(self.utt_meta_path, index=False)

        speaker_rows = []
        for speaker_id, group in utterances.groupby("speaker_id"):
            speaker_rows.append({
                "VoxCeleb ID": speaker_id,
                "Gender": genders.get(str(speaker_id), "unknown"),
                "Set": "dev",
                "Name": names.get(str(speaker_id), str(speaker_id)),
                "Num utterances": int(len(group)),
                "Num videos": int(group["video_id"].nunique()),
                "Total duration": float(group["duration"].sum()) if "duration" in group else None,
            })
        pd.DataFrame(speaker_rows).to_csv(self.meta_path, index=False)
        return utterances

    def make_trials(self, max_trials: int = 200000) -> None:
        if not self.utt_meta_path.exists():
            raise FileNotFoundError(f"No existe metadata de utterances: {self.utt_meta_path}")
        utterances = pd.read_csv(self.utt_meta_path)
        trials_a = self._generate_trials_from_df(utterances, mode="A")[:max_trials]
        trials_b = self._generate_trials_from_df(utterances, mode="B")[:max_trials]
        self._save_trials(trials_a, self.trials_a_path)
        self._save_trials(trials_b, self.trials_b_path)

    @staticmethod
    def _celeb_maps(celebrities_config: List) -> Tuple[Dict[str, str], Dict[str, str]]:
        names = {getattr(c, "id", ""): getattr(c, "name", "") for c in celebrities_config}
        genders = {getattr(c, "id", ""): getattr(c, "gender", "unknown") for c in celebrities_config}
        return names, genders

    @staticmethod
    def _rel_trial_path(path: Path, root: Path) -> str:
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    def export_from_metadata(self, processed_dir: Path, metadata_csv: Path, celebrities_config: List) -> None:
        processed_dir = Path(processed_dir)
        metadata_csv = Path(metadata_csv)

        if not metadata_csv.exists():
            raise FileNotFoundError(f"No existe metadata_csv: {metadata_csv}")

        df = pd.read_csv(metadata_csv)
        if df.empty:
            raise ValueError("metadata.csv está vacío")

        required = {"speaker_id", "video_id", "audio_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"metadata.csv no tiene columnas requeridas: {sorted(missing)}")

        names, genders = self._celeb_maps(celebrities_config)
        exported_rows = []

        for idx, row in df.iterrows():
            spk = str(row["speaker_id"])
            vid = str(row["video_id"])

            src = processed_dir / str(row["audio_path"])
            if not src.exists():
                # Compatibilidad con rutas guardadas absolutas.
                src = Path(str(row["audio_path"]))
            if not src.exists():
                print(f"No encontrado, se omite: {row['audio_path']}")
                continue

            utt = str(row.get("utterance_id", f"{idx:05d}"))
            if not utt.endswith(".wav"):
                utt_name = f"{utt}.wav"
            else:
                utt_name = utt

            dest = self.wav_dir / spk / vid / utt_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

            out = row.to_dict()
            out["name"] = names.get(spk, spk)
            out["gender"] = genders.get(spk, "unknown")
            out["set"] = "dev"
            out["export_path"] = self._rel_trial_path(dest, self.dev_dir)
            exported_rows.append(out)

        utt_df = pd.DataFrame(exported_rows)
        utt_df.to_csv(self.utt_meta_path, index=False)

        speaker_rows = []
        for spk, g in utt_df.groupby("speaker_id"):
            speaker_rows.append({
                "VoxCeleb ID": spk,
                "Gender": genders.get(spk, "unknown"),
                "Set": "dev",
                "Name": names.get(spk, spk),
                "Num utterances": int(len(g)),
                "Num videos": int(g["video_id"].nunique()),
                "Total duration": float(g["duration"].sum()) if "duration" in g else None,
            })
        pd.DataFrame(speaker_rows).to_csv(self.meta_path, index=False)

        trials_a = self._generate_trials_from_df(utt_df, mode="A")
        trials_b = self._generate_trials_from_df(utt_df, mode="B")
        self._save_trials(trials_a, self.trials_a_path)
        self._save_trials(trials_b, self.trials_b_path)

        print("Exportación completada.")
        print(f"Speaker metadata: {self.meta_path}")
        print(f"Utterance metadata: {self.utt_meta_path}")
        print(f"Trial A: {self.trials_a_path}")
        print(f"Trial B: {self.trials_b_path}")

    def export(self, refined_dir: Path, celebrities_config: List):
        refined_dir = Path(refined_dir)
        rows = []
        tmp_processed = self.output_base / "_tmp_export_processed"
        if tmp_processed.exists():
            shutil.rmtree(tmp_processed)
        tmp_processed.mkdir(parents=True, exist_ok=True)

        for spk_dir in sorted(refined_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            for i, wav in enumerate(sorted(spk_dir.glob("*.wav"))):
                rel = Path(spk_dir.name) / wav.name
                dest = tmp_processed / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav, dest)
                rows.append({
                    "speaker_id": spk_dir.name,
                    "video_id": "unknown_video",
                    "utterance_id": f"{i:05d}",
                    "audio_path": str(rel),
                    "duration": None,
                })
        tmp_meta = self.output_base / "_tmp_export_metadata.csv"
        pd.DataFrame(rows).to_csv(tmp_meta, index=False)
        self.export_from_metadata(tmp_processed, tmp_meta, celebrities_config)

    def _generate_trials_from_df(self, df: pd.DataFrame, mode: str) -> List[Tuple[int, str, str]]:
        rng = random.Random(1234)
        rows = df.to_dict("records")
        by_spk: Dict[str, List[dict]] = {}
        for row in rows:
            by_spk.setdefault(str(row["speaker_id"]), []).append(row)

        trials: List[Tuple[int, str, str]] = []

        # TARGETS
        for spk, items in by_spk.items():
            for a, b in itertools.combinations(items, 2):
                same_video = str(a["video_id"]) == str(b["video_id"])
                if mode == "A" and same_video:
                    trials.append((1, str(a["export_path"]), str(b["export_path"])))
                elif mode == "B" and not same_video:
                    trials.append((1, str(a["export_path"]), str(b["export_path"])))

        spk_ids = sorted(by_spk.keys())
        non_target_goal = min(max(len(trials) * 4, 1000 if trials else 0), 20000)
        seen = {self._pair_key(t[1], t[2], t[0]) for t in trials}
        attempts = 0

        while len([t for t in trials if t[0] == 0]) < non_target_goal and attempts < non_target_goal * 20 + 100:
            attempts += 1
            if len(spk_ids) < 2:
                break
            spk1, spk2 = rng.sample(spk_ids, 2)
            a = rng.choice(by_spk[spk1])
            b = rng.choice(by_spk[spk2])
            if str(a["video_id"]) == str(b["video_id"]):
                continue
            key = self._pair_key(str(a["export_path"]), str(b["export_path"]), 0)
            if key in seen:
                continue
            seen.add(key)
            trials.append((0, str(a["export_path"]), str(b["export_path"])))

        return self._dedup_trials(trials)

    @staticmethod
    def _pair_key(w1: str, w2: str, target: int) -> Tuple[int, str, str]:
        a, b = sorted([w1, w2])
        return int(target), a, b

    def _dedup_trials(self, trials: Iterable[Tuple[int, str, str]]) -> List[Tuple[int, str, str]]:
        seen = set()
        cleaned = []
        for target, w1, w2 in trials:
            key = self._pair_key(w1, w2, target)
            if key not in seen and w1 != w2:
                seen.add(key)
                cleaned.append(key)
        return cleaned

    def _save_trials(self, trials: List[Tuple[int, str, str]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for target, w1, w2 in trials:
                f.write(f"{target} {w1} {w2}\n")
        print(f"Trials guardados: {path} ({len(trials):,} pares)")
