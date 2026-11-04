"""Generate a small, clearly synthetic diffusion QC fixture."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


N_VOLUMES = 105
FIRST_RANKED = [46, 60, 92, 79, 69, 86, 48, 27, 98, 94]


def _brain_phantom(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    outer = (x / 0.82) ** 2 + (y / 0.94) ** 2 + (z / 0.78) ** 2 <= 1
    inner = (x / 0.58) ** 2 + (y / 0.70) ** 2 + (z / 0.57) ** 2 <= 1
    tissue = np.where(inner, 0.85, 0.55).astype(np.float32)
    tissue *= outer
    tissue -= 0.16 * np.exp(-((x / 0.17) ** 2 + (y / 0.35) ** 2 + (z / 0.35) ** 2))
    return np.clip(tissue, 0.0, 1.0)


def _ranked_volumes() -> list[int]:
    return FIRST_RANKED + [volume for volume in range(N_VOLUMES) if volume not in FIRST_RANKED]


def create_synthetic_dataset(output_dir: str | Path, *, n_subjects: int = 3, seed: int = 7) -> Path:
    """Create up to five subjects with matching CSV, NIfTI, bval, and bvec files."""

    if not 1 <= n_subjects <= 5:
        raise ValueError("n_subjects must be between 1 and 5")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (24, 24, 16)
    base = _brain_phantom(shape)

    with (output_dir / "synthetic_subjects.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["SubjectID", "ScanDate", "Ylin", "Year", "Month", "Group"])
        for subject_index in range(1, n_subjects + 1):
            subject_id = f"sub-{subject_index:02d}"
            year, month = 2022, subject_index
            ylin = 0.04 + subject_index * 0.035
            group = "HIGH" if subject_index % 2 else "LOW"
            scan_date = year + (month - 0.5) / 12
            writer.writerow([subject_id, f"{scan_date:.6f}", f"{ylin:.6f}", year, month, group])

            raw_dir = output_dir / subject_id / "dMRI" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            data = np.empty((*shape, N_VOLUMES), dtype=np.float32)
            bvalues: list[int] = []
            vectors = np.zeros((3, N_VOLUMES), dtype=np.float32)
            subject_scale = 1.0 + subject_index * 0.04
            for volume in range(N_VOLUMES):
                if volume == 0:
                    bvalue = 0
                elif volume < 70:
                    bvalue = 1000
                else:
                    bvalue = 2000
                bvalues.append(bvalue)
                angle = volume * 2.399963229728653
                z_component = 1.0 - 2.0 * (volume % 53) / 52.0
                radius = np.sqrt(max(0.0, 1.0 - z_component * z_component))
                direction = np.array(
                    [radius * np.cos(angle), radius * np.sin(angle), z_component],
                    dtype=np.float32,
                )
                vectors[:, volume] = direction
                attenuation = np.exp(-bvalue * (0.18 + 0.38 * direction[2] ** 2) / 2000)
                signal = base * subject_scale * attenuation
                signal = signal + rng.normal(0.0, 0.018, shape).astype(np.float32)
                if volume in {19, 57, 91}:
                    signal[:, shape[1] // 2, :] += 0.22
                data[..., volume] = np.clip(signal, 0.0, 1.2)

            nib.save(nib.Nifti1Image(data, np.eye(4)), raw_dir / "AP.nii.gz")
            (raw_dir / "AP.bval").write_text(
                " ".join(str(value) for value in bvalues) + "\n", encoding="utf-8"
            )
            (raw_dir / "AP.bvec").write_text(
                "\n".join(" ".join(f"{value:.7f}" for value in row) for row in vectors)
                + "\n",
                encoding="utf-8",
            )

    with (output_dir / "synthetic_volume_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["volume", "n_bad", "average_minus_ylin"])
        for rank, volume in enumerate(_ranked_volumes()):
            writer.writerow([volume, max(0, 63 - rank), f"{-0.75 + rank * 0.0125:.8f}"])

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic_dwi_dataset"))
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output_dir = create_synthetic_dataset(args.output, n_subjects=args.subjects, seed=args.seed)
    print(f"Created synthetic QC inputs and {args.subjects} subjects in {output_dir}")


if __name__ == "__main__":
    main()