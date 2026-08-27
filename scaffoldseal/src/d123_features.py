from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from r1a_classical import (
    CONTINUOUS_DESCRIPTORS,
    MISSING_FLAGS,
    STANDARD_RESIDUES,
    TOPOLOGY_CATEGORIES,
    TOPOLOGY_COLUMNS,
    _is_n_methyl_token,
    _is_noncanonical_token,
)


DESCRIPTOR_COLUMNS = (*TOPOLOGY_COLUMNS, *CONTINUOUS_DESCRIPTORS, *MISSING_FLAGS)
N_GLOBAL_FEATURES = 27
MAX_MISSING_FRACTION = 0.50
MIN_VARIANCE = 0.0


def canonical_ids_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def canonical_payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def group_balanced_weights(frame: pd.DataFrame) -> pd.Series:
    required = {"curated_id", "source", "analogue_component_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing D1 grouping columns: {sorted(required - set(frame.columns))}")
    if frame["curated_id"].duplicated().any() or len(frame) == 0:
        raise ValueError("D1 requires one nonempty complete fit-scoped training frame")
    group = frame.loc[:, ["curated_id", "source", "analogue_component_id"]].copy()
    if group.isna().any().any():
        raise ValueError("D1 grouping metadata must be complete")
    source_count = int(group["source"].nunique())
    components_per_source = group.groupby("source", sort=True)[
        "analogue_component_id"
    ].nunique()
    records_per_group = group.groupby(
        ["source", "analogue_component_id"], sort=True
    )["curated_id"].size()
    n = len(group)
    values = []
    for row in group.itertuples(index=False):
        c_s = int(components_per_source.loc[row.source])
        n_sc = int(records_per_group.loc[(row.source, row.analogue_component_id)])
        values.append(float(n / (source_count * c_s * n_sc)))
    result = pd.Series(values, index=group["curated_id"].astype(str), name="d1_weight")
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any():
        raise ValueError("D1 generated invalid weights")
    if not np.isclose(float(result.mean()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("D1 weights do not have complete-frame mean one")
    source_totals = group.assign(weight=result.to_numpy()).groupby("source")["weight"].sum()
    if not np.allclose(
        source_totals.to_numpy(), np.repeat(n / source_count, source_count), rtol=0, atol=1e-10
    ):
        raise ValueError("D1 source totals are not equal")
    return result


def build_raw_descriptor_frame(
    analysis_path: Path, curated_public_path: Path
) -> pd.DataFrame:
    analysis = pd.read_csv(
        analysis_path,
        usecols=["curated_id", "canonical_smiles", "topology_signature", "ring_size"],
    )
    public = pd.read_csv(
        curated_public_path, usecols=["curated_id", "canonical_sequence"]
    )
    merged = analysis.merge(public, on="curated_id", validate="one_to_one")
    if len(merged) != 6895 or merged["curated_id"].nunique() != 6895:
        raise ValueError("D2 descriptor join must cover 6,895 unique records")
    rows: list[dict[str, float | str]] = []
    for row in merged.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        if mol is None:
            raise ValueError(f"Unparseable frozen SMILES: {row.curated_id}")
        topology = str(row.topology_signature).split("|", 1)[0]
        if topology not in TOPOLOGY_CATEGORIES:
            raise ValueError(f"Unknown frozen topology: {topology}")
        tokens = json.loads(str(row.canonical_sequence))
        record: dict[str, float | str] = {"curated_id": str(row.curated_id)}
        for category, column in zip(TOPOLOGY_CATEGORIES, TOPOLOGY_COLUMNS):
            record[column] = float(topology == category)
        record.update(
            {
                "ring_size": float(row.ring_size),
                "molecular_weight": float(Descriptors.MolWt(mol)),
                "clogp": float(Crippen.MolLogP(mol)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "hbd": float(Lipinski.NumHDonors(mol)),
                "hba": float(Lipinski.NumHAcceptors(mol)),
                "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
                "formal_charge": float(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
                "fraction_sp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                "stereocenter_count": float(
                    len(
                        Chem.FindMolChiralCenters(
                            mol, includeUnassigned=True, useLegacyImplementation=False
                        )
                    )
                ),
                "n_methyl_count": float(sum(_is_n_methyl_token(token) for token in tokens)),
                "noncanonical_count": float(
                    sum(_is_noncanonical_token(token) for token in tokens)
                ),
            }
        )
        for column in CONTINUOUS_DESCRIPTORS:
            record[f"{column}__missing"] = float(not np.isfinite(float(record[column])))
        rows.append(record)
    result = pd.DataFrame(rows, columns=["curated_id", *DESCRIPTOR_COLUMNS])
    if len(DESCRIPTOR_COLUMNS) != N_GLOBAL_FEATURES:
        raise RuntimeError("Frozen D2 global feature width changed")
    return result


@dataclass(frozen=True)
class FitScopedDescriptorTransform:
    training_ids_sha256: str
    active_columns: tuple[str, ...]
    inactive_columns: tuple[str, ...]
    medians: tuple[tuple[str, float], ...]
    means: tuple[tuple[str, float], ...]
    scales: tuple[tuple[str, float], ...]
    transform_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "scaffoldseal-d123-fit-transform-v1",
            "training_ids_sha256": self.training_ids_sha256,
            "active_columns": list(self.active_columns),
            "inactive_columns": list(self.inactive_columns),
            "medians": dict(self.medians),
            "means": dict(self.means),
            "scales": dict(self.scales),
            "max_missing_fraction": MAX_MISSING_FRACTION,
            "min_variance": MIN_VARIANCE,
            "output_columns": list(DESCRIPTOR_COLUMNS),
            "transform_sha256": self.transform_sha256,
        }

    @classmethod
    def from_dict(cls, serialized: dict[str, object]) -> "FitScopedDescriptorTransform":
        if serialized.get("schema_version") != "scaffoldseal-d123-fit-transform-v1":
            raise ValueError("D123 descriptor transform schema drifted")
        if serialized.get("max_missing_fraction") != MAX_MISSING_FRACTION:
            raise ValueError("D123 maximum missing fraction drifted")
        if serialized.get("min_variance") != MIN_VARIANCE:
            raise ValueError("D123 minimum variance drifted")
        if serialized.get("output_columns") != list(DESCRIPTOR_COLUMNS):
            raise ValueError("D123 descriptor output order drifted")
        payload = {
            "training_ids_sha256": str(serialized["training_ids_sha256"]),
            "active_columns": list(serialized["active_columns"]),
            "inactive_columns": list(serialized["inactive_columns"]),
            "medians": dict(serialized["medians"]),
            "means": dict(serialized["means"]),
            "scales": dict(serialized["scales"]),
            "max_missing_fraction": MAX_MISSING_FRACTION,
            "min_variance": MIN_VARIANCE,
            "output_columns": list(DESCRIPTOR_COLUMNS),
        }
        observed = str(serialized["transform_sha256"])
        if canonical_payload_hash(payload) != observed:
            raise ValueError("D123 serialized descriptor transform hash mismatch")
        continuous = set(CONTINUOUS_DESCRIPTORS)
        active = tuple(map(str, payload["active_columns"]))
        inactive = tuple(map(str, payload["inactive_columns"]))
        if set(active) & set(inactive) or set(active) | set(inactive) != continuous:
            raise ValueError("D123 serialized active/inactive partition drifted")
        for name in ("medians", "means", "scales"):
            if set(payload[name]) != set(active):
                raise ValueError(f"D123 serialized {name} do not match active columns")
        return cls(
            training_ids_sha256=payload["training_ids_sha256"],
            active_columns=active,
            inactive_columns=inactive,
            medians=tuple(sorted((str(k), float(v)) for k, v in payload["medians"].items())),
            means=tuple(sorted((str(k), float(v)) for k, v in payload["means"].items())),
            scales=tuple(sorted((str(k), float(v)) for k, v in payload["scales"].items())),
            transform_sha256=observed,
        )

    @classmethod
    def fit(
        cls, raw_descriptors: pd.DataFrame, training_ids: tuple[str, ...]
    ) -> "FitScopedDescriptorTransform":
        if len(set(training_ids)) != len(training_ids) or not training_ids:
            raise ValueError("Descriptor fit requires unique nonempty training IDs")
        indexed = raw_descriptors.set_index("curated_id", verify_integrity=True)
        if not set(training_ids).issubset(indexed.index):
            raise ValueError("Descriptor training IDs are not fully covered")
        train = indexed.loc[list(training_ids), list(CONTINUOUS_DESCRIPTORS)].apply(
            pd.to_numeric, errors="coerce"
        )
        train = train.where(np.isfinite(train), np.nan)
        active: list[str] = []
        inactive: list[str] = []
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in CONTINUOUS_DESCRIPTORS:
            values = train[column]
            missing_fraction = float(values.isna().mean())
            if missing_fraction > MAX_MISSING_FRACTION or not values.notna().any():
                inactive.append(column)
                continue
            median = float(values.median())
            imputed = values.fillna(median).to_numpy(float)
            variance = float(np.var(imputed, ddof=0))
            if not variance > MIN_VARIANCE:
                inactive.append(column)
                continue
            active.append(column)
            medians[column] = median
            means[column] = float(np.mean(imputed))
            scales[column] = float(np.sqrt(variance))
        payload = {
            "training_ids_sha256": canonical_ids_hash(training_ids),
            "active_columns": active,
            "inactive_columns": inactive,
            "medians": medians,
            "means": means,
            "scales": scales,
            "max_missing_fraction": MAX_MISSING_FRACTION,
            "min_variance": MIN_VARIANCE,
            "output_columns": list(DESCRIPTOR_COLUMNS),
        }
        return cls(
            training_ids_sha256=payload["training_ids_sha256"],
            active_columns=tuple(active),
            inactive_columns=tuple(inactive),
            medians=tuple(sorted(medians.items())),
            means=tuple(sorted(means.items())),
            scales=tuple(sorted(scales.items())),
            transform_sha256=canonical_payload_hash(payload),
        )

    def transform(
        self, raw_descriptors: pd.DataFrame, ids: tuple[str, ...]
    ) -> np.ndarray:
        indexed = raw_descriptors.set_index("curated_id", verify_integrity=True)
        if not set(ids).issubset(indexed.index):
            raise ValueError("Descriptor transform IDs are not fully covered")
        frame = indexed.loc[list(ids), list(DESCRIPTOR_COLUMNS)].copy()
        medians, means, scales = map(dict, (self.medians, self.means, self.scales))
        for column in CONTINUOUS_DESCRIPTORS:
            values = pd.to_numeric(frame[column], errors="coerce")
            values = values.where(np.isfinite(values), np.nan)
            if column in self.inactive_columns:
                frame[column] = 0.0
            else:
                frame[column] = (
                    values.fillna(medians[column]) - means[column]
                ) / scales[column]
        output = frame.loc[:, DESCRIPTOR_COLUMNS].to_numpy(np.float32, copy=True)
        if output.shape != (len(ids), N_GLOBAL_FEATURES) or not np.isfinite(output).all():
            raise ValueError("D2 transformed descriptor matrix is invalid")
        return output
