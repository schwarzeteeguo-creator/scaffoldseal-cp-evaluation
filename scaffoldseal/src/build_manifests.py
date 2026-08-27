"""Build the frozen ScaffoldSeal-CP Milestone-0 data boundary.

The build is deterministic and label-blind after curation. Partition allocation
uses source/component block sizes and structural metadata only. Final labels are
written to the external vault before the header-only access log is created.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
import yaml
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.optimize import Bounds, LinearConstraint, milp


RDLogger.DisableLog("rdApp.*")
PARTITIONS = ("train", "validation", "calibration", "final_test")
PUBLIC_MANIFESTS = (
    "curation_manifest.csv",
    "curated_records_public.csv",
    "analogue_edges.csv",
    "analogue_components.csv",
    "source_component_blocks.csv",
    "split_manifest_public.csv",
    "pcppred_id_ambiguity.csv",
)
OUTCOME_DERIVED_COLUMNS = frozenset(
    {
        "permeability",
        "replicate_min",
        "replicate_max",
        "replicate_spread",
        "label",
        "target",
        "outcome",
        "papp",
        "y",
    }
)
OUTCOME_NAME_FRAGMENTS = (
    "permeab",
    "replicate",
    "label",
    "target",
    "outcome",
    "papp",
    "response",
)
CURATED_PUBLIC_COLUMNS = (
    "curated_id",
    "molecule_id",
    "canonical_smiles",
    "source",
    "year",
    "version",
    "topology_signature",
    "canonical_sequence",
    "ring_size",
    "main_chain_length",
    "raw_ids_all",
    "raw_ids_used",
    "n_raw_rows",
    "n_uncensored_used",
)
PUBLIC_SCHEMA_ALLOWLISTS = {
    "curation_manifest.csv": (
        "raw_id",
        "raw_group_id",
        "source",
        "year",
        "version",
        "canonical_smiles",
        "topology_signature",
        "canonical_sequence",
        "is_censored",
        "curation_status",
        "exclusion_or_link_reason",
        "curated_id",
    ),
    "curated_records_public.csv": CURATED_PUBLIC_COLUMNS,
    "analogue_edges.csv": (
        "molecule_id_a",
        "molecule_id_b",
        "edge_types",
        "ecfp4_tanimoto",
    ),
    "analogue_components.csv": (
        "molecule_id",
        "canonical_smiles",
        "molecular_weight",
        "sources",
        "n_sources",
        "analogue_component_id",
        "analogue_component_size",
        "component_sources",
    ),
    "source_component_blocks.csv": (
        "sealed_block_id",
        "n_curated_rows",
        "n_unique_molecules",
        "n_analogue_components",
        "n_sources",
        "sources",
        "topologies",
        "ring_sizes",
        "partition",
    ),
    "split_manifest_public.csv": (
        "curated_id",
        "molecule_id",
        "canonical_smiles",
        "source",
        "year",
        "version",
        "topology_signature",
        "ring_size",
        "analogue_component_id",
        "sealed_block_id",
        "partition",
        "raw_ids_all",
        "raw_ids_used",
    ),
    "pcppred_id_ambiguity.csv": (
        "pcppred_split",
        "pcppred_id",
        "canonical_smiles",
        "direct_raw_id_structure_match",
        "raw_ids_for_same_structure",
        "raw_sources_for_same_structure",
        "ambiguity_reason",
    ),
}
ACCESS_LOG_HEADER = (
    "timestamp",
    "person_or_process",
    "reason",
    "files_accessed",
    "command_or_script",
    "git_commit",
    "result_location",
    "authorized_by",
    "deviation_id",
)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if (self.size[a], a) < (self.size[b], b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def stable_id(prefix: str, values: Iterable[object], length: int = 16) -> str:
    text = "\x1f".join("" if value is None else str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.write_bytes(payload)
    return sha256_bytes(payload)


def clean_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_sequence(value: object) -> tuple[str, ...] | None:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    return tuple(str(token).strip() for token in parsed)


def cyclic_rotations(sequence: Sequence[str]) -> list[tuple[str, ...]]:
    sequence = tuple(sequence)
    if not sequence:
        return [sequence]
    return [sequence[index:] + sequence[:index] for index in range(len(sequence))]


def canonical_cycle(sequence: Sequence[str]) -> tuple[str, ...]:
    return min(cyclic_rotations(sequence))


def topology_signature(row: pd.Series, sequence: tuple[str, ...] | None) -> str:
    shape = clean_scalar(row.get("Molecule_Shape")) or "unknown"
    total = int(float(row["Monomer_Length"])) if clean_scalar(row.get("Monomer_Length")) else -1
    main = (
        int(float(row["Monomer_Length_in_Main_Chain"]))
        if clean_scalar(row.get("Monomer_Length_in_Main_Chain"))
        else -1
    )
    helm = clean_scalar(row.get("HELM"))
    explicit_head_to_tail = (
        shape.lower() == "circle"
        and total == main
        and main > 0
        and f",1:R1-{main}:R2" in helm
    )
    if explicit_head_to_tail:
        topology = "circle_head_to_tail"
    elif shape.lower() == "lariat":
        topology = "lariat"
    elif shape.lower() == "circle":
        topology = "circle_other"
    else:
        topology = shape.lower()
    sequence_length = len(sequence) if sequence is not None else -1
    return f"{topology}|total={total}|main={main}|tokens={sequence_length}"


def canonical_sequence_text(
    sequence: tuple[str, ...] | None, topology: str
) -> str:
    if sequence is None:
        return ""
    if topology.startswith("circle_head_to_tail"):
        sequence = canonical_cycle(sequence)
    return json.dumps(list(sequence), ensure_ascii=False, separators=(",", ":"))


def is_irrecoverably_censored(row: pd.Series) -> bool:
    return bool(
        clean_scalar(row.get("Detection_Limit_1"))
        or clean_scalar(row.get("Detection_Limit_2"))
    )


@dataclass
class CurationResult:
    curated: pd.DataFrame
    curation_manifest: pd.DataFrame
    raw_public: pd.DataFrame
    flow: dict[str, int]


def curate(raw: pd.DataFrame) -> CurationResult:
    work = raw.copy()
    work["_raw_row"] = np.arange(len(work))
    work["_raw_id"] = ["PAMPA:" + clean_scalar(value) for value in work["ID"]]
    work["_label"] = pd.to_numeric(work["Permeability"], errors="coerce")
    mols = [Chem.MolFromSmiles(clean_scalar(value)) for value in work["SMILES"]]
    work["_canonical_smiles"] = [
        Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""
        for mol in mols
    ]
    sequences = [parse_sequence(value) for value in work["Sequence"]]
    topologies = [
        topology_signature(row, sequence)
        for (_, row), sequence in zip(work.iterrows(), sequences)
    ]
    work["_topology_signature"] = topologies
    work["_canonical_sequence"] = [
        canonical_sequence_text(sequence, topology)
        for sequence, topology in zip(sequences, topologies)
    ]
    work["_is_censored"] = [is_irrecoverably_censored(row) for _, row in work.iterrows()]
    work["_parse_ok"] = work["_canonical_smiles"].ne("") & work["_canonical_sequence"].ne("")
    work["_group_id"] = [
        stable_id("GRP", (source, smiles))
        if smiles
        else stable_id("RAW", (raw_id,))
        for source, smiles, raw_id in zip(
            work["Source"], work["_canonical_smiles"], work["_raw_id"]
        )
    ]

    status: dict[int, tuple[str, str, str]] = {}
    curated_rows: list[dict[str, object]] = []

    for index, row in work.loc[~work["_parse_ok"]].iterrows():
        reason = "invalid_structure" if not row["_canonical_smiles"] else "invalid_sequence"
        status[index] = ("excluded", reason, "")

    valid = work.loc[work["_parse_ok"]].copy()
    for (source, smiles), group in valid.groupby(
        ["Source", "_canonical_smiles"], sort=True, dropna=False
    ):
        usable = group.loc[(~group["_is_censored"]) & group["_label"].notna()].copy()
        if usable.empty:
            for index in group.index:
                status[index] = ("excluded", "irrecoverably_censored_or_missing_label", "")
            continue

        compatibility_fields = (
            "Year",
            "Version",
            "_topology_signature",
            "_canonical_sequence",
        )
        conflicts = [
            field
            for field in compatibility_fields
            if usable[field].map(clean_scalar).nunique(dropna=False) != 1
        ]
        if conflicts:
            reason = "incompatible_same_source_group:" + "|".join(conflicts)
            for index in group.index:
                status[index] = ("excluded", reason, "")
            continue

        representative = usable.sort_values(["ID", "_raw_row"], kind="stable").iloc[0]
        curated_id = stable_id("SSCP", (source, smiles))
        values = usable["_label"].astype(float).to_numpy()
        all_ids = sorted(group["_raw_id"].astype(str))
        used_ids = sorted(usable["_raw_id"].astype(str))
        molecule_id = stable_id("MOL", (smiles,))
        curated_rows.append(
            {
                "curated_id": curated_id,
                "molecule_id": molecule_id,
                "canonical_smiles": smiles,
                "source": clean_scalar(source),
                "year": clean_scalar(representative["Year"]),
                "version": clean_scalar(representative["Version"]),
                "topology_signature": representative["_topology_signature"],
                "canonical_sequence": representative["_canonical_sequence"],
                "ring_size": int(float(representative["Monomer_Length"])),
                "main_chain_length": int(
                    float(representative["Monomer_Length_in_Main_Chain"])
                ),
                "raw_ids_all": "|".join(all_ids),
                "raw_ids_used": "|".join(used_ids),
                "n_raw_rows": int(len(group)),
                "n_uncensored_used": int(len(usable)),
                "replicate_min": float(np.min(values)),
                "replicate_max": float(np.max(values)),
                "replicate_spread": float(np.max(values) - np.min(values)),
                "permeability": float(np.median(values)),
            }
        )
        for index, row in group.iterrows():
            if row["_is_censored"]:
                status[index] = (
                    "excluded_linked_to_curated_group",
                    "irrecoverably_censored_row",
                    curated_id,
                )
            else:
                status[index] = ("included", "compatible_uncensored_replicate", curated_id)

    curated = pd.DataFrame(curated_rows).sort_values("curated_id").reset_index(drop=True)
    manifest_rows = []
    for index, row in work.sort_values("_raw_row").iterrows():
        row_status, reason, curated_id = status[index]
        manifest_rows.append(
            {
                "raw_id": row["_raw_id"],
                "raw_group_id": row["_group_id"],
                "source": clean_scalar(row["Source"]),
                "year": clean_scalar(row["Year"]),
                "version": clean_scalar(row["Version"]),
                "canonical_smiles": row["_canonical_smiles"],
                "topology_signature": row["_topology_signature"],
                "canonical_sequence": row["_canonical_sequence"],
                "is_censored": bool(row["_is_censored"]),
                "curation_status": row_status,
                "exclusion_or_link_reason": reason,
                "curated_id": curated_id,
            }
        )
    curation_manifest = pd.DataFrame(manifest_rows)
    raw_public = curation_manifest[
        ["raw_id", "source", "year", "version", "raw_group_id", "curated_id", "curation_status"]
    ].copy()
    flow = {
        "raw_rows": int(len(work)),
        "invalid_structure_or_sequence_rows": int((~work["_parse_ok"]).sum()),
        "censored_rows": int(work["_is_censored"].sum()),
        "curated_source_structure_groups": int(len(curated)),
        "curated_unique_molecules": int(curated["molecule_id"].nunique()),
        "curated_sources": int(curated["source"].nunique()),
        "collapsed_groups": int((curated["n_uncensored_used"] > 1).sum()),
        "excluded_raw_rows": int(
            curation_manifest["curation_status"].str.startswith("excluded").sum()
        ),
        "incompatible_groups": int(
            curation_manifest.loc[
                curation_manifest["exclusion_or_link_reason"].str.startswith("incompatible"),
                "raw_group_id",
            ].nunique()
        ),
    }
    return CurationResult(curated, curation_manifest, raw_public, flow)


def parse_sequence_text(value: str) -> tuple[str, ...]:
    parsed = json.loads(value)
    return tuple(str(token) for token in parsed)


def token_one_edit(
    sequence_a: Sequence[str],
    sequence_b: Sequence[str],
    topology_a: str,
    topology_b: str,
) -> bool:
    if topology_a != topology_b or len(sequence_a) != len(sequence_b):
        return False
    if topology_a.startswith("circle_head_to_tail"):
        alignments = cyclic_rotations(sequence_b)
    else:
        alignments = [tuple(sequence_b)]
    return any(
        sum(left != right for left, right in zip(sequence_a, aligned)) == 1
        for aligned in alignments
    )


def masked_cycle(sequence: Sequence[str], position: int) -> tuple[str, ...]:
    masked = list(sequence)
    masked[position] = "*"
    return canonical_cycle(masked)


@dataclass
class AnalogueResult:
    molecules: pd.DataFrame
    edges: pd.DataFrame


def build_analogue_graph(curated: pd.DataFrame, config: dict) -> AnalogueResult:
    molecule_rows = []
    representations: dict[str, set[tuple[str, tuple[str, ...]]]] = collections.defaultdict(set)
    for molecule_id, group in curated.groupby("molecule_id", sort=True):
        smiles = group["canonical_smiles"].iloc[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Curated molecule failed reparsing: {molecule_id}")
        for _, row in group.iterrows():
            representations[molecule_id].add(
                (
                    str(row["topology_signature"]),
                    parse_sequence_text(str(row["canonical_sequence"])),
                )
            )
        molecule_rows.append(
            {
                "molecule_id": molecule_id,
                "canonical_smiles": smiles,
                "molecular_weight": float(Descriptors.MolWt(mol)),
                "sources": "|".join(sorted(set(group["source"].astype(str)))),
                "n_sources": int(group["source"].nunique()),
                "_mol": mol,
            }
        )
    molecules = pd.DataFrame(molecule_rows).sort_values("molecule_id").reset_index(drop=True)
    node_ids = molecules["molecule_id"].tolist()
    uf = UnionFind()
    for node in node_ids:
        uf.add(node)

    radius = int(config["analogue"]["ecfp_radius"])
    bits = int(config["analogue"]["ecfp_bits"])
    threshold = float(config["analogue"]["tanimoto_threshold"])
    ratio_min = float(config["analogue"]["molecular_weight_ratio_min"])
    ratio_max = float(config["analogue"]["molecular_weight_ratio_max"])
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits, includeChirality=True
    )
    fps = [generator.GetFingerprint(mol) for mol in molecules["_mol"]]
    weights = molecules["molecular_weight"].to_numpy()
    edge_types: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    edge_similarity: dict[tuple[str, str], float] = {}

    for left in range(len(node_ids) - 1):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[left], fps[left + 1 :])
        for offset, similarity in enumerate(similarities, start=left + 1):
            if similarity < threshold:
                continue
            ratio = weights[left] / weights[offset]
            if not (ratio_min <= ratio <= ratio_max):
                continue
            pair = (node_ids[left], node_ids[offset])
            uf.union(*pair)
            edge_types[pair].add("ecfp4")
            edge_similarity[pair] = float(similarity)

    buckets: dict[tuple[str, int, tuple[str, ...]], set[str]] = collections.defaultdict(set)
    for molecule_id, reps in sorted(representations.items()):
        for topology, sequence in sorted(reps):
            for position in range(len(sequence)):
                if topology.startswith("circle_head_to_tail"):
                    masked = masked_cycle(sequence, position)
                else:
                    local = list(sequence)
                    local[position] = "*"
                    masked = tuple(local)
                buckets[(topology, len(sequence), masked)].add(molecule_id)

    for (topology, _, _), members in sorted(buckets.items(), key=lambda item: repr(item[0])):
        for left, right in itertools.combinations(sorted(members), 2):
            matches = any(
                token_one_edit(seq_a, seq_b, topology_a, topology_b)
                for topology_a, seq_a in representations[left]
                for topology_b, seq_b in representations[right]
                if topology_a == topology_b == topology
            )
            if matches:
                pair = (left, right)
                uf.union(left, right)
                edge_types[pair].add("exact_one_token_edit")

    components: dict[str, list[str]] = collections.defaultdict(list)
    for node in node_ids:
        components[uf.find(node)].append(node)
    component_ids: dict[str, str] = {}
    for members in components.values():
        component_id = stable_id("AC", sorted(members))
        for member in members:
            component_ids[member] = component_id
    molecules["analogue_component_id"] = molecules["molecule_id"].map(component_ids)
    component_sizes = molecules["analogue_component_id"].value_counts().to_dict()
    molecules["analogue_component_size"] = molecules["analogue_component_id"].map(component_sizes)
    molecules = molecules.drop(columns=["_mol"])

    edge_rows = []
    for (left, right), kinds in sorted(edge_types.items()):
        edge_rows.append(
            {
                "molecule_id_a": left,
                "molecule_id_b": right,
                "edge_types": "|".join(sorted(kinds)),
                "ecfp4_tanimoto": edge_similarity.get((left, right), ""),
            }
        )
    edges = pd.DataFrame(
        edge_rows,
        columns=("molecule_id_a", "molecule_id_b", "edge_types", "ecfp4_tanimoto"),
    )
    return AnalogueResult(molecules, edges)


@dataclass
class BlockResult:
    rows: pd.DataFrame
    blocks: pd.DataFrame


def build_source_component_blocks(
    curated: pd.DataFrame, molecules: pd.DataFrame
) -> BlockResult:
    component_map = molecules.set_index("molecule_id")["analogue_component_id"].to_dict()
    rows = curated.copy()
    rows["analogue_component_id"] = rows["molecule_id"].map(component_map)
    uf = UnionFind()
    for _, row in rows.iterrows():
        source_node = "SOURCE::" + str(row["source"])
        component_node = "COMPONENT::" + str(row["analogue_component_id"])
        uf.union(source_node, component_node)
    roots = {}
    for _, row in rows.iterrows():
        source_node = "SOURCE::" + str(row["source"])
        roots[(str(row["source"]), str(row["analogue_component_id"]))] = uf.find(source_node)
    root_members: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in rows.iterrows():
        root = roots[(str(row["source"]), str(row["analogue_component_id"]))]
        root_members[root].append(index)
    block_ids: dict[str, str] = {}
    for root, indices in root_members.items():
        sources = sorted(set(rows.loc[indices, "source"].astype(str)))
        components = sorted(set(rows.loc[indices, "analogue_component_id"].astype(str)))
        block_ids[root] = stable_id("BLOCK", sources + components)
    rows["sealed_block_id"] = [
        block_ids[roots[(str(source), str(component))]]
        for source, component in zip(rows["source"], rows["analogue_component_id"])
    ]
    block_rows = []
    for block_id, group in rows.groupby("sealed_block_id", sort=True):
        block_rows.append(
            {
                "sealed_block_id": block_id,
                "n_curated_rows": int(len(group)),
                "n_unique_molecules": int(group["molecule_id"].nunique()),
                "n_analogue_components": int(group["analogue_component_id"].nunique()),
                "n_sources": int(group["source"].nunique()),
                "sources": "|".join(sorted(set(group["source"].astype(str)))),
                "topologies": "|".join(sorted(set(group["topology_signature"].astype(str)))),
                "ring_sizes": "|".join(
                    str(value) for value in sorted(set(group["ring_size"].astype(int)))
                ),
            }
        )
    blocks = pd.DataFrame(block_rows).sort_values("sealed_block_id").reset_index(drop=True)
    return BlockResult(rows, blocks)


def solve_allocation(
    blocks: pd.DataFrame, config: dict, enforce_minima: bool = True
) -> tuple[dict[str, str], dict[str, object]]:
    fractions = np.array(
        [float(config["partition"][partition]) for partition in PARTITIONS], dtype=float
    )
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Partition fractions must sum to one")
    n_blocks = len(blocks)
    n_partitions = len(PARTITIONS)
    n_x = n_blocks * n_partitions
    metrics = ("n_curated_rows", "n_analogue_components", "n_sources")
    n_dev = len(metrics) * n_partitions * 2
    n_variables = n_x + n_dev
    objective = np.zeros(n_variables)
    total_by_metric = {metric: float(blocks[metric].sum()) for metric in metrics}
    scales = {
        "n_curated_rows": max(total_by_metric["n_curated_rows"], 1.0),
        "n_analogue_components": max(total_by_metric["n_analogue_components"], 1.0),
        "n_sources": max(total_by_metric["n_sources"], 1.0),
    }
    metric_weights = {
        "n_curated_rows": 1.0,
        "n_analogue_components": 0.25,
        "n_sources": 0.10,
    }
    dev_index: dict[tuple[str, int, str], int] = {}
    cursor = n_x
    for metric in metrics:
        for partition_index in range(n_partitions):
            for sign in ("plus", "minus"):
                dev_index[(metric, partition_index, sign)] = cursor
                objective[cursor] = metric_weights[metric] / scales[metric]
                cursor += 1
    for block_index in range(n_blocks):
        for partition_index in range(n_partitions):
            objective[block_index * n_partitions + partition_index] = (
                1e-9 * (block_index + 1) * (partition_index + 1)
            )

    lower = np.zeros(n_variables)
    upper = np.full(n_variables, np.inf)
    upper[:n_x] = 1.0
    integrality = np.zeros(n_variables)
    integrality[:n_x] = 1
    constraints_rows: list[np.ndarray] = []
    constraints_lb: list[float] = []
    constraints_ub: list[float] = []

    for block_index in range(n_blocks):
        row = np.zeros(n_variables)
        start = block_index * n_partitions
        row[start : start + n_partitions] = 1.0
        constraints_rows.append(row)
        constraints_lb.append(1.0)
        constraints_ub.append(1.0)

    for metric in metrics:
        values = blocks[metric].to_numpy(dtype=float)
        for partition_index in range(n_partitions):
            row = np.zeros(n_variables)
            for block_index, value in enumerate(values):
                row[block_index * n_partitions + partition_index] = value
            row[dev_index[(metric, partition_index, "plus")]] = -1.0
            row[dev_index[(metric, partition_index, "minus")]] = 1.0
            target = fractions[partition_index] * total_by_metric[metric]
            constraints_rows.append(row)
            constraints_lb.append(target)
            constraints_ub.append(target)

    if enforce_minima:
        minima = config["partition"]["minima"]
        component_minima = (
            int(minima["train_analogue_components"]),
            int(minima["validation_analogue_components"]),
            int(minima["calibration_analogue_components"]),
            int(minima["final_test_analogue_components"]),
        )
        component_values = blocks["n_analogue_components"].to_numpy(dtype=float)
        source_values = blocks["n_sources"].to_numpy(dtype=float)
        for partition_index, minimum in enumerate(component_minima):
            row = np.zeros(n_variables)
            for block_index, value in enumerate(component_values):
                row[block_index * n_partitions + partition_index] = value
            constraints_rows.append(row)
            constraints_lb.append(float(minimum))
            constraints_ub.append(np.inf)
        outside_train = np.zeros(n_variables)
        final_sources = np.zeros(n_variables)
        for block_index, value in enumerate(source_values):
            for partition_index in range(1, n_partitions):
                outside_train[block_index * n_partitions + partition_index] = value
            final_sources[block_index * n_partitions + 3] = value
        constraints_rows.extend((outside_train, final_sources))
        constraints_lb.extend(
            (
                float(minima["sources_outside_train"]),
                float(minima["final_test_sources"]),
            )
        )
        constraints_ub.extend((np.inf, np.inf))

    matrix = np.vstack(constraints_rows)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, constraints_lb, constraints_ub),
        options={"presolve": True, "time_limit": 300},
    )
    metadata = {
        "enforced_minima": bool(enforce_minima),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": float(result.fun) if result.fun is not None else None,
    }
    if result.x is None:
        return {}, metadata
    assignment: dict[str, str] = {}
    choices = result.x[:n_x].reshape(n_blocks, n_partitions)
    for block_index, block_id in enumerate(blocks["sealed_block_id"].astype(str)):
        assignment[block_id] = PARTITIONS[int(np.argmax(choices[block_index]))]
    return assignment, metadata


def gate_summary(rows: pd.DataFrame, config: dict) -> dict[str, object]:
    counts = {}
    for partition in PARTITIONS:
        subset = rows.loc[rows["partition"] == partition]
        counts[partition] = {
            "rows": int(len(subset)),
            "molecules": int(subset["molecule_id"].nunique()),
            "analogue_components": int(subset["analogue_component_id"].nunique()),
            "sources": int(subset["source"].nunique()),
        }
    minima = config["partition"]["minima"]
    checks = {
        "train_analogue_components": counts["train"]["analogue_components"]
        >= int(minima["train_analogue_components"]),
        "validation_analogue_components": counts["validation"]["analogue_components"]
        >= int(minima["validation_analogue_components"]),
        "calibration_analogue_components": counts["calibration"]["analogue_components"]
        >= int(minima["calibration_analogue_components"]),
        "final_test_analogue_components": counts["final_test"]["analogue_components"]
        >= int(minima["final_test_analogue_components"]),
        "sources_outside_train": int(
            rows.loc[rows["partition"] != "train", "source"].nunique()
        )
        >= int(minima["sources_outside_train"]),
        "final_test_sources": counts["final_test"]["sources"]
        >= int(minima["final_test_sources"]),
    }
    source_partitions = rows.groupby("source")["partition"].nunique()
    component_partitions = rows.groupby("analogue_component_id")["partition"].nunique()
    block_partitions = rows.groupby("sealed_block_id")["partition"].nunique()
    leakage = {
        "sources_in_multiple_partitions": int((source_partitions > 1).sum()),
        "components_in_multiple_partitions": int((component_partitions > 1).sum()),
        "blocks_in_multiple_partitions": int((block_partitions > 1).sum()),
    }
    return {
        "counts": counts,
        "checks": checks,
        "leakage": leakage,
        "admissible": bool(all(checks.values()) and all(value == 0 for value in leakage.values())),
    }


def build_pcppred_ambiguity(
    train_path: Path, test_path: Path, raw: pd.DataFrame
) -> pd.DataFrame:
    raw_work = raw.copy()
    raw_mols = [Chem.MolFromSmiles(clean_scalar(value)) for value in raw_work["SMILES"]]
    raw_work["_canonical_smiles"] = [
        Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""
        for mol in raw_mols
    ]
    raw_work["_raw_id_text"] = raw_work["ID"].map(clean_scalar)
    rows = []
    for split, path in (("train", train_path), ("test", test_path)):
        frame = pd.read_csv(path, low_memory=False)
        for _, row in frame.iterrows():
            pcp_id = clean_scalar(row.get("ID"))
            mol = Chem.MolFromSmiles(clean_scalar(row.get("SMILES")))
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""
            by_id = raw_work.loc[raw_work["_raw_id_text"] == pcp_id]
            by_structure = raw_work.loc[raw_work["_canonical_smiles"] == smiles]
            direct_match = bool(
                len(by_id)
                and smiles
                and (by_id["_canonical_smiles"] == smiles).any()
            )
            if not len(by_id):
                reason = "pcppred_id_absent_from_raw_id_namespace"
            elif not direct_match:
                reason = "pcppred_id_maps_to_different_raw_structure"
            elif len(by_structure) > 1:
                reason = "pcppred_row_aggregates_or_selects_among_multiple_raw_records"
            else:
                reason = "direct_single_raw_record_match"
            rows.append(
                {
                    "pcppred_split": split,
                    "pcppred_id": pcp_id,
                    "canonical_smiles": smiles,
                    "direct_raw_id_structure_match": direct_match,
                    "raw_ids_for_same_structure": "|".join(
                        sorted(set(by_structure["_raw_id_text"].astype(str)))
                    ),
                    "raw_sources_for_same_structure": "|".join(
                        sorted(set(by_structure["Source"].astype(str)))
                    ),
                    "ambiguity_reason": reason,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["pcppred_split", "pcppred_id"], kind="stable"
    )


def assert_public_label_free(output_dir: Path) -> None:
    for filename in PUBLIC_MANIFESTS:
        path = output_dir / filename
        actual = tuple(pd.read_csv(path, nrows=0).columns)
        expected = PUBLIC_SCHEMA_ALLOWLISTS[filename]
        if actual != expected:
            raise RuntimeError(
                f"Public manifest schema differs from explicit allowlist: "
                f"{filename}: actual={actual!r}, expected={expected!r}"
            )
        columns = {column.strip().lower() for column in actual}
        overlap = OUTCOME_DERIVED_COLUMNS & columns
        if overlap:
            raise RuntimeError(
                f"Public manifest exposes outcome-derived fields: "
                f"{filename}: {sorted(overlap)}"
            )
        fragment_hits = sorted(
            column
            for column in columns
            if any(fragment in column for fragment in OUTCOME_NAME_FRAGMENTS)
        )
        if fragment_hits:
            raise RuntimeError(
                f"Public manifest column names indicate outcome-derived content: "
                f"{filename}: {fragment_hits}"
            )


def make_curated_public(curated: pd.DataFrame) -> pd.DataFrame:
    """Project the internal curated table onto the explicit public allowlist."""
    missing = [column for column in CURATED_PUBLIC_COLUMNS if column not in curated.columns]
    if missing:
        raise RuntimeError(f"Internal curated table lacks public fields: {missing}")
    public = curated.loc[:, list(CURATED_PUBLIC_COLUMNS)].copy()
    if OUTCOME_DERIVED_COLUMNS.intersection(
        {str(column).strip().lower() for column in public.columns}
    ):
        raise RuntimeError("Outcome-derived fields survived the public projection")
    return public


def render_report(
    summary: dict[str, object],
    config: dict,
    vault_path: Path,
    vault_hash: str,
) -> str:
    gate = summary["gate"]
    allocation = summary["allocation"]
    status = "PASS" if gate["admissible"] else "FAIL — MODELING MUST STOP"
    lines = [
        "# ScaffoldSeal-CP Milestone-0 build report",
        "",
        f"**Built:** {summary['built_at_utc']}  ",
        f"**Data gate:** **{status}**  ",
        "**Model runs:** none.",
        "",
        "## Frozen implementation",
        "",
        "- Curated directly from raw source + canonical stereochemical structure groups.",
        "- Preserved every raw ID and source link in curation_manifest.csv.",
        "- Median-collapsed only compatible, uncensored same-source replicates.",
        "- Excluded invalid, irrecoverably censored, or incompatible groups without using outcomes for split allocation.",
        "- Built the union graph from chiral ECFP4 radius 2 Tanimoto ≥ 0.80 with MW ratio 0.80–1.25 and exact one-token cyclic edits with identical topology/length.",
        "- Built indivisible connected blocks of the source–analogue-component bipartite graph.",
        "- Allocated blocks label-blind toward 65/10/10/15 by deterministic mixed-integer optimization.",
        "",
        "## Curation flow",
        "",
    ]
    for key, value in summary["curation_flow"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Partition and gate checks",
            "",
            "| Partition | Rows | Molecules | Analogue components | Sources |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for partition in PARTITIONS:
        counts = gate["counts"][partition]
        lines.append(
            f"| {partition} | {counts['rows']} | {counts['molecules']} | "
            f"{counts['analogue_components']} | {counts['sources']} |"
        )
    lines.extend(["", "Frozen minima:"])
    for key, passed in gate["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] {key}")
    lines.extend(
        [
            "",
            "Leakage assertions:",
            f"- sources in multiple partitions: {gate['leakage']['sources_in_multiple_partitions']}",
            f"- analogue components in multiple partitions: {gate['leakage']['components_in_multiple_partitions']}",
            f"- bipartite blocks in multiple partitions: {gate['leakage']['blocks_in_multiple_partitions']}",
            "",
            "## Allocation",
            "",
            f"- Total analogue components available: {summary['analogue_components']}",
            f"- Sum of frozen component minima: {summary['minimum_component_total_required']}",
            f"- Feasible-with-minima solution found: {allocation['feasible_with_minima']}",
            f"- Constrained solver status: {allocation['constrained_solver']['solver_status']}",
            f"- Constrained solver message: {allocation['constrained_solver']['solver_message']}",
            f"- Applied solver enforced minima: {allocation['applied_solver']['enforced_minima']}",
            f"- Applied solver message: {allocation['applied_solver']['solver_message']}",
            f"- Number of indivisible blocks: {summary['n_blocks']}",
            f"- Largest block rows: {summary['largest_block_rows']}",
            f"- Largest block source count: {summary['largest_block_sources']}",
            "",
            "## PCPpred aggregate-ID ambiguity",
            "",
            "PCPpred Train/Test IDs are audited only as a comparison namespace. "
            "The authoritative curated table is rebuilt from raw source records. "
            "pcppred_id_ambiguity.csv records direct matches, aggregate/selection ambiguity, "
            "and IDs that do not map cleanly back to a raw record. No PCPpred split labels were "
            "used in curation or allocation.",
            "",
            "## Sealing and access",
            "",
            f"- External vault: {vault_path}",
            f"- Vault SHA-256: {vault_hash}",
            "- TEST_ACCESS_LOG.tsv was created after the vault and contains its header only.",
            "- split_manifest_public.csv and all other public manifests contain no final labels.",
            "- development_labeled.csv contains labels only for train, validation, and calibration.",
            "",
            "## Integrity decision",
            "",
        ]
    )
    if gate["admissible"]:
        lines.append(
            "The frozen grouped-unit and source-diversity minima pass. Milestone 0 is "
            "eligible for independent verification before any model-development run."
        )
    else:
        failed = [key for key, passed in gate["checks"].items() if not passed]
        lines.append(
            "The preregistered data gate fails. Modeling must stop; thresholds, analogue "
            "rules, or partition minima must not be weakened after this result. Failed checks: "
            + ", ".join(failed)
            + "."
        )
    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: {platform.python_version()}",
            f"- pandas: {pd.__version__}",
            f"- NumPy: {np.__version__}",
            f"- RDKit: {rdBase.rdkitVersion}",
            f"- SciPy: {scipy.__version__}",
            f"- Config version: {config['version']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    repo_root = config_path.parent
    if (repo_root / "FINAL_TEST_RETIRED.md").exists():
        raise RuntimeError(
            "This manifest generation is permanently retired after a final-outcome "
            "exposure incident. Start any authorized replacement as a new generation; "
            "do not overwrite the retired split or vault."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_candidates = sorted(repo_root.glob(config["input"]["raw_pampa_glob"]))
    if len(raw_candidates) != 1:
        raise RuntimeError(f"Expected one raw PAMPA file, found {len(raw_candidates)}")
    raw_path = raw_candidates[0].resolve()
    train_path = (repo_root / config["input"]["pcppred_train"]).resolve()
    test_path = (repo_root / config["input"]["pcppred_test"]).resolve()
    output_dir = (repo_root / config["paths"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_path, low_memory=False)

    curation = curate(raw)
    analogue = build_analogue_graph(curation.curated, config)
    block_result = build_source_component_blocks(curation.curated, analogue.molecules)
    assignment, constrained_solver = solve_allocation(
        block_result.blocks, config, enforce_minima=True
    )
    feasible_with_minima = bool(assignment)
    if not assignment:
        assignment, applied_solver = solve_allocation(
            block_result.blocks, config, enforce_minima=False
        )
    else:
        applied_solver = constrained_solver
    if not assignment:
        raise RuntimeError("Allocation solver failed even without frozen minima")
    rows = block_result.rows.copy()
    rows["partition"] = rows["sealed_block_id"].map(assignment)
    if rows["partition"].isna().any():
        raise RuntimeError("Unassigned sealed blocks")
    gate = gate_summary(rows, config)

    checksums: dict[str, str] = {}
    raw_hash = sha256_file(raw_path)
    raw_manifest = f"{raw_hash}  {raw_path.as_posix()}\n"
    raw_manifest_path = output_dir / "data_manifest_raw.sha256"
    raw_manifest_path.write_text(raw_manifest, encoding="utf-8", newline="\n")
    checksums["artifacts/data_manifest_raw.sha256"] = sha256_file(raw_manifest_path)

    curated_public = make_curated_public(curation.curated)
    molecule_public = analogue.molecules.copy()
    component_sources = (
        rows.groupby("analogue_component_id")["source"]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .to_dict()
    )
    molecule_public["component_sources"] = molecule_public["analogue_component_id"].map(
        component_sources
    )
    public_split = rows[
        [
            "curated_id",
            "molecule_id",
            "canonical_smiles",
            "source",
            "year",
            "version",
            "topology_signature",
            "ring_size",
            "analogue_component_id",
            "sealed_block_id",
            "partition",
            "raw_ids_all",
            "raw_ids_used",
        ]
    ].copy()
    block_public = block_result.blocks.copy()
    block_public["partition"] = block_public["sealed_block_id"].map(assignment)
    pcppred_ambiguity = build_pcppred_ambiguity(train_path, test_path, raw)
    outputs = {
        "curation_manifest.csv": curation.curation_manifest,
        "curated_records_public.csv": curated_public,
        "analogue_edges.csv": analogue.edges,
        "analogue_components.csv": molecule_public,
        "source_component_blocks.csv": block_public,
        "split_manifest_public.csv": public_split,
        "pcppred_id_ambiguity.csv": pcppred_ambiguity,
    }
    for filename, frame in outputs.items():
        checksums[f"artifacts/{filename}"] = write_csv(frame, output_dir / filename)

    development = rows.loc[rows["partition"] != "final_test"].copy()
    development_columns = [
        "curated_id",
        "molecule_id",
        "canonical_smiles",
        "source",
        "year",
        "version",
        "topology_signature",
        "canonical_sequence",
        "ring_size",
        "main_chain_length",
        "analogue_component_id",
        "sealed_block_id",
        "partition",
        "permeability",
    ]
    checksums["artifacts/development_labeled.csv"] = write_csv(
        development[development_columns], output_dir / "development_labeled.csv"
    )
    assert_public_label_free(output_dir)

    vault_path = Path(config["paths"]["sealed_vault"])
    vault = rows.loc[rows["partition"] == "final_test", ["curated_id", "permeability"]]
    vault_hash = write_csv(vault, vault_path)

    # Governance invariant: initialize the zero-entry log only after vault creation.
    access_log_path = repo_root / config["paths"]["test_access_log"]
    access_log_payload = ("\t".join(ACCESS_LOG_HEADER) + "\n").encode("utf-8")
    access_log_path.write_bytes(access_log_payload)
    checksums[config["paths"]["test_access_log"]] = sha256_bytes(access_log_payload)

    ambiguity_counts = (
        pcppred_ambiguity["ambiguity_reason"].value_counts().sort_index().to_dict()
    )
    largest = block_result.blocks.sort_values(
        ["n_curated_rows", "sealed_block_id"], ascending=[False, True]
    ).iloc[0]
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": raw_path.as_posix(),
        "raw_sha256": raw_hash,
        "curation_flow": curation.flow,
        "analogue_edges": int(len(analogue.edges)),
        "analogue_components": int(
            analogue.molecules["analogue_component_id"].nunique()
        ),
        "minimum_component_total_required": int(
            config["partition"]["minima"]["train_analogue_components"]
            + config["partition"]["minima"]["validation_analogue_components"]
            + config["partition"]["minima"]["calibration_analogue_components"]
            + config["partition"]["minima"]["final_test_analogue_components"]
        ),
        "n_blocks": int(len(block_result.blocks)),
        "largest_block_rows": int(largest["n_curated_rows"]),
        "largest_block_sources": int(largest["n_sources"]),
        "allocation": {
            "feasible_with_minima": feasible_with_minima,
            "constrained_solver": constrained_solver,
            "applied_solver": applied_solver,
        },
        "gate": gate,
        "pcppred_ambiguity_counts": {
            str(key): int(value) for key, value in ambiguity_counts.items()
        },
        "vault_path": vault_path.as_posix(),
        "vault_sha256": vault_hash,
        "vault_rows": int(len(vault)),
        "test_access_log_entries": 0,
    }
    summary_path = output_dir / "build_summary.json"
    summary_payload = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    summary_path.write_bytes(summary_payload)
    checksums["artifacts/build_summary.json"] = sha256_bytes(summary_payload)

    report_path = repo_root / "MILESTONE0_BUILD_REPORT.md"
    report_payload = render_report(summary, config, vault_path, vault_hash).encode("utf-8")
    report_path.write_bytes(report_payload)
    checksums["MILESTONE0_BUILD_REPORT.md"] = sha256_bytes(report_payload)
    checksums["config.yaml"] = sha256_file(config_path)
    for code_path in sorted((repo_root / "src").glob("*.py")):
        checksums[code_path.relative_to(repo_root).as_posix()] = sha256_file(code_path)
    for test_code in sorted((repo_root / "tests").glob("*.py")):
        checksums[test_code.relative_to(repo_root).as_posix()] = sha256_file(test_code)
    checksums[f"external:{vault_path.as_posix()}"] = vault_hash
    checksum_payload = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
    ).encode("utf-8")
    (repo_root / "SHA256SUMS").write_bytes(checksum_payload)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if gate["admissible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
