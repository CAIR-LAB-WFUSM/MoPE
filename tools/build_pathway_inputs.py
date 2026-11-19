import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-patient Hallmark pathway matrices (pathways x genes) from an RNA-seq table."
    )
    parser.add_argument(
        "--rna_csv",
        required=True,
        help="Comma-separated table: first column TCGA sample barcode (TCGA-XX-XXXX-01), "
        "remaining columns gene symbols with non-log expression values.",
    )
    parser.add_argument("--gmt", required=True, help="MSigDB Hallmark gene sets (.gmt).")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--include_non_primary",
        action="store_true",
        help="Keep all sample types instead of primary tumors (sample code 01) only.",
    )
    return parser.parse_args()


def load_pathways(gmt_path):
    pathways = []
    for line in open(gmt_path):
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        genes = {g.strip().upper() for g in parts[2:] if g.strip()}
        pathways.append((parts[0].strip(), genes))
    if not pathways:
        raise ValueError("No valid pathways in the GMT file.")
    return pathways


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return None


def patient_id(fields, primary_only):
    if len(fields) < 2:
        return None
    parts = fields[0].strip().split("-")
    if len(parts) < 4 or parts[0].upper() != "TCGA":
        return None
    if primary_only and parts[3].upper() != "01":
        return None
    return "-".join(parts[:3])


def iter_patients(csv_path, primary_only):
    seen = set()
    with open(csv_path) as handle:
        handle.readline()
        for line in handle:
            fields = line.rstrip("\r\n").split(",")
            pid = patient_id(fields, primary_only)
            if pid is None or pid.upper() in seen:
                continue
            seen.add(pid.upper())
            yield pid, fields


def gene_statistics(csv_path, columns, primary_only):
    count = np.zeros(len(columns))
    mean = np.zeros(len(columns))
    m2 = np.zeros(len(columns))
    patients = 0
    for _, fields in iter_patients(csv_path, primary_only):
        patients += 1
        for i, col in enumerate(columns):
            value = parse_float(fields[col])
            if value is None:
                continue
            x = np.log2(value + 1.0)
            count[i] += 1
            delta = x - mean[i]
            mean[i] += delta / count[i]
            m2[i] += delta * (x - mean[i])
    std = np.where(count > 1, np.sqrt(m2 / np.maximum(count - 1, 1)), 0.0)
    return mean, std, patients


def main():
    args = parse_args()
    primary_only = not args.include_non_primary
    pathways = load_pathways(args.gmt)
    union = set().union(*(genes for _, genes in pathways))

    with open(args.rna_csv) as handle:
        header = handle.readline().rstrip("\r\n").split(",")
    columns = [i for i in range(1, len(header)) if header[i].upper() in union]
    genes = [header[i] for i in columns]
    members = [
        np.array([j for j, g in enumerate(genes) if g.upper() in gene_set], dtype=int) for _, gene_set in pathways
    ]

    mean, std, patients = gene_statistics(args.rna_csv, columns, primary_only)
    print(f"Pathways: {len(pathways)}, genes: {len(genes)}, patients: {patients}")

    matrix_dir = os.path.join(args.out_dir, "patient_matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pathways.txt"), "w") as handle:
        handle.write("\n".join(name for name, _ in pathways) + "\n")
    with open(os.path.join(args.out_dir, "genes.txt"), "w") as handle:
        handle.write("\n".join(genes) + "\n")
    mask = np.zeros((len(pathways), len(genes)), dtype=np.float32)
    for k, idx in enumerate(members):
        mask[k, idx] = 1.0
    np.save(os.path.join(args.out_dir, "membership_mask.npy"), mask)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as handle:
        handle.write(f"primary_tumor_only={primary_only}\nnum_pathways={len(pathways)}\nnum_genes={len(genes)}\n")
        handle.write("".join(f"{name}\t{len(idx)}\n" for (name, _), idx in zip(pathways, members)))

    written = 0
    for pid, fields in iter_patients(args.rna_csv, primary_only):
        z = np.zeros(len(genes), dtype=np.float32)
        for i, col in enumerate(columns):
            value = parse_float(fields[col])
            if value is None or not std[i] > 0:
                continue
            z[i] = np.float32((np.log2(value + 1.0) - mean[i]) / std[i])
        matrix = np.zeros((len(pathways), len(genes)), dtype=np.float32)
        for k, idx in enumerate(members):
            matrix[k, idx] = z[idx]
        np.save(os.path.join(matrix_dir, f"{pid}.npy"), matrix)
        written += 1
    print(f"Wrote {written} patient matrices to {matrix_dir}")


if __name__ == "__main__":
    main()
