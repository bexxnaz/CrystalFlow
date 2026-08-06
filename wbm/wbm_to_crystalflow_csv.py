"""Convert WBM relaxed (ground-truth) structures into a CrystalFlow-style CSV.

Uses matbench_discovery's DataFiles.wbm_relaxed_atoms -- the DFT-relaxed
structures -- as ground truth for evaluating CrystalFlow's generated
structures via match rate / RMSD, the same way the CrystalFlow paper
evaluates CSP performance.

Assumes wbm_relaxed_atoms is already cached locally and verified as a valid
zip (ase_atoms_from_zip should load without triggering any download).
"""

import random

import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from matbench_discovery.data import DataFiles, ase_atoms_from_zip, df_wbm

# --- config ------------------------------------------------------------
OUT_CSV = "wbm_test_crystalflow.csv"
UNIQ_PROTO_ONLY = True   # keep only the ~215k deduplicated, harder subset
N_SAMPLES = None         # set to None to convert the full set

# --- load all relaxed structures ----------------------------------------
print("Loading WBM relaxed atoms (this reads ~257k structures, may take a bit)...")
atoms_list = ase_atoms_from_zip(DataFiles.wbm_relaxed_atoms.path)
print(f"loaded {len(atoms_list):,} Atoms objects")

atoms_by_id = {}
for atoms in atoms_list:
    mat_id = atoms.info["material_id"]
    if mat_id in atoms_by_id:
        raise ValueError(f"Duplicate material_id: {mat_id}")
    atoms_by_id[mat_id] = atoms

# sanity check against the summary table
missing = set(df_wbm.index) - set(atoms_by_id)
extra = set(atoms_by_id) - set(df_wbm.index)
print(f"coverage check: {len(missing)} missing, {len(extra)} extra vs df_wbm")

# --- select subset --------------------------------------------------------
material_ids = list(atoms_by_id)

if UNIQ_PROTO_ONLY:
    uniq_ids = set(df_wbm.query("unique_prototype").index)
    material_ids = [mid for mid in material_ids if mid in uniq_ids]
    print(f"restricted to unique-prototype subset: {len(material_ids):,} structures")

if N_SAMPLES is not None and N_SAMPLES < len(material_ids):
    random.seed(0)
    material_ids = random.sample(material_ids, N_SAMPLES)
    print(f"subsampled to {len(material_ids):,} structures")

# --- convert to CIF and write CSV -----------------------------------------
adaptor = AseAtomsAdaptor()
rows = []
for mat_id in tqdm(material_ids, desc="Converting to CIF"):
    structure = adaptor.get_structure(atoms_by_id[mat_id])
    cif_str = structure.to(fmt="cif")
    rows.append({"material_id": mat_id, "cif": cif_str})

df_out = pd.DataFrame(rows)

# --- attach number of atoms and formation energy from df_wbm ---------------
df_out["num_atoms"] = df_out["material_id"].map(df_wbm["n_sites"])
df_out["formation_energy_per_atom"] = df_out["material_id"].map(
    df_wbm["e_form_per_atom_mp2020_corrected"]
)

df_out.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df_out):,} rows to {OUT_CSV}")
print(df_out.head(2))