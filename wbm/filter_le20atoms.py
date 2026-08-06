
import pandas as pd

MAX_ATOMS = 20

pairs = [
    ("wbm_test_crystalflow.csv", "wbm_test_crystalflow_le20atoms.csv"),
    ("wbm_initial_crystalflow.csv", "wbm_initial_crystalflow_le20atoms.csv"),
]

for in_path, out_path in pairs:
    df = pd.read_csv(in_path)
    n_before = len(df)

    df_filtered = df.query("num_atoms <= @MAX_ATOMS").reset_index(drop=True)
    n_after = len(df_filtered)

    df_filtered.to_csv(out_path, index=False)
    print(
        f"{in_path}: kept {n_after:,} / {n_before:,} rows "
        f"(num_atoms <= {MAX_ATOMS}) -> {out_path}"
    )
