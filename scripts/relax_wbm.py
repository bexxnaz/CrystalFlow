import time
import argparse
import torch
from pathlib import Path
from torch_geometric.data import Batch
import hydra
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from eval_utils import load_model, lattices_to_params_shape
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

import numpy as np
from diffcsp.common.data_utils import (
    lattice_params_to_matrix_torch,
    lattice_polar_build_torch,
)

import pandas as pd
import sys

def get_material_ids_for_loader(loader, csv_path):
    df = pd.read_csv(csv_path)
    if 'material_id' not in df.columns:
        print(f'WARNING: no material_id column in {csv_path} -- saving without it.')
        return None
    material_ids = df['material_id'].tolist()

    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        base_len = len(dataset.dataset)
        if len(material_ids) != base_len:
            raise ValueError(
                f"material_id count ({len(material_ids)}) != underlying dataset "
                f"length ({base_len}) -- CrystDataset preprocessing may have "
                f"dropped/reordered rows; cannot safely align material_id.")
        material_ids = [material_ids[i] for i in dataset.indices]
    else:
        if len(material_ids) != len(dataset):
            raise ValueError(
                f"material_id count ({len(material_ids)}) != dataset length "
                f"({len(dataset)}) -- cannot safely align material_id.")
    return material_ids


def subsample_loader(loader, n, seed=0):
    dataset = loader.dataset
    total = len(dataset)
    n = min(n, total)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(total, size=n, replace=False).tolist())
    subset = torch.utils.data.Subset(dataset, indices)
    new_loader = type(loader)(subset, batch_size=loader.batch_size)
    print(f'Subsampled test set: {n} / {total} structures (seed={seed})')
    return new_loader


def perturb_batch(batch, coord_noise, lattice_noise, device, model):
    frac_coords = batch.frac_coords.clone().to(device)
    frac_coords_distorted = (frac_coords + torch.randn_like(frac_coords) * coord_noise) % 1.0

    if model.lattice_polar:
        lattice_polar_gt = batch.lattice_polar.clone().to(device)
        lattice_polar_distorted = lattice_polar_gt + torch.randn_like(lattice_polar_gt) * lattice_noise
        lattices_mat_distorted = lattice_polar_build_torch(lattice_polar_distorted)
    else:
        lengths = batch.lengths.clone().to(device)
        angles = batch.angles.clone().to(device)
        lengths_distorted = lengths * (1.0 + torch.randn_like(lengths) * lattice_noise)
        angles_distorted = angles + torch.randn_like(angles) * lattice_noise * 10.0
        lattices_mat_distorted = lattice_params_to_matrix_torch(lengths_distorted, angles_distorted)

    return {
        'frac_coords': frac_coords_distorted,
        'lattices_mat': lattices_mat_distorted,
    }




def build_mace_calculator(mace_model, device, default_dtype="float32", dispersion=False):
    from pathlib import Path as _Path
    if _Path(mace_model).exists():
        from mace.calculators import MACECalculator
        print(f'Loading custom MACE checkpoint: {mace_model} (dtype={default_dtype})')
        return MACECalculator(model_paths=[mace_model], device=device,
                               default_dtype=default_dtype)
    else:
        from mace.calculators import mace_mp
        print(f'Loading MACE-MP foundation model (size={mace_model!r}, '
              f'dtype={default_dtype}, dispersion={dispersion}) -- '
              f'downloads weights on first use if not already cached.')
        return mace_mp(model=mace_model, device=device,
                        default_dtype=default_dtype, dispersion=dispersion)
 
 
def _build_ase_atoms(frac_coords, lattice_matrix, atomic_numbers):
    from ase import Atoms
    return Atoms(
        numbers=atomic_numbers.detach().cpu().numpy(),
        scaled_positions=frac_coords.detach().cpu().numpy(),
        cell=lattice_matrix.detach().cpu().numpy(),
        pbc=True,
    )
 
 
def compute_mace_energy_trajectory(frac_traj, lattices_traj, num_atoms, atom_types,
                                    mace_calc, stride=1):
    T = frac_traj.shape[0]
    n_graphs = num_atoms.shape[0]
    frac_per_graph = torch.split(frac_traj, num_atoms.tolist(), dim=1)  
    z_per_graph = torch.split(atom_types, num_atoms.tolist(), dim=0)     
 
    results = []
    for i in range(n_graphs):
        n_atoms_i = int(num_atoms[i])
        steps, e_tot, e_per_atom = [], [], []
        for t in range(0, T, stride):
            atoms = _build_ase_atoms(frac_per_graph[i][t], lattices_traj[t, i], z_per_graph[i])
            atoms.calc = mace_calc
            e = atoms.get_potential_energy()
            steps.append(t)
            e_tot.append(float(e))
            e_per_atom.append(float(e) / n_atoms_i)
        if steps[-1] != T - 1:  # always include the true final step
            atoms = _build_ase_atoms(frac_per_graph[i][T - 1], lattices_traj[T - 1, i], z_per_graph[i])
            atoms.calc = mace_calc
            e = atoms.get_potential_energy()
            steps.append(T - 1)
            e_tot.append(float(e))
            e_per_atom.append(float(e) / n_atoms_i)
        results.append({'steps': steps, 'energy_total_eV': e_tot, 'energy_per_atom_eV': e_per_atom})
 
    return results
 
 

def load_ground_truth_energies(gt_file, material_ids, mace_calc, num_atoms_ref=None):
    gt_df = pd.read_csv(gt_file)
    if 'material_id' not in gt_df.columns or 'cif' not in gt_df.columns:
        raise ValueError(f"--gt_file {gt_file} needs 'material_id' and 'cif' columns "
                          f"(same convention as compute_metrics.py's --gt_file).")
    cif_by_id = dict(zip(gt_df['material_id'], gt_df['cif']))

    n_atoms_out, e_tot_out, e_per_atom_out = [], [], []
    n_missing, n_failed, n_mismatch = 0, 0, 0
    for idx, mat_id in enumerate(material_ids):
        cif = cif_by_id.get(mat_id)
        if cif is None:
            n_missing += 1
            n_atoms_out.append(None); e_tot_out.append(float('nan')); e_per_atom_out.append(float('nan'))
            continue
        try:
            structure = Structure.from_str(cif, fmt='cif')
            n_atoms_gt = len(structure)

            if num_atoms_ref is not None and int(num_atoms_ref[idx]) != n_atoms_gt:
                n_mismatch += 1
                print(f'WARNING: gt_file atom count for {mat_id} ({n_atoms_gt}) != '
                      f'evaluated structure\'s atom count ({int(num_atoms_ref[idx])}) -- '
                      f'skipping, this looks like a mismatched gt_file or id collision, '
                      f'not the same material.')
                n_atoms_out.append(None); e_tot_out.append(float('nan')); e_per_atom_out.append(float('nan'))
                continue

            frac = torch.tensor(structure.frac_coords, dtype=torch.float32)
            lat = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
            z = torch.tensor([s.specie.Z for s in structure], dtype=torch.long)
            atoms = _build_ase_atoms(frac, lat, z)
            atoms.calc = mace_calc
            e = atoms.get_potential_energy()
            n_atoms_out.append(n_atoms_gt)
            e_tot_out.append(float(e))
            e_per_atom_out.append(float(e) / n_atoms_gt)
        except Exception as exc:
            n_failed += 1
            print(f'WARNING: failed to compute ground-truth MACE energy for {mat_id}: {exc}')
            n_atoms_out.append(None); e_tot_out.append(float('nan')); e_per_atom_out.append(float('nan'))

    n_ok = len(material_ids) - n_missing - n_failed - n_mismatch
    print(f'Ground-truth MACE energies: {n_ok}/{len(material_ids)} computed '
          f'({n_missing} not found in gt_file, {n_mismatch} atom-count mismatch, '
          f'{n_failed} failed to parse/evaluate).')
    return n_atoms_out, e_tot_out, e_per_atom_out

def save_initial_structure_cifs(outdir, mat_id, eval_idx, frac_coords, lattice_matrix, atom_types):
    try:
        struct = Structure(
            lattice=Lattice(lattice_matrix.detach().cpu().numpy()),
            species=atom_types.detach().cpu().numpy(),
            coords=frac_coords.detach().cpu().numpy(),
            coords_are_cartesian=False,
        )
    except Exception as exc:
        print(f'WARNING: could not write CIF for {mat_id} (eval {eval_idx}): {exc}')
        return False
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    struct.to(filename=str(outdir / f'{mat_id}_eval{eval_idx}.cif'), fmt='cif')
    return True


def relax(loader, model, num_evals, coord_noise, lattice_noise, null_baseline=False,
          perturb=False,material_ids=None, init_structure_from_batch=True,save_initial_cifs=False,initial_cifs_dir=None, outdir=None, diff_out_name=None,
        model_path=None,mace_calc=None, mace_stride=1, **sample_kwargs):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    n_steps_used = []
    coord_norm_traj_all, lattice_norm_traj_all = [], []
    mace_traj_all = []   
    input_data_list = []
    device = next(model.parameters()).device
    struct_offset = 0

    def _write_initial_cifs(frac0, lat0, atom_types_, num_atoms_, eval_idx):
        frac_per_graph = torch.split(frac0, num_atoms_.tolist(), dim=0)
        z_per_graph = torch.split(atom_types_, num_atoms_.tolist(), dim=0)
        for local_i in range(num_atoms_.shape[0]):
            global_i = struct_offset + local_i
            mat_id = material_ids[global_i] if material_ids is not None else f'graph_{global_i}'
            save_initial_structure_cifs(initial_cifs_dir, mat_id, eval_idx,
                                         frac_per_graph[local_i], lat0[local_i],
                                         z_per_graph[local_i])
    

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_n_steps_used= []
        batch_lattices = []
        batch_coord_traj, batch_lattice_traj = [], []
        batch_mace = []  
    
        for eval_idx in range(num_evals):
            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            init_structure = None
            if perturb:
                init_structure = perturb_batch(
                    batch, coord_noise, lattice_noise, device, model)
            elif init_structure_from_batch:
                init_structure = {
                    'frac_coords': batch.frac_coords,
                    'lattices_mat': lattice_params_to_matrix_torch(batch.lengths, batch.angles),
                }
            if null_baseline:
                if init_structure is None:
                    raise ValueError(
                        "null_baseline needs an actual starting structure to work with")
                out_frac = init_structure['frac_coords'].detach().cpu()
                out_lattices = init_structure['lattices_mat'].detach().cpu()
                out_num_atoms = batch.num_atoms.detach().cpu()
                out_atom_types = batch.atom_types.detach().cpu()
                out_coord_traj = torch.zeros(0, batch.num_graphs)
                out_lattice_traj = torch.zeros(0, batch.num_graphs)
                out_mace = None   

                batch_size = batch.num_graphs
                nan_placeholder = torch.full((batch_size,), float('nan'))
                out_n_steps_used = nan_placeholder
      
            else:

                outputs, traj = model.sample(batch, init_structure=init_structure, **sample_kwargs)


                lengths, angles = lattices_to_params_shape(traj['all_lattices'])  
                T = traj['all_frac_coords'].shape[0]

                input_data = Batch.from_data_list(batch.to_data_list()).cpu()

                torch.save({
                    'input_data_batch': input_data,
                    'frac_coords': traj['all_frac_coords'].detach().cpu(),
                    'lengths':     lengths.detach().cpu(),
                    'angles':      angles.detach().cpu(),
                    'atom_types':  batch.atom_types.detach().cpu().unsqueeze(0).repeat(T, 1),
                    'num_atoms':   batch.num_atoms.detach().cpu().unsqueeze(0).repeat(T, 1),
                },  outdir / diff_out_name)
                
                if save_initial_cifs:
                    _write_initial_cifs(
                        traj['all_frac_coords'][0].detach().cpu(),
                        traj['all_lattices'][0].detach().cpu(),
                        batch.atom_types.detach().cpu(), batch.num_atoms.detach().cpu(),
                        eval_idx,
                    )

                print("traj['all_frac_coords'].shape           =", traj['all_frac_coords'].shape)
                print("traj['all_lattices'].shape            =", traj['all_lattices'].shape)


                print("input_data_batch           =", type(input_data))
                print("input_data_batch.frac_coords.shape =",
                    input_data.frac_coords.shape)
                print("input_data_batch.atom_types.shape  =",
                    input_data.atom_types.shape)
                print("input_data_batch.lengths.shape     =",
                    input_data.lengths.shape)
                print("input_data_batch.angles.shape      =",
                    input_data.angles.shape)
                print("input_data_batch.num_atoms.shape   =",
                    input_data.num_atoms.shape)


                out_mace = None
                if mace_calc is not None:
                    out_mace = compute_mace_energy_trajectory(
                        traj['all_frac_coords'], traj['all_lattices'],
                        batch.num_atoms.detach().cpu(), batch.atom_types.detach().cpu(),
                        mace_calc, stride=mace_stride,
                    )

                out_frac = outputs['frac_coords'].detach().cpu()
                out_lattices = outputs['lattices'].detach().cpu()
                out_num_atoms = outputs['num_atoms'].detach().cpu()
                out_atom_types = outputs['atom_types'].detach().cpu()
                out_coord_traj = traj['coord_field_norm_traj']
                out_lattice_traj = traj['lattice_field_norm_traj']
                out_n_steps_used = traj['n_steps_used']


            batch_frac_coords.append(out_frac)
            batch_num_atoms.append(out_num_atoms)
            batch_atom_types.append(out_atom_types)
            batch_lattices.append(out_lattices)
            batch_n_steps_used.append(out_n_steps_used)
            batch_coord_traj.append(out_coord_traj)
            batch_lattice_traj.append(out_lattice_traj)
            batch_mace.append(out_mace)   
   

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))
        n_steps_used.append(torch.stack(batch_n_steps_used, dim=0))
        coord_norm_traj_all.append(batch_coord_traj)
        lattice_norm_traj_all.append(batch_lattice_traj)
        mace_traj_all.append(batch_mace)   

        input_data_list = input_data_list + batch.to_data_list()
        struct_offset += batch.num_graphs

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)
    n_steps_used = torch.cat(n_steps_used, dim=1)

    print("mace_traj_all:", type(mace_traj_all), len(mace_traj_all))
    print("mace_traj_all[0]:", type(mace_traj_all[0]), len(mace_traj_all[0]))
    print("mace_traj_all[0][0].shape:",type(mace_traj_all[0][0]),  len(mace_traj_all[0][0]))


    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
        ,n_steps_used, 
        coord_norm_traj_all, lattice_norm_traj_all,
        mace_traj_all
    )


def save_field_norm_csv(
    csv_path, material_ids, num_atoms, n_steps_used,
    coord_norm_traj_all, lattice_norm_traj_all,
):

    num_atoms_np = num_atoms.numpy()
    n_steps_np = n_steps_used.numpy()

    n_structs = num_atoms_np.shape[1]
    if material_ids is not None and len(material_ids) != n_structs:
        print(f'WARNING: material_id count ({len(material_ids)}) != number of '
              f'evaluated structures ({n_structs}) -- writing CSV without '
              f'material_id (order/count mismatch, do not trust an unaligned join).')
        material_ids = None

    rows = []
    struct_offset = 0
    for batch_coord_traj, batch_lattice_traj in zip(coord_norm_traj_all, lattice_norm_traj_all):
        batch_size = batch_coord_traj[0].shape[1] if batch_coord_traj[0].numel() > 0 else \
                     batch_lattice_traj[0].shape[1]

        for e, (coord_traj, lattice_traj) in enumerate(zip(batch_coord_traj, batch_lattice_traj)):
            n_steps_this = coord_traj.shape[0]
            for local_i in range(batch_size):
                global_i = struct_offset + local_i
                mat_id = material_ids[global_i] if material_ids is not None else None
                n_atoms_val = int(num_atoms_np[e, global_i])
                n_steps_used_val = float(n_steps_np[e, global_i])

                if n_steps_this == 0:
                    rows.append({
                        'material_id': mat_id, 'eval_idx': e, 'step': None,
                        'n_atoms': n_atoms_val, 'n_steps_used': n_steps_used_val,
                        'coord_field_norm': float('nan'),
                        'lattice_field_norm': float('nan'),
                    })
                else:
                    for step in range(n_steps_this):
                        if step > n_steps_used_val:
                            break
                        rows.append({
                            'material_id': mat_id, 'eval_idx': e, 'step': step + 1,
                            'n_atoms': n_atoms_val, 'n_steps_used': n_steps_used_val,
                            'coord_field_norm': float(coord_traj[step, local_i]),
                            'lattice_field_norm': float(lattice_traj[step, local_i]),
                        })
        struct_offset += batch_size

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f'Saved per-step field-norm trajectory ({len(df)} rows, '
          f'{df["material_id"].nunique() if material_ids is not None else n_structs} structures) '
          f'to {csv_path}')
    return df



def save_mace_energy_csv(csv_path, material_ids, num_atoms, mace_traj_all):
    num_atoms_np = num_atoms.numpy()
    n_structs = num_atoms_np.shape[1]

    if material_ids is not None and len(material_ids) != n_structs:
        print(f'WARNING: material_id count ({len(material_ids)}) != number of '
              f'evaluated structures ({n_structs}) -- writing CSV without material_id.')
        material_ids = None

    rows = []
    struct_offset = 0
    for batch_mace in mace_traj_all:
        batch_size = None
        for m in batch_mace:
            if m is not None:
                batch_size = len(m)
                break
        if batch_size is None:
            continue  # every eval in this batch had no MACE data

        for e, m in enumerate(batch_mace):
            if m is None:
                continue
            for local_i, per_graph in enumerate(m):
                global_i = struct_offset + local_i
                mat_id = material_ids[global_i] if material_ids is not None else None
                n_atoms_val = int(num_atoms_np[e, global_i])
                for step, e_tot, e_pa in zip(per_graph['steps'], per_graph['energy_total_eV'],
                                              per_graph['energy_per_atom_eV']):
                    rows.append({
                        'material_id': mat_id, 'eval_idx': e, 'step': step,
                        'n_atoms': n_atoms_val,
                        'energy_total_eV': e_tot, 'energy_per_atom_eV': e_pa,
                    })
        struct_offset += batch_size

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f'Saved MACE energy trajectory ({len(df)} rows, '
          f'{df["material_id"].nunique() if material_ids is not None and len(df) else n_structs} '
          f'structures) to {csv_path}')
    return df


def build_split_loader(cfg, model, split, test_bs=None):
    ds_group = cfg.data.datamodule.datasets
    if split == 'train':
        ds_cfgs = [ds_group.train]
    elif split == 'val':
        ds_cfgs = list(ds_group.val)
    elif split == 'test':
        ds_cfgs = list(ds_group.test)
    else:
        raise ValueError(f"unknown split {split!r} (expected train/val/test)")

    lattice_scaler = getattr(model, 'lattice_scaler', None)
    scaler = getattr(model, 'scaler', None)
    scalers = getattr(model, 'scalers', None)
    if scaler is None or lattice_scaler is None:
        raise RuntimeError(
            "model has no scaler / lattice_scaler -- expected "
            "lattice_scaler.pt / prop_scaler.pt / prop_scalers.pt next to the "
            "checkpoint (load_model loads them). CrystDataset.__getitem__ needs "
            "them to iterate.")

    datasets = [hydra.utils.instantiate(c) for c in ds_cfgs]
    for ds in datasets:
        ds.lattice_scaler = lattice_scaler
        ds.scaler = scaler
        ds.scalers = scalers

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    bs = test_bs or cfg.data.datamodule.batch_size.test
    return DataLoader(dataset, batch_size=bs, shuffle=False), ds_cfgs


def main(args):

    model_path = Path(args.model_path)

    outdir = Path(args.outdir) if args.outdir else model_path
    outdir.mkdir(parents=True, exist_ok=True)

    model, _, cfg = load_model(model_path, load_data=False, test_bs=args.test_bs)

    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()

    test_loader, split_dataset_cfgs = build_split_loader(
        cfg, model, args.split, test_bs=args.test_bs)
    print(f'Evaluating on the {args.split!r} split.')

    if args.eval_size is not None:
        test_loader = subsample_loader(test_loader, args.eval_size, seed=args.eval_seed)

    material_ids = None
    if len(split_dataset_cfgs) == 1:
        material_ids = get_material_ids_for_loader(test_loader, split_dataset_cfgs[0].path)
    else:
        print(f'WARNING: multiple {args.split} datasets configured -- material_id '
              'retrieval not implemented for this case, saving without it.')

    if args.perturb and args.from_noise:
        print("NOTE: both --perturb and --from_noise were passed -- --perturb "
              "takes precedence (init_structure is built from the rattled "
              "ground-truth structure, NOT random noise). Pass only one if "
              "that's not what you meant.")

    mace_calc = None
    if args.mace_energy_check:
        if args.eval_size is None or args.eval_size > 50:
            print(f"WARNING: --mace_energy_check is a manual, small-N sanity "
                  f"check (computes energy at every step, per structure) -- "
                  f"you have --eval_size={args.eval_size}. This is meant for "
                  f"~5-20 materials, not a full-dataset run; consider adding "
                  f"--eval_size 5 unless you really mean to run this at scale.")
        mace_device = args.mace_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        mace_calc = build_mace_calculator(args.mace_model, mace_device,
                                           default_dtype=args.mace_default_dtype,
                                           dispersion=args.mace_dispersion)

    if args.gt_file is not None:
        if not args.mace_energy_check:
            raise SystemExit(
                "--gt_file needs --mace_energy_check too (that's what builds "
                "the MACE calculator used to score the ground-truth structures).")
        if material_ids is None:
            raise SystemExit(
                "--gt_file needs material_ids to match ground-truth structures "
                "against -- material_id lookup failed above (see the WARNING "
                "printed for the loaded split), so --gt_file can't be used here.")

    print('Perturb-and-recover evaluation (relaxation feasibility test).')

    if args.ode_int_steps is not None:
        N = args.ode_int_steps
    elif args.step_lr is not None and args.step_lr > 0:
        N = round(1 / args.step_lr)
    else:
        raise SystemExit(
            "No integration step count given. Pass -N / --ode-int-steps "
            "(e.g. -N 1000), or a positive --step_lr. The default step_lr=-1 "
            "is a sentinel and is not resolved by this script."
        )

    if args.label == '':
        diff_out_name = 'eval_diff.pt'
        csv_out_name = 'eval_field_norms.csv'
        mace_csv_out_name = 'eval_mace_energy.csv'
    else:
        diff_out_name = f'eval_diff_{args.label}.pt'
        csv_out_name = f'eval_field_norms_{args.label}.csv'
        mace_csv_out_name = f'eval_mace_energy_{args.label}.csv'


    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch,
     n_steps_used, coord_norm_traj_all, lattice_norm_traj_all, mace_traj_all) = relax(
        test_loader, model, num_evals=args.num_evals,
        coord_noise=args.coord_noise, lattice_noise=args.lattice_noise,
        perturb=args.perturb,
        init_structure_from_batch=not args.from_noise,
        material_ids=material_ids, save_initial_cifs=args.save_initial_cifs,
        initial_cifs_dir=(outdir / 'initial_structures') if args.save_initial_cifs else None,
        outdir=outdir,
        model_path=model_path,
        diff_out_name = diff_out_name,
        mace_calc=mace_calc, mace_stride=args.mace_stride,
        null_baseline=args.null_baseline,
        N=N, eta=args.eta, sampler=args.sampler, mu=args.mu,
        anneal_lattice=args.anneal_lattice, anneal_coords=args.anneal_coords, anneal_type=args.anneal_type, anneal_slope=args.anneal_slope, anneal_offset=args.anneal_offset,
        guide_factor=args.guide_factor,
        grad_stop=args.grad_stop, grad_stop_coord=args.grad_stop_coord, grad_stop_lattice=args.grad_stop_lattice, min_steps=args.min_steps
    )

    print("\n===== RELAX OUTPUT SHAPES =====")

    print("frac_coords.shape          =", frac_coords.shape)
    print("atom_types.shape           =", atom_types.shape)
    print("lattices.shape             =", lattices.shape)
    print("lengths.shape              =", lengths.shape)
    print("angles.shape               =", angles.shape)
    print("num_atoms.shape            =", num_atoms.shape)
    print("input_data_batch           =", type(input_data_batch))
    print("input_data_batch.frac_coords.shape =",
        input_data_batch.frac_coords.shape)
    print("input_data_batch.atom_types.shape  =",
        input_data_batch.atom_types.shape)
    print("input_data_batch.lengths.shape     =",
        input_data_batch.lengths.shape)
    print("input_data_batch.angles.shape      =",
        input_data_batch.angles.shape)
    print("input_data_batch.num_atoms.shape   =",
        input_data_batch.num_atoms.shape)

    print("coord_norm_traj_all:", type(coord_norm_traj_all), len(coord_norm_traj_all))
    print("coord_norm_traj_all[0]:", type(coord_norm_traj_all[0]), len(coord_norm_traj_all[0]))
    print("coord_norm_traj_all[0][0].shape:", coord_norm_traj_all[0][0].shape)

    print("lattice_norm_traj_all[0][0].shape:", lattice_norm_traj_all[0][0].shape)


    # torch.save({
    #     'eval_setting': args,
    #     'input_data_batch': input_data_batch,
    #     'frac_coords': frac_coords,
    #     'num_atoms': num_atoms,
    #     'atom_types': atom_types,
    #     'lattices': lattices,
    #     'lengths': lengths,
    #     'angles': angles,
    #     'time': time.time() - start_time,
    #     'n_steps_used': n_steps_used,
    # }, outdir / diff_out_name)

    # print(f'Saved to {outdir / diff_out_name}')

    save_field_norm_csv(
        outdir / csv_out_name,
        material_ids, num_atoms, n_steps_used,
        coord_norm_traj_all, lattice_norm_traj_all,
    )

    if args.mace_energy_check:
        save_mace_energy_csv(
            outdir / mace_csv_out_name,
            material_ids, num_atoms,
            mace_traj_all,
        )

    if args.gt_file is not None:
        gt_csv_out_name = 'eval_gt_energy.csv' if args.label == '' else f'eval_gt_energy_{args.label}.csv'
        gt_n_atoms, gt_e_tot, gt_e_per_atom = load_ground_truth_energies(
            args.gt_file, material_ids, mace_calc, num_atoms_ref=num_atoms[0].tolist(),
        )
        gt_df = pd.DataFrame({
            'material_id': material_ids,
            'n_atoms': gt_n_atoms,
            'gt_energy_total_eV': gt_e_tot,
            'gt_energy_per_atom_eV': gt_e_per_atom,
        })
        gt_df.to_csv(outdir / gt_csv_out_name, index=False)
        print(f'Saved ground-truth MACE energies ({len(gt_df)} rows) to '
              f'{outdir / gt_csv_out_name}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('--outdir', default=None,
                     help='where this run writes its outputs (eval_diff*.pt, '
                          'eval_diff_traj.pt, the CSVs, initial_structures/) -- '
                          'default: same as --model_path (old behavior). Point '
                          'this at a per-experiment folder to keep repeated runs '
                          'against the same model from overwriting each other.')

    parser.add_argument('--num_evals', metavar='NEVAL', default=1, type=int, help="num repeat for each sample.")
    parser.add_argument('--test_bs', type=int, help="overwrite testset batchsize.")
    parser.add_argument('--label', default='', help="label for output")
    parser.add_argument('--split', choices=['test', 'train', 'val'], default='test',
                        help="which data split to evaluate on")

    step_group = parser.add_argument_group('evaluate step')
    step_group.add_argument('--dataset', help='load default step_lr of which dataset; effect when step_lr is -1')
    step_group.add_argument('--step_lr', default=-1, type=float, help="Step interval for ODE/SDE, -1 for SDE dataset defaults.")
    step_group.add_argument('-N', '--ode-int-steps', metavar='N', default=None, type=int, help="ODE integrate steps number; overwrite step_lr (default: None)")

    anneal_group = parser.add_argument_group('annealing')
    anneal_group.add_argument('--anneal_lattice', action="store_true", help="Anneal lattice.")
    anneal_group.add_argument('--anneal_coords', action="store_true", help="Anneal coords.")
    anneal_group.add_argument('--anneal_type', action="store_true", help="Anneal type.")
    anneal_group.add_argument('--anneal_slope', type=float, default=0.0, help="Anneal scope")
    anneal_group.add_argument('--anneal_offset', type=float, default=0.0, help="Anneal offset.")

    guidance_group = parser.add_argument_group('guidance')
    guidance_group.add_argument('--guide-factor', type=float, help='guidance factor (default: None)')

    eqm_group = parser.add_argument_group('EqM sampling')
    eqm_group.add_argument('--eta', type=float, default=None, help="EqM step size, independent of N")
    eqm_group.add_argument('--sampler', choices=['gd','nag'], default='gd')
    eqm_group.add_argument('--mu', type=float, default=0.3)


    perturb_group = parser.add_argument_group('perturbation')
    perturb_group.add_argument('--coord_noise', type=float, default=0.02,
                                help='stddev of fractional-coord Gaussian rattle')
    perturb_group.add_argument('--lattice_noise', type=float, default=0.02,
                                help='relative stddev on lengths / scale factor on angle noise (degrees)')
    perturb_group.add_argument('--from_noise', action='store_true',
                                help='sample from random noise (pure generation), for getting '
                                     "generation-mode data -- NOT the same thing as "
                                     "--null_baseline. This still calls model.sample() and runs "
                                     "the full sampler; --null_baseline instead skips the model "
                                     "entirely and echoes the input structure back unchanged, "
                                     "which is why it needs a real input and is incompatible "
                                     "with this flag. Default is OFF -- i.e. relax from "
                                     "ground truth, which is this script's whole purpose.")
    perturb_group.add_argument('--perturb', action='store_true',
                                help='initialize the sampler from a perturbed (rattled) structure '
                                     'instead of the ground-truth init structure')


    parser.add_argument('--null_baseline', action='store_true',
                     help='skip the model, evaluate the distortion itself (sanity check)')

    step_group.add_argument('--grad-stop', dest='grad_stop', type=float, default=None,
                         help="EqM adaptive early stop: field-norm threshold")
    step_group.add_argument('--min-steps', dest='min_steps', type=int, default=1)


    mace_group = parser.add_argument_group(
        'MACE energy check (manual, small-N diagnostic -- NOT for full-dataset runs)')
    mace_group.add_argument('--mace_energy_check', action='store_true',
                         help='compute MACE-predicted energy at each sampling step, per '
                              'structure, via ASE. Requires ase and mace-torch to be '
                              'installed -- both are imported lazily, only if this flag '
                              'is passed. Meant for --eval_size ~5-20, not full-dataset runs.')
    mace_group.add_argument('--mace-model', dest='mace_model', default='medium',
                         help="mace_mp foundation-model size ('small'/'medium'/'large'), "
                              "OR a path to a custom .model checkpoint (auto-detected). "
                              "The foundation model downloads weights on first use.")
    mace_group.add_argument('--mace_device', default=None, help='cuda/cpu; default: auto')
    mace_group.add_argument('--mace_stride', type=int, default=1,
                         help='compute energy every Nth step (default: every step -- '
                              'cheap at small --eval_size)')
    mace_group.add_argument('--mace_default_dtype', default='float32',
                         help="MACE calculator precision. Default 'float32' -- NOTE this "
                              "differs from mace_relax_single_cif.py's default ('float64'); "
                              "if comparing energies between the two tools, set this to "
                              "match whichever you're comparing against.")
    mace_group.add_argument('--mace_dispersion', action='store_true',
                         help='include a Grimme D3-style dispersion correction on top of '
                              'the raw MACE energy. Default OFF, matching the project\'s '
                              'other MACE scripts -- previously this was left unset here '
                              '(relying on mace_mp\'s own implicit default), now pinned '
                              'explicitly.')
    mace_group.add_argument('--gt_file', default=None,
                         help="CSV with 'material_id' and 'cif' columns (same convention "
                              "as compute_metrics.py's --gt_file) -- if given, also computes "
                              "MACE energy of the true ground-truth structure for each "
                              "material, matched by material_id (not row position), saved "
                              "to eval_gt_energy.csv. Requires --mace_energy_check.")



    parser.add_argument('--eval_size', type=int, default=None,
                         help='evaluate on a random subsample of this many structures '
                              'instead of the full test set (for fast iteration); '
                              'None = full test set')
    parser.add_argument('--eval_seed', type=int, default=0,
                         help='random seed for --eval_size subsampling (fixed default '
                              'so repeated runs at the same size are comparable)')

    step_group.add_argument('--grad-stop-coord', dest='grad_stop_coord', type=float, default=None,
                         help="EqM adaptive early stop: coord-field-norm threshold (overrides --grad-stop for coords)")

    step_group.add_argument('--grad-stop-lattice', dest='grad_stop_lattice', type=float, default=None,
                         help="EqM adaptive early stop: lattice-field-norm threshold (overrides --grad-stop for lattice)")

     
    parser.add_argument('--save_initial_cifs', action='store_true',
                     help='also write one .cif file per (material_id, eval_idx) for the '
                          'STARTING structure, to initial_structures/ next to the checkpoint '
                          '-- for opening directly in VESTA/OVITO/pymatgen. Off by default: '
                          'writes one file per material, wasteful at full-dataset scale. The '
                          'summary CSV (eval_initial_structure*.csv) is always written '
                          'regardless of this flag.')

    args = parser.parse_args()
    main(args)