import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group


import copy

import numpy as np
from diffcsp.common.data_utils import (
    EPSILON,
    cart_to_frac_coords,
    frac_to_cart_coords,
    lattice_params_to_matrix_torch,
    lattice_polar_build_torch,
    lattice_polar_decompose_torch,
    lengths_angles_to_volume,
    mard,
    min_distance_sqr_pbc,
)

import pandas as pd  # add if not already imported


def get_material_ids_for_loader(loader, csv_path):
    """Read the material_id column from the CSV the dataset was actually
    built from, in the SAME order CrystDataset iterates it. A length check
    guards against silent misalignment if preprocessing dropped/reordered
    rows. If loader.dataset is a torch.utils.data.Subset (from
    subsample_loader), the same subsetting is applied to material_ids so
    the two stay in lockstep."""
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
    """Return a new loader over a random subset of n structures from the
    same underlying dataset, same batch_size/collate behavior. Use this to
    iterate quickly (e.g. testing --symmetrize, tuning grad_stop) before
    committing to a full-dataset run."""
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
        # perturb in the SAME space and SAME scale convention the model trained on
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



from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def symmetrize_batch(frac_coords, lattices_mat, atom_types, num_atoms, symprec=0.1):
    """Post-hoc symmetrization of a batched sampler output. Splits the batch
    into individual pymatgen Structures via num_atoms, symmetrizes each,
    and reassembles into padded tensors. Falls back to the original
    structure on any symmetrization failure (never drops a structure).

    IMPORTANT: symmetrization can reorder atoms within a structure, so
    atom_types is re-derived from each symmetrized structure's own species
    list -- do not assume the original atom_types tensor still lines up.
    """
    frac_coords = frac_coords.detach().cpu().numpy()
    lattices_mat = lattices_mat.detach().cpu().numpy()
    atom_types = atom_types.detach().cpu().numpy()
    num_atoms_list = num_atoms.detach().cpu().numpy().tolist()
    n_still_p1 = 0
    n_exception = 0
    n_changed = 0

    out_frac, out_lattices, out_atom_types = [], [], []
    offset = 0
    for i, n in enumerate(num_atoms_list):
        fc = frac_coords[offset:offset + n]
        at = atom_types[offset:offset + n]
        lat = lattices_mat[i]
        offset += n

        try:
            struct = Structure(
                lattice=Lattice(lat), species=at, coords=fc, coords_are_cartesian=False
            )
            sga = SpacegroupAnalyzer(struct, symprec=symprec)
            sym_struct = sga.get_symmetrized_structure()
            spg_num = sga.get_space_group_number()
            if spg_num == 1:
                n_still_p1 += 1
            else:
                n_changed += 1

            if len(sym_struct) != n:
                raise ValueError("symmetrization changed atom count")  # safety guard, keep original

            out_frac.append(sym_struct.frac_coords)
            out_lattices.append(sym_struct.lattice.matrix)
            out_atom_types.append(np.array([s.specie.Z for s in sym_struct]))
        except Exception:
            n_exception += 1
            out_frac.append(fc)
            out_lattices.append(lat)
            out_atom_types.append(at)
    

    total = len(num_atoms_list)
    print(f'symmetrize_batch: {n_changed}/{total} found real symmetry, '
            f'{n_still_p1}/{total} still P1 at symprec={symprec}, '
            f'{n_exception}/{total} exceptions (fell back to original)')

    frac_coords_out = torch.tensor(np.concatenate(out_frac), dtype=torch.float32)
    lattices_out = torch.tensor(np.stack(out_lattices), dtype=torch.float32)
    atom_types_out = torch.tensor(np.concatenate(out_atom_types), dtype=torch.long)
    return frac_coords_out, lattices_out, atom_types_out



def relax(loader, model, num_evals, coord_noise, lattice_noise, null_baseline=False,symmetrize=False, symprec=0.1, **sample_kwargs):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    n_steps_used = []              # <-- ADD
    final_coord_field_norm = []    # <-- ADD
    final_lattice_field_norm = []  # <-- ADD
    coord_norm_traj_all = []    # <-- ADD
    lattice_norm_traj_all = []  # <-- ADD
    input_data_list = []
    device = next(model.parameters()).device

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_n_steps_used, batch_final_coord_norm, batch_final_lattice_norm = [], [], []
        batch_lattices = []
        batch_coord_traj, batch_lattice_traj = [], []
        for eval_idx in range(num_evals):
            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            init_structure = {
            'frac_coords': batch.frac_coords,
            'lattices_mat': lattice_params_to_matrix_torch(batch.lengths, batch.angles),
            }
            if null_baseline:
                # skip the model entirely -- "output" IS the initial structure,
                # i.e. what a "do nothing" relaxer would produce
                out_frac = init_structure['frac_coords'].detach().cpu()
                out_lattices = init_structure['lattices_mat'].detach().cpu()
                out_num_atoms = batch.num_atoms.detach().cpu()
                out_atom_types = batch.atom_types.detach().cpu()
                out_coord_traj = torch.zeros(0, batch.num_graphs)   # <-- ADD: nothing to record
                out_lattice_traj = torch.zeros(0, batch.num_graphs) # <-- ADD

                # no sampling happened, so these have no meaning -- fill with
                # NaN rather than 0, so they're visibly "not applicable"
                # rather than silently misread as "converged instantly"
                batch_size = batch.num_graphs
                nan_placeholder = torch.full((batch_size,), float('nan'))
                out_n_steps_used = nan_placeholder
                out_final_coord_norm = nan_placeholder
                out_final_lattice_norm = nan_placeholder
            else:


                outputs, traj = model.sample(batch, init_structure=init_structure, **sample_kwargs)
                out_frac = outputs['frac_coords'].detach().cpu()
                out_lattices = outputs['lattices'].detach().cpu()
                out_num_atoms = outputs['num_atoms'].detach().cpu()
                out_atom_types = outputs['atom_types'].detach().cpu()
                out_coord_traj = traj['coord_field_norm_traj']      # <-- ADD
                out_lattice_traj = traj['lattice_field_norm_traj']  # <-- ADD

                if symmetrize:
                    out_frac, out_lattices, out_atom_types = symmetrize_batch(
                        out_frac, out_lattices, out_atom_types, out_num_atoms, symprec=symprec
                    )
                out_n_steps_used = traj['n_steps_used']
                out_final_coord_norm = traj['final_coord_field_norm']
                out_final_lattice_norm = traj['final_lattice_field_norm']


            batch_frac_coords.append(out_frac)
            batch_num_atoms.append(out_num_atoms)
            batch_atom_types.append(out_atom_types)
            batch_lattices.append(out_lattices)
            batch_n_steps_used.append(out_n_steps_used)
            batch_final_coord_norm.append(out_final_coord_norm)
            batch_final_lattice_norm.append(out_final_lattice_norm)
            batch_coord_traj.append(out_coord_traj)    # <-- ADD
            batch_lattice_traj.append(out_lattice_traj) # <-- ADD



        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))
        n_steps_used.append(torch.stack(batch_n_steps_used, dim=0))
        final_coord_field_norm.append(torch.stack(batch_final_coord_norm, dim=0))
        final_lattice_field_norm.append(torch.stack(batch_final_lattice_norm, dim=0))
        coord_norm_traj_all.append(batch_coord_traj)     # <-- ADD: list of lists (ragged across batches)
        lattice_norm_traj_all.append(batch_lattice_traj) # <-- ADD

        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)
    n_steps_used = torch.cat(n_steps_used, dim=1)                       
    final_coord_field_norm = torch.cat(final_coord_field_norm, dim=1)   
    final_lattice_field_norm = torch.cat(final_lattice_field_norm, dim=1)  

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
        ,n_steps_used, final_coord_field_norm, final_lattice_field_norm,
        coord_norm_traj_all, lattice_norm_traj_all,
    )

def save_field_norm_csv(
    csv_path, material_ids, num_atoms, n_steps_used,
    coord_norm_traj_all, lattice_norm_traj_all,
):
    """Per-structure, per-STEP summary CSV: one row per (material_id,
    eval_idx, step), covering the FULL convergence trajectory rather than
    just the final norm at freeze time -- lets you plot/inspect how
    final_coord_field_norm / final_lattice_field_norm actually decay over
    the sampling trajectory for any given structure.

    coord_norm_traj_all / lattice_norm_traj_all: ragged nested lists,
    shape [n_batches][num_evals] of tensors [steps_this_batch_ran, batch_size],
    as returned by relax(). material_ids is a flat list aligned to the
    structure order relax() iterated (same convention as elsewhere).
    """
    num_atoms_np = num_atoms.numpy()          # [num_evals, n_structs]
    n_steps_np = n_steps_used.numpy()          # [num_evals, n_structs]

    n_structs = num_atoms_np.shape[1]
    if material_ids is not None and len(material_ids) != n_structs:
        print(f'WARNING: material_id count ({len(material_ids)}) != number of '
              f'evaluated structures ({n_structs}) -- writing CSV without '
              f'material_id (order/count mismatch, do not trust an unaligned join).')
        material_ids = None

    rows = []
    struct_offset = 0
    for batch_coord_traj, batch_lattice_traj in zip(coord_norm_traj_all, lattice_norm_traj_all):
        # batch_coord_traj is a list of length num_evals, each [steps, batch_size]
        batch_size = batch_coord_traj[0].shape[1] if batch_coord_traj[0].numel() > 0 else \
                     batch_lattice_traj[0].shape[1]

        for e, (coord_traj, lattice_traj) in enumerate(zip(batch_coord_traj, batch_lattice_traj)):
            n_steps_this = coord_traj.shape[0]  # 0 for null_baseline
            for local_i in range(batch_size):
                global_i = struct_offset + local_i
                mat_id = material_ids[global_i] if material_ids is not None else None
                n_atoms_val = int(num_atoms_np[e, global_i])
                n_steps_used_val = float(n_steps_np[e, global_i])

                if n_steps_this == 0:
                    # null baseline: no trajectory, one row with NaN norms
                    rows.append({
                        'material_id': mat_id, 'eval_idx': e, 'step': None,
                        'n_atoms': n_atoms_val, 'n_steps_used': n_steps_used_val,
                        'coord_field_norm': float('nan'),
                        'lattice_field_norm': float('nan'),
                    })
                else:
                    for step in range(n_steps_this):
                        if step + 1 > n_steps_used_val:   # <-- ADD: stop once this structure has frozen
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


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=True, test_bs=args.test_bs)

    if torch.cuda.is_available():
        model.to('cuda')
       
    
    if args.eval_size is not None:
        test_loader = subsample_loader(test_loader, args.eval_size, seed=args.eval_seed)

    material_ids = None
    test_dataset_cfgs = cfg.data.datamodule.datasets.test
    if len(test_dataset_cfgs) == 1:
        material_ids = get_material_ids_for_loader(test_loader, test_dataset_cfgs[0].path)
    else:
        print('WARNING: multiple test datasets configured -- material_id '
              'retrieval not implemented for this case, saving without it.')



    print('Perturb-and-recover evaluation (relaxation feasibility test).')

    N = args.ode_int_steps if args.ode_int_steps is not None else round(1 / args.step_lr)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch,
     n_steps_used, final_coord_field_norm, final_lattice_field_norm,coord_norm_traj_all, lattice_norm_traj_all) = relax(
        test_loader, model, num_evals=args.num_evals,
        coord_noise=args.coord_noise, lattice_noise=args.lattice_noise,
        symmetrize=args.symmetrize, symprec=args.symprec,        
        null_baseline=args.null_baseline,
        N=N, eta=args.eta, sampler=args.sampler, mu=args.mu,
        anneal_lattice=args.anneal_lattice, anneal_coords=args.anneal_coords, anneal_type=args.anneal_type, anneal_slope=args.anneal_slope, anneal_offset=args.anneal_offset,
        guide_factor=args.guide_factor,
        grad_stop=args.grad_stop, grad_stop_coord=args.grad_stop_coord, grad_stop_lattice=args.grad_stop_lattice, min_steps=args.min_steps
    )

    if args.label == '':
        diff_out_name = 'eval_diff.pt'
        csv_out_name = 'eval_field_norms.csv'
    else:
        diff_out_name = f'eval_diff_{args.label}.pt'
        csv_out_name = f'eval_field_norms_{args.label}.csv'
 

    torch.save({
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'time': time.time() - start_time,
        'n_steps_used': n_steps_used,
        'final_coord_field_norm': final_coord_field_norm, 
        'final_lattice_field_norm': final_lattice_field_norm,  
    }, model_path / diff_out_name)


    print(f'Saved to {model_path / diff_out_name}')

        
    save_field_norm_csv(
        model_path / csv_out_name,
        material_ids, num_atoms, n_steps_used,
        coord_norm_traj_all, lattice_norm_traj_all,
    )
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('--num_evals', metavar='NEVAL', default=1, type=int, help="num repeat for each sample.")
    parser.add_argument('--test_bs', type=int, help="overwrite testset batchsize.")
    parser.add_argument('--label', default='', help="label for output")

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


    parser.add_argument('--null_baseline', action='store_true',
                     help='skip the model, evaluate the distortion itself (sanity check)')
    
    step_group.add_argument('--grad-stop', dest='grad_stop', type=float, default=None,
                         help="EqM adaptive early stop: field-norm threshold")
    step_group.add_argument('--min-steps', dest='min_steps', type=int, default=1)


    parser.add_argument('--symmetrize', action='store_true',
                         help='apply post-hoc SpacegroupAnalyzer symmetrization to the final structure')
    parser.add_argument('--symprec', type=float, default=0.1,
                         help='symmetry-finding tolerance for --symmetrize')

    
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

    args = parser.parse_args()
    main(args) 