import sys
sys.path.append('.')
import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.reconstruct as recon
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH
from models.diffleop import Diffleop
from datasets.pl_pair_dataset_affinity import get_dataset_linker_aff, get_dataset_dec_aff

torch.set_printoptions(profile="full")

def seperate_outputs_remove(outputs, n_graphs, batch_node, edge_index, batch_edge):
    pos = np.array(outputs['pos'], dtype = np.double)
    v = np.array(outputs['v'])
    bond = np.array(outputs['bond'])

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_edge = (batch_edge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_edge.sum()
        v_ = v[ind_node]
        pos_ = pos[ind_node]
        bond_ = bond[ind_edge]

        isnot_masked_atom = (v_ < 9)
        is_bond = (bond_ > 0)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())

        v_ = v_[isnot_masked_atom]
        pos_ = pos_[isnot_masked_atom]
        bond_ = bond_[is_bond]
        
        halfedge_index_this = edge_index[:, ind_edge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        halfedge_index_this = halfedge_index_this[:, is_bond]
        if not isnot_masked_atom.all():
            halfedge_index_this = edge_index_changer[halfedge_index_this]
            bond_for_masked_atom = (halfedge_index_this < 0).any(axis=0)
            halfedge_index_this = halfedge_index_this[:, ~bond_for_masked_atom]
            bond_ = bond_[~bond_for_masked_atom]

        new_pred_this_1 = [v_,  # node type
                         pos_,  # node pos
                         bond_]  # halfedge type
        
        new_outputs.append({
            'pred': new_pred_this_1,
            'edge_index': halfedge_index_this,
        })
    return new_outputs

def seperate_outputs_remove_traj(outputs, n_graphs, batch_node, edge_index, batch_edge, traj_i):
    pos = np.array(outputs['pos_traj'][traj_i], dtype = np.double)
    v = np.array(outputs['v_traj'][traj_i])
    bond = np.array(outputs['bond_traj'][traj_i])

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_edge = (batch_edge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_edge.sum()
        v_ = v[ind_node]
        pos_ = pos[ind_node]
        bond_ = bond[ind_edge]

        isnot_masked_atom = (v_ < 9)
        is_bond = (bond_ > 0)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())

        v_ = v_[isnot_masked_atom]
        pos_ = pos_[isnot_masked_atom]
        bond_ = bond_[is_bond]
        
        halfedge_index_this = edge_index[:, ind_edge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        halfedge_index_this = halfedge_index_this[:, is_bond]
        if not isnot_masked_atom.all():
            halfedge_index_this = edge_index_changer[halfedge_index_this]
            bond_for_masked_atom = (halfedge_index_this < 0).any(axis=0)
            halfedge_index_this = halfedge_index_this[:, ~bond_for_masked_atom]
            bond_ = bond_[~bond_for_masked_atom]

        new_pred_this_1 = [v_,  # node type
                         pos_,  # node pos
                         bond_]  # halfedge type
        
        new_outputs.append({
            'pred': new_pred_this_1,
            'edge_index': halfedge_index_this,
        })
    print(new_outputs)
    return new_outputs

def seperate_outputs_remove_wobond(outputs, n_graphs, batch_node):
    pos = np.array(outputs['pos'], dtype = np.double)
    v = np.array(outputs['v'])

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        v_ = v[ind_node]
        pos_ = pos[ind_node]

        isnot_masked_atom = (v_ < 9)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())

        v_ = v_[isnot_masked_atom]
        pos_ = pos_[isnot_masked_atom]
        
        new_pred_this_1 = [v_,  # node type
                         pos_,]  # node pos
        
        new_outputs.append({
            'pred': new_pred_this_1,
        })
    return new_outputs

def seperate_outputs(outputs, n_graphs, batch_node, edge_index, batch_edge):
    pos = np.array(outputs['pos'], dtype = np.double)
    v = np.array(outputs['v'])
    bond = np.array(outputs['bond'])

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_edge = (batch_edge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_edge.sum()

        new_pred_this_1 = [v[ind_node],  # node type
                         pos[ind_node],  # node pos
                         bond[ind_edge]]  # halfedge type
        
        halfedge_index_this = edge_index[:, ind_edge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        new_outputs.append({
            'pred': new_pred_this_1,
            'edge_index': halfedge_index_this,
        })
    return new_outputs


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def check_if_generated(id_path_list, n_samples):
    generated = True
    starting_points = []
    for id_path in id_path_list:
        numbers = []
        for fname in os.listdir(id_path):
            try:
                num = int(fname.split('.')[0])
                numbers.append(num)
            except:
                continue
        if len(numbers) == 0 or max(numbers) != n_samples - 1:
            generated = False
            if len(numbers) == 0:
                starting_points.append(0)
            else:
                starting_points.append(max(numbers) + 1)
    if len(starting_points) == 0:
        starting = 0
    elif len(starting_points) == len(id_path_list) or (max(starting_points) - min(starting_points)) > 10:
        starting = min(starting_points)
    else:
        generated = True
        starting = None

    return generated, starting

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--recon_with_bond', type=eval, default=True)
    parser.add_argument('--type', type=str, default='dec')

    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    # Load config
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.sample.seed)

    log_dir = os.path.join(args.outdir, '%s_%03d' % (config_name, args.data_id))
    os.makedirs(log_dir, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Load checkpoint
    assert config.model.checkpoint or args.ckpt_path
    ckpt_path = args.ckpt_path if args.ckpt_path is not None else config.model.checkpoint
    ckpt = torch.load(ckpt_path, map_location=args.device)
    if 'train_config' in config.model:
        logger.info(f"Load training config from: {config.model['train_config']}")
        ckpt['config'] = misc.load_config(config.model['train_config'])
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    cfg_transform = ckpt['config'].data.transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = cfg_transform.ligand_atom_mode
    ligand_bond_mode = cfg_transform.ligand_bond_mode

    ligand_featurizer = trans.FeaturizeLigandAtom(
        ligand_atom_mode)
    indicator = trans.AddIndicator()
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        indicator,
    ]
    if getattr(ckpt['config'].model, 'bond_diffusion', False):
        transform_list.append(
            trans.FeaturizeLigandBond(mode=config.data.transform.ligand_bond_mode, set_bond_type=True)
        )
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    if args.type == 'linker':
        train, test = get_dataset_linker_aff(
            config=config.data,
            transform=transform,
        )
    elif args.type == 'dec':
        train, test = get_dataset_dec_aff(
            config=config.data,
            transform=transform,
        )
    train_set, test_set = train, test
    logger.info(f'Training: {len(train_set)} Validation: {len(test_set)}')

    protein_feature_dim = sum([getattr(t, 'protein_feature_dim', 0) for t in transform_list])
    ligand_feature_dim = sum([getattr(t, 'ligand_feature_dim', 0) for t in transform_list])
    ligand_feature_dim += 1

    # Load model
    model = Diffleop(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_feature_dim,
        ligand_atom_feature_dim=ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        prior_atom_types=ligand_featurizer.atom_types_prob, prior_bond_types=ligand_featurizer.bond_types_prob
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=True)

    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    collate_exclude_keys = ['ligand_nbh_list']
    ## traj
    test_loader = DataLoader(test_set, config.sample.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys, num_workers = 2)
    
    if config.affinity_predictor is not None:
        print('=== guiding ===')
        gui_scale_pos = 5.0
        gui_scale_node = 50.0
        gui_scale_bond = 50.0
    else:
        print('=== no guiding ===')
        gui_scale_pos = None
        gui_scale_node = None
        gui_scale_bond = None
        affinity_predictor = None

    print('=== recon_with_bond', args.recon_with_bond, '===')

    results_list = [[] for i in range(len(test_set))]
    if os.path.exists(os.path.join(log_dir, 'results.pt')):
        results_list = torch.load(os.path.join(log_dir, 'results.pt'))
    sdf_dir = os.path.join(log_dir, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    for batch_idx, batch in enumerate(test_loader):
        batch = batch.to(args.device)
        n_graphs = batch.protein_element_batch.max().item() + 1
        ids = []
        id_path_list = []
        for i_mol, idx in enumerate(batch.id):
            idx = str(idx.item())
            ids.append(idx)
            id_path = os.path.join(sdf_dir, idx)
            id_path_list.append(id_path)
            os.makedirs(id_path, exist_ok=True)
            smiles_path = os.path.join(id_path, 'smiles_retain.smi')
            with open(smiles_path, 'w') as smiles_f:
                smiles_f.write(batch.retain_smi[i_mol] + '\n')
                smiles_f.write(batch.mask_smi[i_mol] + '\n')
                smiles_f.write(batch.ligand_smiles[i_mol] + '\n')
                smiles_f.write(batch.ligand_filename[i_mol] + '\n')
                smiles_f.write(batch.protein_filename[i_mol] + '\n')
        
        generated, starting_point = check_if_generated(id_path_list, config.sample.num_samples)
        if generated:
            print(f'Already generated batch={batch_idx}, max_uuid={max(ids)}')
            continue
        if starting_point > 0:
            print(f'Generating {config.sample.num_samples - starting_point} for batch={batch_idx}')
        print(f'starting point {starting_point}')
        
        n_recon_success, n_complete = 0, 0
        for i in tqdm(range(starting_point, config.sample.num_samples), desc=str(batch_idx)):
            results = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                ligand_v_aux=batch.ligand_atom_aux_feature.float(),
                batch_ligand=batch.ligand_element_batch,

                ligand_fc_bond_index=getattr(batch, 'ligand_fc_bond_index', None),
                ligand_fc_bond_type=getattr(batch, 'ligand_fc_bond_type', None),
                batch_ligand_bond=getattr(batch, 'ligand_fc_bond_type_batch', None),

                anchor_pos = batch.ligand_anchor_pos,
                mask_mask = batch.ligand_mask_mask,
                mask_edge_mask = getattr(batch, 'ligand_mask_edge_mask', None),
                ligand_mask_mask_b_batch = batch.ligand_mask_mask_b_batch,
                ligand_mask_edge_mask_b_batch = getattr(batch, 'ligand_mask_edge_mask_b_batch', None),

                anchor_feature = batch.ligand_anchor_feature,

                gui_scale_pos = gui_scale_pos,
                gui_scale_node = gui_scale_node,
                gui_scale_bond = gui_scale_bond,
            )
            results = {key:[v.cpu().numpy() for v in value] for key, value in results.items()}
            batch_node, edge_index, batch_edge = batch.ligand_element_batch.clone().cpu().numpy(), batch.ligand_fc_bond_index.clone().cpu().numpy(), batch.ligand_fc_bond_type_batch.clone().cpu().numpy()
            # batch_node = batch.ligand_element_batch.clone().cpu().numpy()
            # output_list = seperate_outputs(results, n_graphs, batch_node, edge_index, batch_edge)
            # output_list_remove = seperate_outputs_remove_traj(results, n_graphs, batch_node, edge_index, batch_edge, 0)
            output_list_remove = seperate_outputs_remove(results, n_graphs, batch_node, edge_index, batch_edge)
            # output_list_remove = seperate_outputs_remove_wobond(results, n_graphs, batch_node)
            gen_list = []
            for i_mol, output_mol in enumerate(output_list_remove):
                pred_atom_type = trans.get_atomic_number_from_index(output_mol['pred'][0], mode=ligand_atom_mode)
                pred_bond_index = output_mol['edge_index'].tolist()
                # reconstruction
                try:
                    pred_aromatic = trans.is_aromatic_from_index(output_mol['pred'][0], mode=ligand_atom_mode)
                    if args.recon_with_bond:
                        mol = recon.reconstruct_from_generated_with_bond(output_mol['pred'][1], pred_atom_type, pred_bond_index,
                                                                        output_mol['pred'][2])
                    else:
                        mol = recon.reconstruct_from_generated(output_mol['pred'][1], pred_atom_type, pred_aromatic)
                    smiles = Chem.MolToSmiles(mol)
                    n_recon_success += 1

                except recon.MolReconsError:
                    logger.warning('Reconstruct failed %s' % f'{i}')
                    mol = None
                    smiles = ''

                if mol is not None and '.' not in smiles:
                    n_complete += 1
                print(smiles)
                gen_list.append(mol)
                idx = int(batch.id[i_mol].item())
                results_list[idx].append(
                    {
                        'mol': mol,
                        'smiles': smiles,
                    }
                )

            for rdmol, id_path in zip(gen_list, id_path_list):
                smiles_path = os.path.join(id_path, 'smiles.smi')
                if rdmol is None:
                    with open(smiles_path, 'a') as smiles_f:
                        smiles_f.write('*\n')
                    continue
                else:
                    with open(smiles_path, 'a') as smiles_f:
                        smiles_f.write(Chem.MolToSmiles(rdmol) + '\n')
                    Chem.MolToMolFile(rdmol, os.path.join(id_path, '%d.sdf' % (i)))
        torch.save(results_list, os.path.join(log_dir, 'results.pt'))
