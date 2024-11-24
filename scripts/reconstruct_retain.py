import sys
sys.path.append('.')
import utils.misc as misc
import utils.transforms as trans
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from datasets.pl_data import FOLLOW_BATCH
import numpy as np
import utils.reconstruct as recon
from rdkit import Chem
import argparse
import torch
from Project.Diffleop.utils.data_leop import prepare_retain_and_mask, parse_molecule
import os
from datasets.pl_pair_dataset_affinity import get_dataset_linker_aff, get_dataset_dec_aff
torch.set_printoptions(threshold=np.inf)

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')

def get_test_set(config_file, task_type):
    config = misc.load_config(config_file)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = 'basic'
    ligand_featurizer = trans.FeaturizeLigandAtom(
        ligand_atom_mode)
    indicator = trans.AddIndicator()
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        indicator,
    ]
    transform_list.append(
        trans.FeaturizeLigandBond(mode='fc', set_bond_type=True)
    )
    transform = Compose(transform_list)

    if task_type == 'linker':
        train, test = get_dataset_linker_aff(
            config=config.data,
            transform=transform,
        )
    elif task_type == 'dec':
        train, test = get_dataset_dec_aff(
            config=config.data,
            transform=transform,
        )
    return test

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

if __name__ == '__main__':
    disable_rdkit_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/sampling_dec.yml')
    parser.add_argument('--dataset_dir', type=str, default='/path/to/crossdock2020/crossdocked_pocket10/')
    parser.add_argument('--samples_dir', type=str, default='./outputs/sampling/sdf')
    parser.add_argument('--type', type=str, default='dec')
    args = parser.parse_args()

    test_set = get_test_set(args.config_file, args.type)

    collate_exclude_keys = ['ligand_nbh_list']
    test_loader = DataLoader(test_set, 1, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys, num_workers = 0)
    ligand_atom_mode = 'basic'
    for batch_idx, batch in enumerate(test_loader):
        ligand_fn = os.path.join(args.dataset_dir, batch.ligand_filename[0])
        mol = next(iter(Chem.SDMolSupplier(ligand_fn, removeHs=True)))

        retain, mask = prepare_retain_and_mask(batch.retain_smi[0], batch.mask_smi[0], mol)

        mask_pos, mask_ele = parse_molecule(mask)
        mask_ele = mask_ele.tolist()
        row, col, mask_edge_type = [], [], []
        for bond in mask.GetBonds():
            b_type = int(bond.GetBondType())
            assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
            b_type = b_type if b_type != 12 else 4
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            mask_edge_type += 2 * [b_type]
        mask_edge_index = [row, col]

        retain_pos, retain_ele = parse_molecule(retain)
        retain_ele = retain_ele.tolist()
        row, col, retain_edge_type = [], [], []
        for bond in retain.GetBonds():
            b_type = int(bond.GetBondType())
            assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
            b_type = b_type if b_type != 12 else 4
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            retain_edge_type += 2 * [b_type]
        retain_edge_index = [row, col]

        # with bond
        mask_recon = recon.reconstruct_from_generated_with_bond(mask_pos, mask_ele, mask_edge_index, mask_edge_type)
        retain_recon = recon.reconstruct_from_generated_with_bond(retain_pos, retain_ele, retain_edge_index, retain_edge_type)
        # wo bond
        # pred_aromatic = None
        # mask_recon = recon.reconstruct_from_generated(mask_pos, mask_ele, pred_aromatic)

        Chem.MolToMolFile(mask_recon, args.samples_dir + '/' + str(batch_idx) + '/mask.sdf')
        Chem.MolToMolFile(retain_recon, args.samples_dir + '/' + str(batch_idx) + '/retain.sdf')
        results = {
            'pos': batch.ligand_pos.numpy(),
            'v': batch.ligand_atom_feature_full.numpy(),
            'bond': batch.ligand_fc_bond_type.numpy(),
        }
        n_graphs = batch.protein_element_batch.max().item() + 1
        batch_node, edge_index, batch_edge = batch.ligand_element_batch.numpy(), batch.ligand_fc_bond_index.numpy(), batch.ligand_fc_bond_type_batch.numpy()
        # batch_node = batch.ligand_element_batch.numpy()
        output_list_remove = seperate_outputs_remove(results, n_graphs, batch_node, edge_index, batch_edge)
        # output_list_remove = seperate_outputs_remove_wobond(results, n_graphs, batch_node)
        gen_list = []
        for i_mol, output_mol in enumerate(output_list_remove):
            pred_atom_type = trans.get_atomic_number_from_index(output_mol['pred'][0], mode=ligand_atom_mode)
            pred_bond_index = output_mol['edge_index'].tolist()
            mol = recon.reconstruct_from_generated_with_bond(output_mol['pred'][1], pred_atom_type, pred_bond_index,
                                                                        output_mol['pred'][2])
            # mol = recon.reconstruct_from_generated(output_mol['pred'][1], pred_atom_type, None)
            gen_list.append(mol)
        Chem.MolToMolFile(gen_list[0], args.samples_dir + '/' + str(batch_idx) + '/true_recon.sdf')
