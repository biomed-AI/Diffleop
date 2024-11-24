# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import numpy as np
import torch
import torch.nn.functional as F

from utils import data as utils_data
from utils.data import ProteinLigandData

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
    35: 7,
    53: 8,
    0: 9,
}

# # with B
# MAP_ATOM_TYPE_ONLY_TO_INDEX = {
#     6: 0,
#     7: 1,
#     8: 2,
#     9: 3,
#     15: 4,
#     16: 5,
#     17: 6,
#     35: 7,
#     53: 8,
#     5: 9,
#     0: 10,
# }

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}


def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number

def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


def get_hybridization_from_index(index, mode):
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    else:
        raise ValueError
    return hybridization
    
def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':
        return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
    else:
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]

class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def protein_feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data

class FeaturizeLigandAtom(object):

    def __init__(self, mode='basic'):
        super().__init__()
        assert mode in ['basic', 'add_aromatic', 'full']
        self.mode = mode
        self.atom_types_prob, self.bond_types_prob = None, None

    @property
    def ligand_feature_dim(self):
        if self.mode == 'basic':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        else:
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)

    def __call__(self, data: ProteinLigandData):
        element_list = data.ligand_element
        x = [get_index(e, None, None, self.mode) for e in element_list]
        x = torch.tensor(x)
        data.ligand_atom_feature_full = x
        data.ligand_mask_mask_b = data.ligand_mask_mask[data.ligand_mask_mask == True]
        return data

class AddIndicator(object):
    def __init__(self, add_to_ligand=True):
        super().__init__()
        self.add_to_ligand = add_to_ligand

    @property
    def ligand_feature_dim(self):
        ndim = 2 
        return ndim

    def __call__(self, data: ProteinLigandData):
        if self.add_to_ligand:
            arm_ind = F.one_hot((data.ligand_atom_mask >= 0).long(), num_classes=2)
            data.ligand_atom_aux_feature = arm_ind

        return data

class FeaturizeLigandBond(object):

    def __init__(self, mode='fc', set_bond_type=False):
        super().__init__()
        self.mode = mode
        self.set_bond_type = set_bond_type

    def __call__(self, data: ProteinLigandData):
        if self.mode == 'fc':
            n_atoms = len(data.ligand_atom_mask) 
            full_dst = torch.repeat_interleave(torch.arange(n_atoms), n_atoms)
            full_src = torch.arange(n_atoms).repeat(n_atoms)
            mask = full_dst != full_src
            full_dst, full_src = full_dst[mask], full_src[mask]
            data.ligand_fc_bond_index = torch.stack([full_src, full_dst], dim=0)
            assert data.ligand_fc_bond_index.size(0) == 2
        else:
            raise ValueError(self.mode)

        if hasattr(data, 'ligand_bond_index') and self.set_bond_type:
            n_atoms = data.ligand_pos.size(0)
            bond_matrix = torch.zeros(n_atoms, n_atoms).long()
            src, dst = data.ligand_bond_index
            bond_matrix[src, dst] = data.ligand_bond_type
            data.ligand_fc_bond_type = bond_matrix[data.ligand_fc_bond_index[0], data.ligand_fc_bond_index[1]]
        
        retain_num = torch.sum(~data.ligand_mask_mask)
        mask_edge_mask = (data.ligand_fc_bond_index[0, :] >= retain_num) | (data.ligand_fc_bond_index[1, :] >= retain_num)
        data.ligand_mask_edge_mask = mask_edge_mask

        data.ligand_mask_mask_b = data.ligand_mask_mask[data.ligand_mask_mask == True]
        data.ligand_mask_edge_mask_b = data.ligand_mask_edge_mask[data.ligand_mask_edge_mask == True]

        return data
        
class RandomRotation(object):

    def __init__(self):
        super().__init__()

    def __call__(self,  data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data
