import os
import numpy as np
from rdkit import Chem, Geometry
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import itertools
from copy import deepcopy
import re

# from utils.mol2frag import mol2frag

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}

def parse_sdf_file_leop(ligand_fn, retain_smi, mask_smi):
    
    fake_atom_num = 20

    mol = next(iter(Chem.SDMolSupplier(ligand_fn, removeHs=True)))

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    rd_num_atoms = mol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(mol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in mol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    mol_smi = Chem.MolToSmiles(mol)
    retain, mask = prepare_retain_and_mask(retain_smi, mask_smi, mol)

    anchors_idx_list = get_anchors_idx(retain)
    anchors_idx_list_mask = get_anchors_idx(mask)

    retain_pos, retain_ele = parse_molecule(retain)
    mask_pos, mask_ele = parse_molecule(mask)
    retain_pos = np.array(retain_pos, dtype = np.float32)
    mask_pos = np.array(mask_pos, dtype = np.float32)

    anchor1 = (0, 0)
    dist1 = 1000.
    anchor2 = (0, 0)
    dist2 = 1000.
    d_list = []
    d_i_list = []
    for i in range(len(retain_pos)):
        for j in range(len(mask_pos)):
            d = np.linalg.norm(retain_pos[i] - mask_pos[j])
            d_list.append(d)
            d_i_list.append((i, j))
    for i in range(len(d_list)):
        if d_list[i] < dist1:
            dist1 = d_list[i]
            anchor1 = d_i_list[i]
    if len(anchors_idx_list) == 2:
        for i in range(len(d_list)):
            if d_list[i] < dist2 and (d_i_list[i][0] != anchor1[0] or d_i_list[i][1] != anchor1[1]):
                dist2 = d_list[i]
                anchor2 = d_i_list[i]

    if len(anchors_idx_list) == 1:
        fake_pos = retain_pos[anchor1[0]]
    elif len(anchors_idx_list) == 2:
        fake_pos = (retain_pos[anchor1[0]] + retain_pos[anchor2[0]]) / 2

    mask_pos, mask_ele = parse_mask(mask, fake_pos, fake_atom_num)
    element = np.concatenate([retain_ele, mask_ele], axis = 0)
    pos = np.concatenate([retain_pos, mask_pos], axis = 0)
    element = np.array(element, dtype=np.int)
    pos = np.array(pos, dtype = np.float32)
    mask_mask = np.concatenate([np.zeros_like(retain_ele, dtype = np.bool_), np.ones_like(mask_ele, dtype = np.bool_)])

    # anchor feature
    anchor_feature = np.zeros(element.shape[0])
    if len(anchors_idx_list) == 1:
        anchor_feature[anchor1[0]] = 1
    elif len(anchors_idx_list) == 2:
        anchor_feature[anchor1[0]] = 1
        anchor_feature[anchor2[0]] = 1

    row, col, edge_type = [], [], []
    for bond in retain.GetBonds():
        b_type = int(bond.GetBondType())
        assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
        b_type = b_type if b_type != 12 else 4
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [b_type]
    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)
    perm = (edge_index[0] * retain.GetNumAtoms() + edge_index[1]).argsort()
    edge_index_retain = edge_index[:, perm]
    edge_type_retain = edge_type[perm]

    ptr = retain.GetNumAtoms()
    row, col, edge_type = [], [], []
    for bond in mask.GetBonds():
        b_type = int(bond.GetBondType())
        assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
        b_type = b_type if b_type != 12 else 4
        start = bond.GetBeginAtomIdx() + ptr
        end = bond.GetEndAtomIdx() + ptr
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [b_type]
    if len(anchors_idx_list) == 1:
        row += [anchor1[0], anchor1[1] + ptr]
        col += [anchor1[1] + ptr, anchor1[0]]
        edge_type += 2 * [1]
    else:
        row += [anchor1[0], anchor1[1] + ptr]
        col += [anchor1[1] + ptr, anchor1[0]]
        edge_type += 2 * [1]
        row += [anchor2[0], anchor2[1] + ptr]
        col += [anchor2[1] + ptr, anchor2[0]]
        edge_type += 2 * [1]

    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)
    perm = (edge_index[0] * mask.GetNumAtoms() + edge_index[1]).argsort()
    edge_index_mask = edge_index[:, perm]
    edge_type_mask = edge_type[perm]

    edge_index = np.concatenate([edge_index_retain, edge_index_mask], axis = 1)
    edge_type = np.concatenate([edge_type_retain, edge_type_mask], axis = 0)

    num_atoms = retain_ele.shape[0] + mask_ele.shape[0]
    num_bonds = edge_type.shape[0] // 2

    data = {
        'rdmol': mol,
        'smiles': mol_smi,
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,

        'atom_feature': feat_mat,
        'hybridization': hybridization, 

        'num_atoms': num_atoms,
        'num_bonds': num_bonds,

        'anchor_pos': np.array([fake_pos], dtype = np.float32),
        'mask_mask': mask_mask,

        'anchor_feature': anchor_feature,
    }
    return data

def get_exits(mol):
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

def set_anchor_flags(mol, anchor_idx):
    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            if atom.HasProp('_Anchor'):
                anchor_num = int(atom.GetProp('_Anchor')) + 1
                atom.SetProp('_Anchor', str(anchor_num))
            else:
                atom.SetProp('_Anchor', '1')
        else:
            if not atom.HasProp('_Anchor'):
                atom.SetProp('_Anchor', '0')

def update_scaffold(scaffold):
    exits = get_exits(scaffold)

    # Sort exit atoms by id for further correct deletion
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)

    # Remove exit bonds
    for exit in exits:
        exit_idx = exit.GetIdx()
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        anchor_idx = source_idx if target_idx == exit_idx else target_idx
        set_anchor_flags(scaffold, anchor_idx)

    escaffold = Chem.EditableMol(scaffold)
    for exit in exits:
        exit_idx = exit.GetIdx()
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        escaffold.RemoveBond(source_idx, target_idx)

    # Remove exit atoms
    for exit in exits:
        escaffold.RemoveAtom(exit.GetIdx())

    return escaffold.GetMol()

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def transfer_conformers(scaf, mol):
    matches = mol.GetSubstructMatches(scaf)

    if len(matches) < 1:
        raise Exception('Could not find scaffold or rgroup matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        scaf_coords = mol_coords[np.array(match)]
        scaf_conformer = create_conformer(scaf_coords)
        match2conf[match] = scaf_conformer

    return match2conf

def find_non_intersecting_matches(matches1, matches2):
    triplets = list(itertools.product(matches1, matches2))
    non_intersecting_matches = set()
    for m1, m2 in triplets:
        m1m2 = set(m1) & set(m2)
        if len(m1m2) == 0:
            non_intersecting_matches.add((m1, m2))
    return list(non_intersecting_matches)

def find_matches_with_rgroup_in_the_middle(non_intersecting_matches, mol):
    matches_with_rgroup_in_the_middle = []
    for m1, lm in non_intersecting_matches:
        neighbors = set()
        for atom_idx in lm:
            atom_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            for neighbor in atom_neighbors:
                neighbors.add(neighbor.GetIdx())

        conn1 = set(m1) & neighbors
        if len(conn1) == 1:
            matches_with_rgroup_in_the_middle.append((m1, lm))

    return matches_with_rgroup_in_the_middle

def find_correct_matches(matches_scaf, matches_rgroup, mol):
    non_intersecting_matches = find_non_intersecting_matches(matches_scaf, matches_rgroup)
    if len(non_intersecting_matches) == 1:
        return non_intersecting_matches

    return find_matches_with_rgroup_in_the_middle(non_intersecting_matches, mol)

def prepare_retain_and_mask(retain_smi, mask_smi, mol):
    retain = Chem.MolFromSmiles(retain_smi)
    mask = Chem.MolFromSmiles(mask_smi)

    newscaf = update_scaffold(retain)
    newrgroup = update_scaffold(mask)

    match2conf_scaf = transfer_conformers(newscaf, mol)
    match2conf_rgroup = transfer_conformers(newrgroup, mol)

    correct_matches = find_correct_matches(
        match2conf_scaf.keys(),
        match2conf_rgroup.keys(),
        mol,
    )

    if len(correct_matches) > 2:
        raise Exception('Found more than two scaffold matches')

    conf_scaf = match2conf_scaf[correct_matches[0][0]]
    conf_rgroup = match2conf_rgroup[correct_matches[0][1]]
    newscaf.AddConformer(conf_scaf)
    newrgroup.AddConformer(conf_rgroup)

    return newscaf, newrgroup

def fix_valence(mol):
    mol = deepcopy(mol)
    fixed = False
    cnt_loop = 0
    while True:
        try:
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except Exception as e:
            return mol, False  # from HERE: rerun sample
        cnt_loop += 1
        if cnt_loop > 100:
            break
        N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
        index = N4_valence.findall(err.args[0])
        if len(index) > 0:
            mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
    return mol

def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        anchor_num = int(atom.GetProp('_Anchor'))
        if anchor_num != 0:
            for _ in range(anchor_num):
                anchors_idx.append(atom.GetIdx())

    return anchors_idx

def parse_molecule(mol):
    elements = []
    for atom in mol.GetAtoms():
        elements.append(atom.GetAtomicNum())
    positions = mol.GetConformer().GetPositions()
    return np.array(positions), np.array(elements)

def parse_mask(mol, fake_pos, fake_atom_num):
    fake_pos = list(fake_pos)
    elements = []
    for atom in mol.GetAtoms():
        elements.append(atom.GetAtomicNum())
    positions = mol.GetConformer().GetPositions()
    elements.extend(0 for _ in range(fake_atom_num - mol.GetNumAtoms()))
    positions = positions.tolist()
    positions.extend(list(fake_pos + np.random.standard_normal(3) * 0.1) for _ in range(fake_atom_num - mol.GetNumAtoms()))
    return np.array(positions), np.array(elements)

def parse_mask_ta(mol):
    elements = []
    for atom in mol.GetAtoms():
        elements.append(atom.GetAtomicNum())
    positions = mol.GetConformer().GetPositions()
    return np.array(positions), np.array(elements)