tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)
from Bio.PDB import PDBParser
from feature_utils import get_protein_feature
from feature_utils import extract_torchdrug_feature_from_mol
from rdkit import Chem
import torch
torch.set_num_threads(1)
from data import TankBind_prediction
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model import get_model
import pandas as pd
import os
import argparse
import heapq
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem import QED
from sascorer import compute_sa_score
from copy import deepcopy
import numpy as np

def cal_affinity(ligand_fn, protein_fn):
    '''
    using tankbind to calculate affinity
    '''
    batch_size = 1
    device= 'cpu'
    logging.basicConfig(level=logging.INFO)
    model = get_model(0, logging, device)
    # re-dock model
    # modelFile = "../saved_models/re_dock.pt"
    # self-dock model
    modelFile = "../saved_models/self_dock.pt"

    model.load_state_dict(torch.load(modelFile, map_location=device))
    _ = model.eval()

    pdb = protein_fn
    ligand_name = ligand_fn

    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", protein_fn)
    res_list = list(s.get_residues())

    protein_dict = {}
    protein_dict[pdb] = get_protein_feature(res_list)

    compound_dict = {}
    mol = Chem.MolFromMolFile(ligand_fn)
    compound_dict[ligand_name] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)

    l_pos = mol.GetConformer(0).GetPositions()
    l_center = (l_pos.max(0) + l_pos.min(0)) / 2

    info = []
    pdb_list = [pdb]
    for pdb in pdb_list:
        for compound_name in list(compound_dict.keys()):
            # use protein center as the block center.
            # com = ",".join([str(a.round(3)) for a in protein_dict[pdb][0].mean(axis=0).numpy()])
            com = ",".join([str(a.round(3)) for a in l_center])
            info.append([pdb, compound_name, "protein_center", com])
    info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])

    dataset_path = "test_dataset/"
    os.system(f"rm -r {dataset_path}")
    os.system(f"mkdir -p {dataset_path}")
    dataset = TankBind_prediction(dataset_path, data=info, protein_dict=protein_dict, compound_dict=compound_dict)

    dataset = TankBind_prediction(dataset_path)

    data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=0)
    affinity_pred_list = []
    y_pred_list = []
    for data in tqdm(data_loader):
        data = data.to(device)
        y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        for k in range(data.y_batch.max() + 1):
            y_pred_list.append((y_pred[data['y_batch'] == k]).detach().cpu())

    affinity_pred_list = torch.cat(affinity_pred_list)
    aff = affinity_pred_list.item()
    return aff

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/path/to/crossdock2020/crossdocked_pocket10/')
    parser.add_argument('--samples_dir', type=str, default='./outputs/sampling/sdf')
    args = parser.parse_args()
    
    disable_rdkit_logging()

    aff_ref_list = []
    aff_sam_list = []
    index_sam_list = []

    files = os.listdir(args.samples_dir)
    n_case = 0
    for i in files:
        n_case = max(n_case, int(i))
    for i in range(n_case + 1):
        tmp_sdf_dir = os.path.join(args.samples_dir, str(i)) # sdf/0/
        smiles_retain = open(os.path.join(tmp_sdf_dir, 'smiles_retain.smi'), 'r')
        lines = smiles_retain.readlines()
        protein_fn = lines[-1].strip()
        ref_ligand_fn = lines[-2].strip()
        aff_ref = cal_affinity(os.path.join(args.dataset_dir, ref_ligand_fn), os.path.join(args.dataset_dir, protein_fn))
        aff_ref_list.append(aff_ref)
        files = os.listdir(tmp_sdf_dir)
        n_samples = 0
        for name in files:
            num = name.split('.')[0]
            if num.isdigit():
                n_samples = max(n_samples, int(num))
        aff_sam = []
        index_sam = []
        for j in range(n_samples + 1):
            tmp_sdf = os.path.join(tmp_sdf_dir, str(j) + '.sdf') # sdf/0/0.sdf
            if not os.path.exists(tmp_sdf):
                continue
            mol = next(iter(Chem.SDMolSupplier(tmp_sdf, removeHs=True)))
            if mol is None:
                continue
            aff = cal_affinity(tmp_sdf, os.path.join(args.dataset_dir, protein_fn))
            aff_sam.append(aff)
            index_sam.append(j)
        aff_sam_list.append(aff_sam)
        index_sam_list.append(index_sam)

    # cal top 5 metric
    qed_list = []
    sa_list = []
    logp_list = []
    lipinski_list = []
    high_aff_list = []
    aff_list = []
    for i in range(len(aff_sam_list)):
        if len(aff_sam_list[i]) == 0:
            continue
        aff_ref = aff_ref_list[i]
        aff_top = heapq.nlargest(5, aff_sam_list[i])
        index_top = list(map(aff_sam_list[i].index, aff_top))
        tmp_sdf_dir = os.path.join(args.samples_dir, str(i)) # sdf/0/
        qed_tmp_list = []
        sa_tmp_list = []
        logp_tmp_list = []
        lipinski_tmp_list = []
        high_cnt = 0
        for j in range(len(index_top)):
            tmp_sdf = os.path.join(tmp_sdf_dir, str(index_sam_list[i][index_top[j]]) + '.sdf')
            if not os.path.exists(tmp_sdf):
                continue
            mol = next(iter(Chem.SDMolSupplier(tmp_sdf, removeHs=True)))
            if mol is None:
                continue
            if aff_top[j] >= aff_ref:
                high_cnt += 1
            qed = QED.qed(mol)
            qed_tmp_list.append(qed)
            sa = compute_sa_score(mol)
            sa_tmp_list.append(sa)
            logp = Crippen.MolLogP(mol)
            logp_tmp_list.append(logp)
            lipinski = obey_lipinski(mol)
            lipinski_tmp_list.append(lipinski)
        high_aff_list.append(high_cnt / len(index_top))
        aff_list.append(sum(aff_top) / len(aff_top))
        qed_list.append(sum(qed_tmp_list) / len(qed_tmp_list))
        sa_list.append(sum(sa_tmp_list) / len(sa_tmp_list))
        logp_list.append(sum(logp_tmp_list) / len(logp_tmp_list))
        lipinski_list.append(sum(lipinski_tmp_list) / len(lipinski_tmp_list))
    print('qed:', sum(qed_list) / len(qed_list))
    print('sa:', sum(sa_list) / len(sa_list))
    print('logp:', sum(logp_list) / len(logp_list))
    print('lipinski:', sum(lipinski_list) / len(lipinski_list))
    print('high affinity:', sum(high_aff_list) / len(high_aff_list))
    print('affinity:', sum(aff_list) / len(aff_list))