import os
import pickle

import lmdb
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.data import get_pocket
from utils.data import ProteinLigandData, torchify_dict
from utils.data_leop import parse_sdf_file_leop, parse_sdf_file_leo_for_case

def get_dataset_linker_aff(config, **kwargs):
    name = config.name
    root = config.path
    if name == 'leop_linker':
        dataset_train = LeopDataset_linker(root, 'processed_train_linker.lmdb', './data/demo/linker/demo_dict.pt', **kwargs)
        dataset_test = LeopDataset_linker(root, 'processed_test_linker.lmdb', './data/demo/linker/demo_dict.pt', **kwargs)
    elif name == 'leop_linker_hop':
        dataset_test = LeopDataset_linker(root, 'processed_test_linker_hop.lmdb', './data/demo/linker/demo_dict.pt', **kwargs)
        dataset_train = None
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    return dataset_train, dataset_test

def get_dataset_dec_aff(config, **kwargs):
    name = config.name
    root = config.path
    if name == 'leop_dec':
        dataset_train = LeopDataset_dec(root, 'processed_train_dec.lmdb', './data/demo/dec/demo_dict.pt', **kwargs)
        dataset_test = LeopDataset_dec(root, 'processed_test_dec.lmdb', './data/demo/dec/demo_dict.pt', **kwargs)
    elif name == 'leop_dec_hop':
        dataset_test = LeopDataset_dec(root, 'processed_test_dec_hop.lmdb', './data/demo/dec/demo_dict.pt', **kwargs)
        dataset_train = None
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    return dataset_train, dataset_test

def get_dataset_aff_for_case(config, protein_filename, ligand_filename, anchor_id_given_1, anchor_id_given_2 = None, transform = None):
    name = config.name
    root = config.path
    if name == 'leop_for_case':
        dataset_test = LeoDataset_for_case(root, 'processed_case.lmdb', protein_filename, ligand_filename, anchor_id_given_1, anchor_id_given_2, transform)
        dataset_train = None
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    return dataset_train, dataset_test

class LeopDataset_linker(Dataset):

    def __init__(self, root, processed, leop_dict_path, transform=None):
        super().__init__()
        self.root = root
        
        self.processed_path = os.path.join(root, processed)
        self.name2id_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_name2idx.pt'
        self.leop_dict_path = leop_dict_path
        data_dir = '/path/to/crossdock2020/crossdocked_pocket10/'
        self.ligand_dir = data_dir
        self.protein_dir = data_dir
        self.transform = transform
        self.db = None

        self.keys = None

        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.name2id_path)):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=16*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )
        leop_dict = torch.load(self.leop_dict_path)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i in range(len(leop_dict['ligand_filename'])):
                pocket_fn, ligand_fn, retain_smi, mask_smi = \
                    leop_dict['protein_filename'][i], leop_dict['ligand_filename'][i], leop_dict['retain_smi'][i], leop_dict['mask_smi'][i]
                affinity = leop_dict['affinity'][i]
                if pocket_fn is None: continue
                try:
                    mask_mol = Chem.MolFromSmiles(mask_smi)
                    if mask_mol is None:
                        continue
                    pocket_path = os.path.join(self.protein_dir, pocket_fn)
                    ligand_path = os.path.join(self.ligand_dir, ligand_fn)
                    if not os.path.exists(pocket_path):
                        continue
                    try:
                        mol = Chem.MolFromMolFile(ligand_path)
                        mol = Chem.RemoveAllHs(mol)
                        Chem.SanitizeMol(mol)
                    except:
                        continue
                    if mol is None:
                        continue
                    pocket_dict = get_pocket(mol, pocket_path)
                    if len(pocket_dict['element']) == 0:
                        continue

                    ligand_dict = parse_sdf_file_leop(ligand_path, retain_smi, mask_smi)

                    num_ligand_atoms = ligand_dict['num_atoms']
                    # extract pocket atom mask - pocket_atom_masks - None

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )

                    # extract ligand atom mask
                    ligand_atom_mask = torch.zeros(num_ligand_atoms, dtype = int) - 1
                    ligand_atom_mask[data.ligand_mask_mask] = 0
                    data.ligand_atom_mask = ligand_atom_mask
                    
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data.retain_smi = retain_smi
                    data.mask_smi = mask_smi
                    data.affinity = affinity
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print(e)
                    print('Skipping (%d) %s %d' % (num_skipped, ligand_fn, i))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data

class LeopDataset_dec(Dataset):

    def __init__(self, root, processed, leop_dict_path, transform=None):
        super().__init__()
        self.root = root
        
        self.processed_path = os.path.join(root, processed)
        self.name2id_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_name2idx.pt'
        self.leop_dict_path = leop_dict_path
        data_dir = '/path/to/crossdock2020/crossdocked_pocket10/'
        self.ligand_dir = data_dir
        self.protein_dir = data_dir
        self.transform = transform
        self.db = None

        self.keys = None

        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.name2id_path)):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=16*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )
        leop_dict = torch.load(self.leop_dict_path)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i in range(len(leop_dict['ligand_filename'])):
                pocket_fn, ligand_fn, retain_smi, mask_smi = \
                    leop_dict['protein_filename'][i], leop_dict['ligand_filename'][i], leop_dict['retain_smi'][i], leop_dict['mask_smi'][i]
                affinity = leop_dict['affinity'][i]
                if pocket_fn is None: continue
                try:
                    mask_mol = Chem.MolFromSmiles(mask_smi)
                    if mask_mol is None:
                        continue
                    pocket_path = os.path.join(self.protein_dir, pocket_fn)
                    ligand_path = os.path.join(self.ligand_dir, ligand_fn)
                    if not os.path.exists(pocket_path):
                        continue
                    try:
                        mol = Chem.MolFromMolFile(ligand_path)
                        mol = Chem.RemoveAllHs(mol)
                        Chem.SanitizeMol(mol)
                    except:
                        continue
                    if mol is None:
                        continue
                    pocket_dict = get_pocket(mol, pocket_path)
                    if len(pocket_dict['element']) == 0:
                        continue

                    ligand_dict = parse_sdf_file_leop(ligand_path, retain_smi, mask_smi)

                    num_ligand_atoms = ligand_dict['num_atoms']
                    # extract pocket atom mask - pocket_atom_masks - None

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )

                    # extract ligand atom mask
                    ligand_atom_mask = torch.zeros(num_ligand_atoms, dtype = int) - 1
                    ligand_atom_mask[data.ligand_mask_mask] = 0
                    data.ligand_atom_mask = ligand_atom_mask
                    
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data.retain_smi = retain_smi
                    data.mask_smi = mask_smi
                    data.affinity = affinity
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print(e)
                    print('Skipping (%d) %s %d' % (num_skipped, ligand_fn, i))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data

class LeoDataset_for_case(Dataset):

    def __init__(self, root, processed, protein_filename, ligand_filename, anchor_id_given_1, anchor_id_given_2 = None, transform=None):
        super().__init__()
        self.root = root
        
        self.processed_path = os.path.join(root, processed)
        self.name2id_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_name2idx.pt'
        self.protein_filename, self.ligand_filename = protein_filename, ligand_filename
        self.anchor_id_given_1 = anchor_id_given_1
        self.anchor_id_given_2 = anchor_id_given_2
        self.transform = transform
        self.db = None

        self.keys = None

        # if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.name2id_path)):
        self._process()
        self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=16*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            pocket_fn, ligand_fn = \
                self.protein_filename, self.ligand_filename
            try:
                pocket_path = pocket_fn
                ligand_path = ligand_fn
                mol = Chem.MolFromMolFile(ligand_path)
                mol = Chem.RemoveAllHs(mol)
                Chem.SanitizeMol(mol)
                if mol is None:
                    print('[Error] Mol is None!')
                pocket_dict = get_pocket(mol, pocket_path)

                ligand_dict = parse_sdf_file_leo_for_case(ligand_path, self.anchor_id_given_1, self.anchor_id_given_2)

                num_ligand_atoms = ligand_dict['num_atoms']
                # extract pocket atom mask - pocket_atom_masks - None

                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pocket_dict),
                    ligand_dict=torchify_dict(ligand_dict),
                )

                # extract ligand atom mask
                ligand_atom_mask = torch.zeros(num_ligand_atoms, dtype = int) - 1
                ligand_atom_mask[data.ligand_mask_mask] = 0
                data.ligand_atom_mask = ligand_atom_mask
                
                data.protein_filename = pocket_fn
                data.ligand_filename = ligand_fn
                data.retain_smi = Chem.MolToSmiles(mol)
                data.mask_smi = ''
                data.affinity = 0.
                data = data.to_dict()  # avoid torch_geometric version issue
                txn.put(
                    key=str(0).encode(),
                    value=pickle.dumps(data)
                )
            except Exception as e:
                num_skipped += 1
                print(e)
                print('[Error] Unvalid ligand file!')
                
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data
