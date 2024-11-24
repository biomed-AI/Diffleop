# Datasets

We use the CrossDocked dataset to construct a demo dataset for example. If you want to process the dataset from scratch, you can follow the steps:

1. Download the dataset archive `crossdocked_pocket10.tar.gz` from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM).
2. Extract the TAR archive using the command: 
```bash
tar -xzvf crossdocked_pocket10.tar.gz
```
3. Using [`TANKBind`](https://github.com/luwei0917/TankBind) to calculate the binding affinity missing from the demo data, and construct the tuple list `[(ligand_filename, protein_filename, affinity)]`
4. Using MMAP to slice the molecules into scaffolds and R-group or fragments and linker in `demo_data_crossdock.ipynb`.
