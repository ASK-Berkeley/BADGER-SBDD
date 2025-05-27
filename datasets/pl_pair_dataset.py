import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from pathlib import Path
import json
import torch

from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict

debug = False

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final',eva_mode=False):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.eva_mode = eva_mode
        if not self.eva_mode:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        else:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + f'_sampled_processed_{version}.lmdb')
        self.transform = transform
        self.db = None

        self.keys = None
        # TODO: using json to load the dict() for looking up the label (Yue Jian)
        # TODO: trace back to parent dir ./data(YUE JIAN)

        self.parent = os.path.abspath(os.path.join(self.raw_path, os.pardir))

        # TODO: load .json file(YUE JIAN)
        if not self.eva_mode:
            with open(self.parent+'/dock_dict.json', 'r') as fp:
                temp = json.load(fp)
            self.dock_dict = temp
            del(temp) # release the memory
            self.SAQED = torch.load(os.path.join(self.parent,"SAQED.pt"))
            # print(self.dock_dict)

        # TODO: load SA and QED here
        

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

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
        # print("key after db",self.keys)
    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                # print("ddddddd")
                if pocket_fn is None: continue
                try:
                    if self.eva_mode:
                        data_prefix = os.path.join(self.raw_path+'/res')
                    else:
                        data_prefix = self.raw_path

                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        # print("h\n\n\n")
        if self.db is None:
            self._connect_db()

        key = self.keys[idx]

        data = pickle.loads(self.db.begin().get(key))

        data = ProteinLigandData(**data)
        data.id = idx
        # TODO: labeling the data here (Yue Jian)
        if not self.eva_mode:
            # print("fn",data.ligand_filename[:-4])
            # print("y",self.dock_dict[data.ligand_filename[:-4]])
            data.y = self.dock_dict[data.ligand_filename[:-4]]
            data.sa = self.SAQED[data.ligand_filename]["SA"]
            data.qed = self.SAQED[data.ligand_filename]["QED"]
        assert data.protein_pos.size(0) > 0
        if debug:
            print("one data",data)
        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
