import subprocess
import glob
import argparse
import os
import torch
import MDAnalysis as mda
from openbabel import openbabel as ob
import pandas as pd
import prolif as plf
import pandas as pd
import rdkit.Chem as Chem
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from datamol.conformers._conformers import _get_ff
from rdkit.Chem import AllChem
from copy import deepcopy
import datamol as dm
REDUCE_PATH = "reduce"


def load_protein_prolif(protein_path: str):
    """Load protein from PDB file using MDAnalysis
    and convert to plf.Molecule. Assumes hydrogens are present."""
    prot = mda.Universe(protein_path)
    prot = plf.Molecule.from_mda(prot, NoImplicit=False)
    return prot

def load_protein_chemmol(protein_path: str):
    """Load a protein from a PDB file and convert to an RDKit Mol object."""
    rdkit_mol = Chem.MolFromPDBFile(protein_path, removeHs=False)
    return rdkit_mol

def load_protein_from_pdb(pdb_path: str, reduce_path: str = REDUCE_PATH):
    """Load protein from PDB file, add hydrogens, and convert it to a prolif.Molecule.

    Args:
        pdb_path (str): The path to the PDB file.
        reduce_path (str, optional): The path to the reduce executable. Defaults to REDUCE_PATH.

    Returns:
        plf.Molecule: The loaded protein as a prolif.Molecule.
    """
    tmp_path = pdb_path.split(".pdb")[0] + "_tmp.pdb"

    # Call reduce to make tmp PDB with waters
    reduce_command = f"{reduce_path} -NOFLIP  {pdb_path} -Quiet > {tmp_path}"
    subprocess.run(reduce_command, shell=True)

    # Load the protein from the temporary PDB file
    prot = load_protein_prolif(tmp_path)
    # os.remove(tmp_path)

    return prot

def load_sdf_prolif(sdf_path: str) -> plf.Molecule:
    """Load ligand from an SDF file and convert it to a prolif.Molecule.

    Args:
        sdf_path (str): Path to the SDF file.

    Returns:
        plf.Molecule: The loaded ligand as a prolif.Molecule.
    """
    return plf.sdf_supplier(sdf_path)

def load_mols_from_sdf(sdf_path: str, add_hs: bool = True):
    """Load ligand from an SDF file, add hydrogens, and convert it to a prolif.Molecule.

    Args:
        sdf_path (str): Path to the SDF file.
        add_hs (bool, optional): Whether to add hydrogens. Defaults to True.

    Returns:
        Union[plf.Molecule, List]: The loaded ligand as a prolif.Molecule, or an empty list if no molecule could be loaded.
    """
    tmp_path = sdf_path.split(".sdf")[0] + "_tmp.sdf"

    # Load molecules from the SDF file
    mols = dm.read_sdf(sdf_path)[0]
    # print(type(mols))

    # Remove radicals from the molecules
    mol = remove_radicals(mols)

    # Add hydrogens to the molecules
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    dm.to_sdf([mol], tmp_path)
    ligs = load_sdf_prolif(tmp_path)
    os.remove(tmp_path)

    # # Turn into list
    # ligs = list(ligs)

    return ligs[0]

def get_pdbqt_mol(pdbqt_block: str) -> Chem.Mol:
    """Convert pdbqt block to rdkit mol by converting with openbabel"""
    # write pdbqt file
    with open("test_pdbqt.pdbqt", "w") as f:
        f.write(pdbqt_block)

    # read pdbqt file from autodock
    mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    obConversion.ReadFile(mol, "test_pdbqt.pdbqt")

    # convert to RDKIT
    mol = Chem.MolFromPDBBlock(obConversion.WriteString(mol))

    # remove tmp file
    os.remove("test_pdbqt.pdbqt")

    return mol

def has_radicals(mol: Chem.Mol) -> bool:
    """Check if a molecule has any free radicals."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radicals)
            atom.SetNumRadicalElectrons(0)

    return False

def remove_radicals(mol: Chem.Mol, sanitize: bool = True) -> Chem.Mol:
    """Remove free radicals from a molecule."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            # Saturate the atom with hydrogen atoms
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radicals)
            atom.SetNumRadicalElectrons(0)

    if sanitize:
        dm.sanitize_mol(mol)

    return mol

def count_clashes(prot: Chem.Mol, lig: Chem.Mol, tollerance: float = 0.5) -> int:
    """
    Counts the number of clashes between atoms in a protein and a ligand.

    Args:
        prot: RDKit Mol object representing the protein.
        lig: RDKit Mol object representing the ligand.
        tolerance: Distance tolerance for clash detection (default: 0.5).

    Returns:
        clashes: Number of clashes between the protein and the ligand.
    """

    # Check if the molecule has radicals   
         
    if not prot or not lig:
        return np.nan
    
    assert not has_radicals(
        lig
    ), "Molecule has radicals, consider removing them first. (`posecheck.utils.chem.remove_radicals()`)"

    clashes = 0

    try:
        # Get the positions of atoms in the protein and ligand
        prot_pos = prot.GetConformer().GetPositions()
        lig_pos = lig.GetConformer().GetPositions()

        pt = Chem.GetPeriodicTable()

        # Get the number of atoms in the protein and ligand
        
        num_prot_atoms = prot.GetNumAtoms()
        num_lig_atoms = lig.GetNumAtoms()

        # Calculate the Euclidean distances between all atom pairs in the protein and ligand
        dists = np.linalg.norm(
            prot_pos[:, np.newaxis, :] - lig_pos[np.newaxis, :, :], axis=-1
        )

        # Iterate over the ligand atoms
        for i in range(num_lig_atoms):
            lig_vdw = pt.GetRvdw(lig.GetAtomWithIdx(i).GetAtomicNum())

            # Iterate over the protein atoms
            for j in range(num_prot_atoms):
                prot_vdw = pt.GetRvdw(prot.GetAtomWithIdx(j).GetAtomicNum())

                # Check for clash by comparing the distances with tolerance
                if dists[j, i] + tollerance < lig_vdw + prot_vdw:
                    clashes += 1

    except AttributeError:
        raise ValueError(
            "Invalid input molecules. Please provide valid RDKit Mol objects."
        )

    return clashes

def calculate_energy(
    mol: Chem.Mol, forcefield: str = "UFF", add_hs: bool = True
) -> float:
    """
    Evaluates the energy of a molecule using a force field.

    Args:
        mol: RDKit Mol object representing the molecule.
        forcefield: Force field to use for energy calculation (default: "UFF").
        add_hs: Whether to add hydrogens to the molecule (default: True).

    Returns:
        energy: Calculated energy of the molecule.
                Returns NaN if energy calculation fails.
    """
    mol = Chem.Mol(mol)  # Make a deep copy of the molecule

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = _get_ff(mol, forcefield=forcefield)
    except Exception:
        return np.nan

    energy = ff.CalcEnergy()

    return energy


def relax_constrained(
    mol: Chem.Mol, forcefield: str = "UFF", add_hs: bool = True, maxDispl=0.1
) -> float:
    """
    Calculates the energy of a molecule using a force field.

    Args:
        mol: RDKit Mol object representing the molecule.
        forcefield: Force field to use for energy calculation (default: "UFF").
        add_hs: Whether to add hydrogens to the molecule (default: True).

    Returns:
        energy: Calculated energy of the molecule (rounded to 2 decimal places).
                Returns NaN if energy calculation fails.
    """
    mol = deepcopy(mol)  # Make a deep copy of the molecule
    #mol = Chem.Mol(mol)  # Make a deep copy of the molecule

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = _get_ff(mol, forcefield=forcefield)
    except Exception:
        return np.nan
    
    for i in range(mol.GetNumAtoms()):
        ff.UFFAddPositionConstraint(i, maxDispl=maxDispl, forceConstant=1)


    try:
        ff.Minimize()
        return mol
    except:
        None
        
def relax_global(mol: Chem.Mol) -> Chem.Mol:
    """Relax a molecule by adding hydrogens, embedding it, and optimizing it
    using the UFF force field.

    Args:
        mol (Chem.Mol): The molecule to relax.

    Returns:
        Chem.Mol: The relaxed molecule.
    """

    # if the molecule is None, return None
    if mol is None:
        return None

    # Incase ring info is not present
    Chem.GetSSSR(mol)  # SSSR: Smallest Set of Smallest Rings

    # make a copy of the molecule
    mol = deepcopy(mol)

    # add hydrogens
    mol = Chem.AddHs(mol, addCoords=True)

    # embed the molecule
    #AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.EmbedMolecule(mol)

    # optimize the molecule
    AllChem.UFFOptimizeMolecule(mol)

    # return the molecule
    return mol

def calculate_strain_energy(mol: Chem.Mol, maxDispl: float = 0.1, num_confs: int = 50) -> float:
    """Calculate the strain energy of a molecule.
    
    In order to evaluate the global strain energy of a molecule, rather than local imperfections
    in bonds distances and angles, we first perform a local relaxation of the molecule (by minimizing and allowing 
    a small displacement of the atoms) and then sample and minimize n conformers of the molecule.

    Args:
        mol (Chem.Mol): The molecule to calculate the strain energy for.
        maxDispl (float): The maximum displacement for position constraints during local relaxation. (Default: 0.1)
        num_confs (int): The number of conformers to generate for global relaxation.

    Returns:
        float: The calculated strain energy, or None if the calculation fails.
    """
    try:
        # relax molecule enforcing constraints on the atom positions
        locally_relaxed = relax_constrained(mol, maxDispl=maxDispl)
        # sample and minimize n conformers 
        global_relaxed = [relax_global(mol) for i in range(num_confs)]
            
        # calculate the energy of the locally relaxed molecule
        local_energy = calculate_energy(locally_relaxed)
        
        # calculate the energy of the globally relaxed molecules and take the minimum
        global_energy = min([calculate_energy(mol) for mol in global_relaxed])
        
        # calculate the strain energy
        strain_energy = local_energy - global_energy
        
        return strain_energy
    
    except Exception as e:
        print('Warning: Strain energy calculation failed')
        print(e)
        return None
    
def generate_interaction_df(prot: plf.Molecule, lig: plf.Molecule) -> pd.DataFrame:
    """
    Generate a DataFrame with all interactions between protein and ligand.

    Args:
        prot: A protein molecule of type plf.Molecule.
        lig: A ligand molecule of type plf.Molecule.

    Returns:
        A DataFrame representing all interactions between the protein and ligand.
    """
    fp = plf.Fingerprint()
    fp.run_from_iterable(lig, prot)
    df = fp.to_dataframe()
    return df

def rmsd(mol1_o: Chem.Mol, mol2_o: Chem.Mol) -> float:
    """
    Calculate the RMSD between two molecules.

    Args:
        mol1: The first molecule.
        mol2: The second molecule.

    Returns:
        The RMSD value between the two molecules.
    """

    if mol1_o is None or mol2_o is None:
        return np.nan
    
    # mol1 = Chem.RemoveHs(mol1_o)
    # mol2 = Chem.RemoveHs(mol2_o)
    
    mol1_coords = mol1_o.GetConformer().GetPositions()
    mol2_coords = mol2_o.GetConformer().GetPositions()

    try:
        rmsd = np.sqrt(np.mean((mol1_coords - mol2_coords) ** 2))
    except Exception as e:
        print(f"Error calculating RMSD: {e}")
        rmsd = np.nan

    return rmsd


class PoseCheck(object):

    def __init__(self, docked_pt, docked_folder, protein_root_path = "./data/test_set", unpack = False, add_Hs = True, use_plf = True, save_to = ""):
        self.unpack = unpack
        self.use_plf = use_plf
        self.docked_pt_path = docked_pt
        self.docked_folder_path = docked_folder
        self.data_dict_generated = defaultdict(list)
        self.data_dict_docked = defaultdict(list)
        self.protein_root_path = protein_root_path
        self.addHs = add_Hs
        self.save_to = save_to
        self.initialize_dict()
    
    def initialize_dict(self):
        if self.unpack:
            load = torch.load(self.docked_pt_path)
            # Enable this line for crossdocked reference result
            # load = [[v] for v in load]
            docked_results = [r for pr in load for r in pr]
        else:
            docked_results = torch.load(self.docked_pt_path)["all_results"]
        # prolif protein and Chem.Mol ligands
        for i in tqdm(range(len(docked_results)), desc="Processing docked results"):
            protein_path = os.path.join(
                self.protein_root_path,
                os.path.dirname(docked_results[i]["ligand_filename"]),
                os.path.basename(docked_results[i]["ligand_filename"])[:10] + '_tmp.pdb'
            )
            mol_gen = docked_results[i]['mol']
            # mol_gen = Chem.MolFromSmiles(docked_results[i]['smiles'])
            # mol_gen = Chem.AddHs(mol_gen)
            mol_docked = get_pdbqt_mol(docked_results[i]["vina"]["dock"][0]["pose"])


            if mol_docked is None or mol_gen is None:
                print("Error: No molecule loaded.")
                continue  # or return, depending on your loop or function structure

            smiles_gen = Chem.MolToSmiles(mol_gen)
            smiles_docked = Chem.MolToSmiles(mol_docked)

            if not self.use_plf:
                
                # mol_gen = Chem.RemoveHs(mol_gen)
                # mol_docked = Chem.RemoveHs(mol_docked)

                Chem.SanitizeMol(mol_gen)
                Chem.SanitizeMol(mol_docked)

                mol_gen = Chem.AddHs(mol_gen)
                mol_docked = Chem.AddHs(mol_docked)

                self.data_dict_generated[protein_path].append(mol_gen)
                self.data_dict_docked[protein_path].append(mol_docked)
            else:

                mol_gen_added_H = dm.add_hs(mol_gen, add_coords=True)
                mol_docked_added_H = dm.add_hs(mol_docked, add_coords=True)

                generated_tmp_path = self.docked_folder_path + f"_generated_tmp_{smiles_gen}.sdf"
                docked_tmp_path = self.docked_folder_path + f"_docked_tmp_{smiles_docked}.sdf"

                dm.to_sdf(mol_gen_added_H, generated_tmp_path)
                dm.to_sdf(mol_docked_added_H, docked_tmp_path)

                try:
                    lig_gen = load_mols_from_sdf(generated_tmp_path)
                    lig_docked = load_mols_from_sdf(docked_tmp_path)


                    self.data_dict_generated[protein_path].append(lig_gen)
                    self.data_dict_docked[protein_path].append(lig_docked)

                except Exception as e:
                    print(e)

                os.remove(generated_tmp_path)
                os.remove(docked_tmp_path)

    def calculate_clashes(self):
        # Load in Chem.Mol and Chem.Mol
        all_clashes_list_generated = []
        all_clashes_list_docked = []

        for protein_path, lig_list in tqdm(self.data_dict_generated.items(), desc="Calculating clahses for protein-ligand pairs"):
            
            prot = load_protein_chemmol(protein_path)

            for lig in lig_list:
                all_clashes_list_generated.append(count_clashes(prot, lig))

            for lig in self.data_dict_docked[protein_path]:
                all_clashes_list_docked.append(count_clashes(prot, lig))

        all_clashes_list_generated = [x for x in all_clashes_list_generated if not np.isnan(x)]
        all_clashes_list_docked = [x for x in all_clashes_list_docked if not np.isnan(x)]
                
        self.generated_clashes_list = all_clashes_list_generated
        self.docekd_clashes_list = all_clashes_list_docked

        torch.save({"generated": all_clashes_list_generated,
                    "docked": all_clashes_list_docked}
                    , os.path.join(self.save_to, f"all_clashes_plf_{self.use_plf}.pt"))

    def calculate_strain_energy(self, start_id = 0, end_id = 100):
        # Load in Chem Mol
        all_strain_list_generated = []
        all_strain_list_docked = []

        sorted_items = sorted(self.data_dict_generated.items())[start_id:end_id]
        print(f"Now calculating strain energy for id {start_id} to {end_id}")

        for protein_path, lig_list in tqdm(dict(sorted_items).items(), desc="Calculating strain energies for ligands"):

            # prot = load_protein_prolif(protein_path)

            for lig in lig_list:
                all_strain_list_generated.append(calculate_strain_energy(lig))

            for lig in self.data_dict_docked[protein_path]:
                all_strain_list_docked.append(calculate_strain_energy(lig))

        filtered_list_gen = [x for x in all_strain_list_generated if x is not None]
        filtered_list_dock = [x for x in all_strain_list_docked if x is not None]

        self.generated_strain_list = filtered_list_gen
        self.docekd_strain_list = filtered_list_dock

        torch.save({"generated": filtered_list_gen,
                    "docked": filtered_list_dock}, 
                    os.path.join(self.docked_folder_path, f"posecheck_eval/all_strain_{start_id}-{end_id}_plf_{self.use_plf}.pt"))

    def calculate_interactions(self):
        # Load in plf.molecule and plf.molecule
        all_interactions_list_generated = []
        all_interactions_list_docked = []

        for protein_path, lig_list in tqdm(self.data_dict_generated.items(), desc="Calculating interactions for protein-ligand pairs"):

            prot = load_protein_prolif(protein_path)

            lig_list_docked = self.data_dict_docked[protein_path]
            for i, lig in enumerate(lig_list):
                try:
                    interactions_df_gen = generate_interaction_df(prot, lig)
                    interactions_df_docked = generate_interaction_df(prot, lig_list[i])
                except:
                    print(f"Interaction df generation failed for {i}")
                    continue

                print(interactions_df_gen)
                print(interactions_df_docked)
                return 
                all_interactions_list_generated.append(interactions_df_gen)
                all_interactions_list_docked.append(interactions_df_docked)

            # for lig in self.data_dict_docked[protein_path]:
                # all_interactions_list_docked.append(generate_interaction_df(prot, lig))
                
        self.generated_strain_list = all_interactions_list_generated
        self.docekd_strain_list = all_interactions_list_docked

        torch.save(all_interactions_list_generated, os.path.join(self.docked_pt_path, "posecheck_eval/all_strain_list_generated.pt"))
        torch.save(all_interactions_list_docked, os.path.join(self.docked_pt_path, "posecheck_eval/all_strain_list_docked.pt"))


    def calculate_rmsd(self):
        rmsd_list = []
        
        for protein_path, lig_list in tqdm(self.data_dict_generated.items(), desc="Calculating rmsd for ligands"):
            
            lig_list_docked = self.data_dict_docked[protein_path]
            print(len(lig_list), len(lig_list_docked))
            for i in range(len(lig_list)):
                calculated_rmsd = rmsd(lig_list[i], lig_list_docked[i])
                if not np.isnan(calculated_rmsd):
                    rmsd_list.append(calculated_rmsd)

        self.all_rmsd_list = rmsd_list
        torch.save(rmsd_list, os.path.join(self.save_to, f"rmsd_plf_{self.use_plf}.pt"))
        

if __name__ == "__main__":
    # Generate tmp files for proteins, only need to run once
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/data/yuejian/JCIM/targetdiff/multiconstraints_clsgds-50.0-c-14.0-cqed1.0-csa2.0-L1/eval_results/metrics_-1.pt")
    parser.add_argument('--unpack', type=bool, default=False)
    parser.add_argument('--use_plf', type=bool, default=True)
    parser.add_argument('--prepare_protein',type=bool,default=False)
    args = parser.parse_args()

    if args.prepare_protein:
        print("first time run, preparing protein!")
        file_list = glob.glob("./data/test_set/*")

        for protein_folder in file_list:

            files = glob.glob(os.path.join(protein_folder, "*"))
            # print(files)
            for file in files:
                if file.endswith(".pdb") and file[-7:-4] != "tmp":
                    pdb_path = file
                    # print(f"Now running tmp file for {pdb_path}")
                    tmp_path = pdb_path.split(".pdb")[0] + "_tmp.pdb"

                    # Call reduce to make tmp PDB with waters
                    print(tmp_path)
                    reduce_command = f"{REDUCE_PATH} -NOFLIP  {pdb_path} -Quiet > {tmp_path}"
                    subprocess.run(reduce_command, shell=True)
    
    savedir = os.path.dirname(args.path)
    docked_pt_folder_path = "benchmark/" + args.path.strip().split("/")[1]
    "benchmark/guided_target"
    

    os.makedirs(docked_pt_folder_path + "/posecheck_eval", exist_ok=True)
    pc = PoseCheck(docked_pt=args.path, docked_folder=docked_pt_folder_path, unpack=args.unpack, use_plf = args.use_plf, save_to = savedir)

    # less than 5 m
    pc.calculate_rmsd()
    print(f"The length of rmsd is {len(pc.all_rmsd_list)}")
    print(f"The median and mean of rmsd is {np.median(pc.all_rmsd_list)}, {np.mean(pc.all_rmsd_list)}")

    # 3-4h for 100*100
    pc.calculate_clashes()
    print(f"The length of generated and docked clashes lists are {len(pc.generated_clashes_list)}, {len(pc.docekd_clashes_list)}")
    print(f"The median and mean of generated clashes is {np.median(pc.generated_clashes_list)}, {np.mean(pc.generated_clashes_list)}")
    print(f"The median and mean of docked clashes is {np.median(pc.docekd_clashes_list)}, {np.mean(pc.docekd_clashes_list)}")

