from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.feature.symmetry import get_symmetries
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.tokenizer import Tokenizer
from boltz.data.types import MSA, Connection, Input, Manifest, Record, Structure

import numpy as np
import random
import networkx as nx
from dataclasses import replace

# ----------------------------
# Helper Functions
# ----------------------------

def pairwise_distances(group1, group2):
    """
    Calculate the pairwise distances between two groups of atoms.

    Parameters:
        group1: numpy array of shape (n, 3) representing coordinates of group 1
        group2: numpy array of shape (m, 3) representing coordinates of group 2

    Returns:
        distances: numpy array of shape (n, m) with distances.
    """
    diff = group1[:, np.newaxis, :] - group2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distances


def get_contacts_dict(group1, group2, cutoff):
    """
    For each atom in group1, return a list of indices in group2 that are within the cutoff.

    Parameters:
        group1: numpy array (n, 3)
        group2: numpy array (m, 3)
        cutoff: float, distance threshold.

    Returns:
        A dict mapping each atom index in group1 to a list of indices (local to group2) within cutoff.
    """
    distances = pairwise_distances(group1, group2)
    within_cutoff = (distances < cutoff) & (distances > 0)
    contact_dictionary = {}
    for i in range(group1.shape[0]):
        contact_dictionary[i] = np.where(within_cutoff[i])[0].tolist()
    return contact_dictionary

def get_peptide_chain_id(sample):
    """
    Select at random one peptide chain (mol_type==0, valid==True, num_residues ≤30),
    and return its chain_id. Raises a ValueError if no such chain exists.
    """
    chains = sample.record.chains
    peptide_chains = [c for c in chains if c.mol_type == 0 and c.valid and c.num_residues <= 30]

    if not peptide_chains:
        raise ValueError(
            f"No valid protein chain with ≤30 residues in sample {sample.record.id}"
        )
    peptide_chain = random.choice(peptide_chains)

    return peptide_chain.chain_id

def get_chain_coordinates(structure, chain_id):
        """
        Return the global start index, atom count, and coordinate array for a given chain.
        """
        atom_start = int(structure.chains[chain_id]['atom_idx'])
        atom_count = int(structure.chains[chain_id]['atom_num'])
        coords = structure.atoms[atom_start : atom_start + atom_count]['coords']
        return atom_start, atom_count, coords

# ----------------------------
# Main Function: compute_connections
# ----------------------------

def compute_connections(sample, structure, cutoff_distance=3.5):
    """
    Gets information from the record and structure, computes the peptide-protein contacts,
    and returns a numpy array of connection tuples.

    The connection tuple has the following structure:
      (peptide_chain_id, protein_chain_id, peptide_residue, protein_residue, peptide_atom, protein_atom)

    Parameters:
        sample
        structure
        cutoff_distance: distance threshold for contacts.

    Returns:
        connections: numpy array with the specified Connection dtype.
    """
    peptide_chain_id = get_peptide_chain_id(sample)

    # Identify the protein chains that interact with the peptide from the manifest interfaces
    interfaces = sample.record.interfaces
    prot_chain_ids = [
        i.chain_2 if i.chain_1 == peptide_chain_id else i.chain_1
        for i in interfaces if i.valid and peptide_chain_id in (i.chain_1, i.chain_2)
    ]

    # Extract peptide atom coordinates
    peptide_start_idx, peptide_atom_count, peptide_coords = get_chain_coordinates(structure, peptide_chain_id)

    # Build protein coordinates and a mapping of local (concatenated) indices to global indices
    protein_coords_list = []
    protein_global_indices_list = []
    for prot_chain_id in prot_chain_ids:
        start_idx, atom_count, coords = get_chain_coordinates(structure, prot_chain_id)
        protein_coords_list.append(coords)
        global_indices = np.arange(start_idx, start_idx + atom_count)
        protein_global_indices_list.append(global_indices)

    all_protein_coords = np.concatenate(protein_coords_list, axis=0)
    all_protein_global_indices = np.concatenate(protein_global_indices_list, axis=0)

    # Compute the contacts between peptide and protein atoms (in local indices)
    local_contacts_dict = get_contacts_dict(peptide_coords, all_protein_coords, cutoff_distance)

    # in global indices
    global_contacts_dict = {
        peptide_start_idx + pep_local_idx: [all_protein_global_indices[prot_local_idx] for prot_local_idx in prot_local_idx_list]
        for pep_local_idx, prot_local_idx_list in local_contacts_dict.items()
    }

    # Build atom → residue and atom → chain mappings for later connection lookup
    atom2res = {}
    res_id = 0
    for res in structure.residues:
        for i in range(res['atom_num']):
            atom2res[res['atom_idx'] + i] = res_id
        res_id += 1

    atom2chain = {}
    for atom_idx in range(peptide_start_idx, peptide_start_idx + peptide_atom_count):
        atom2chain[atom_idx] = peptide_chain_id
    for prot_chain_id in prot_chain_ids:
        start_idx, atom_count, coords = get_chain_coordinates(structure, prot_chain_id)
        for atom_idx in range(start_idx, start_idx + atom_count):
            atom2chain[atom_idx] = prot_chain_id

    # Map each peptide residue to the list of its global atom indices
    peptide_residue_atoms = {
        res_id: list(range(res['atom_idx'], res['atom_idx'] + res['atom_num']))
        for res_id, res in enumerate(structure.residues)
        if atom2chain.get(res['atom_idx']) == peptide_chain_id
    }

    # Build the list of peptide‐residues that have at least one contacting atom
    contacted_peptide_residues = []
    for res_id, atoms in peptide_residue_atoms.items():
        contacted_atoms = [atom for atom in atoms if atom in global_contacts_dict]
        if contacted_atoms:
            contacted_peptide_residues.append((res_id, atoms, contacted_atoms))

    # randomly choose one peptide residue and its atoms in contact
    chosen_res_id, all_atoms, selected_atoms = random.choice(contacted_peptide_residues)
    fixed_contacts = {global_atom: global_contacts_dict[global_atom] for global_atom in selected_atoms}

    # Build the final connections list
    connections = [
        (
            peptide_chain_id,                # peptide chain
            atom2chain[prot_atom],           # protein chain
            atom2res[pep_atom],              # peptide residue
            atom2res[prot_atom],             # protein residue
            pep_atom,                        # peptide atom (global idx)
            prot_atom                        # protein atom (global idx)
        )
        for pep_atom, prot_list in fixed_contacts.items()
        for prot_atom in prot_list
    ]

    connections = np.array(connections, dtype=Connection)
    return connections

@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: float
    sampler: Sampler
    cropper: Cropper
    filters: Optional[list] = None
    split: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""

    datasets: list[DatasetConfig]
    filters: list[DynamicFilter]
    featurizer: BoltzFeaturizer
    tokenizer: Tokenizer
    max_atoms: int
    max_tokens: int
    max_seqs: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    random_seed: int
    pin_memory: bool
    symmetries: str
    atoms_per_window_queries: int
    min_dist: float
    max_dist: float
    num_bins: int
    overfit: Optional[int] = None
    pad_to_max_tokens: bool = False
    pad_to_max_atoms: bool = False
    pad_to_max_seqs: bool = False
    crop_validation: bool = False
    return_train_symmetries: bool = False
    return_val_symmetries: bool = True
    train_binder_pocket_conditioned_prop: float = 0.0
    val_binder_pocket_conditioned_prop: float = 0.0
    binder_pocket_cutoff: float = 6.0
    binder_pocket_sampling_geometric_p: float = 0.0
    val_batch_size: int = 1


@dataclass
class Dataset:
    """Data holder."""

    target_dir: Path
    msa_dir: Path
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: BoltzFeaturizer


def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(target_dir / "structures" / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1 and msa_id != "":
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class TrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        samples_per_epoch: int,
        symmetries: dict,
        max_atoms: int,
        max_tokens: int,
        max_seqs: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        return_symmetries: Optional[bool] = False,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.datasets = datasets
        self.probs = [d.prob for d in datasets]
        self.samples_per_epoch = samples_per_epoch
        self.symmetries = symmetries
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff
        self.binder_pocket_sampling_geometric_p = binder_pocket_sampling_geometric_p
        self.return_symmetries = return_symmetries
        self.samples = []
        for dataset in datasets:
            records = dataset.manifest.records
            if overfit is not None:
                records = records[:overfit]
            iterator = dataset.sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick a random dataset
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.probs,
        )
        dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset
        sample: Sample = next(self.samples[dataset_idx])

        # Get the structure
        try:
            input_data = load_input(sample.record, dataset.target_dir, dataset.msa_dir)
        except Exception as e:
            print(
                f"Failed to load input for {sample.record.id} with error {e}. Skipping."
            )
            return self.__getitem__(idx)

        connections = compute_connections(sample, input_data.structure)
        input_data = replace(
            input_data, structure=replace(input_data.structure, connections=connections))

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Compute crop
        try:
            if self.max_tokens is not None:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Compute features
        try:
            features = dataset.featurizer.process(
                tokenized,
                training=True,
                max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=self.binder_pocket_sampling_geometric_p,
            )
        except Exception as e:
            print(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class ValidationDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        seed: int,
        symmetries: dict,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_seqs: Optional[int] = None,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        crop_validation: bool = False,
        return_symmetries: Optional[bool] = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
    ) -> None:
        """Initialize the validation dataset."""
        super().__init__()
        self.datasets = datasets
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = seed
        self.symmetries = symmetries
        self.random = np.random if overfit else np.random.RandomState(self.seed)
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.overfit = overfit
        self.crop_validation = crop_validation
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick dataset based on idx
        for dataset in self.datasets:
            size = len(dataset.manifest.records)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if idx < size:
                break
            idx -= size

        # Get a sample from the dataset
        record = dataset.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, dataset.target_dir, dataset.msa_dir)
        except Exception as e:
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Compute crop
        try:
            if self.crop_validation and (self.max_tokens is not None):
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_tokens=self.max_tokens,
                    random=self.random,
                    max_atoms=self.max_atoms,
                )
        except Exception as e:
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Compute features
        try:
            pad_atoms = self.crop_validation and self.pad_to_max_atoms
            pad_tokens = self.crop_validation and self.pad_to_max_tokens

            features = dataset.featurizer.process(
                tokenized,
                training=False,
                max_atoms=self.max_atoms if pad_atoms else None,
                max_tokens=self.max_tokens if pad_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=1.0,  # this will only sample a single pocket token
                only_ligand_binder_pocket=True,
            )
        except Exception as e:
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        if self.overfit is not None:
            length = sum(len(d.manifest.records[: self.overfit]) for d in self.datasets)
        else:
            length = sum(len(d.manifest.records) for d in self.datasets)

        return length


class BoltzTrainingDataModule(pl.LightningDataModule):
    """DataModule for boltz."""

    def __init__(self, cfg: DataConfig) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg

        assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."

        # Load symmetries
        symmetries = get_symmetries(cfg.symmetries)

        # Load datasets
        train: list[Dataset] = []
        val: list[Dataset] = []

        for data_config in cfg.datasets:
            # Set target_dir
            target_dir = Path(data_config.target_dir)
            msa_dir = Path(data_config.msa_dir)

            # Load manifest
            if data_config.manifest_path is not None:
                path = Path(data_config.manifest_path)
            else:
                path = target_dir / "manifest.json"
            manifest: Manifest = Manifest.load(path)

            # Split records if given
            if data_config.split is not None:
                with Path(data_config.split).open("r") as f:
                    split = {x.lower() for x in f.read().splitlines()}

                train_records = []
                val_records = []
                for record in manifest.records:
                    if record.id.lower() in split:
                        val_records.append(record)
                    else:
                        train_records.append(record)
            else:
                train_records = manifest.records
                val_records = []

            # Filter training records
            train_records = [
                record
                for record in train_records
                if all(f.filter(record) for f in cfg.filters)
            ]
            # Filter training records
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]

            # Create train dataset
            train_manifest = Manifest(train_records)
            train.append(
                Dataset(
                    target_dir,
                    msa_dir,
                    train_manifest,
                    data_config.prob,
                    data_config.sampler,
                    data_config.cropper,
                    cfg.tokenizer,
                    cfg.featurizer,
                )
            )

            # Create validation dataset
            if val_records:
                val_manifest = Manifest(val_records)
                val.append(
                    Dataset(
                        target_dir,
                        msa_dir,
                        val_manifest,
                        data_config.prob,
                        data_config.sampler,
                        data_config.cropper,
                        cfg.tokenizer,
                        cfg.featurizer,
                    )
                )

        # Print dataset sizes
        for dataset in train:
            dataset: Dataset
            print(f"Training dataset size: {len(dataset.manifest.records)}")

        for dataset in val:
            dataset: Dataset
            print(f"Validation dataset size: {len(dataset.manifest.records)}")

        # Create wrapper datasets
        self._train_set = TrainingDataset(
            datasets=train,
            samples_per_epoch=cfg.samples_per_epoch,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            binder_pocket_conditioned_prop=cfg.train_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p=cfg.binder_pocket_sampling_geometric_p,
            return_symmetries=cfg.return_train_symmetries,
        )
        self._val_set = ValidationDataset(
            datasets=train if cfg.overfit is not None else val,
            seed=cfg.random_seed,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            crop_validation=cfg.crop_validation,
            return_symmetries=cfg.return_val_symmetries,
            binder_pocket_conditioned_prop=cfg.val_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )
