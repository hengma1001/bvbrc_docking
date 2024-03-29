import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import MDAnalysis as mda
import yaml
from pydantic import BaseModel as _BaseModel
from pydantic import validator

T = TypeVar("T")
PathLike = Union[str, Path]


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    """Check if a path exists (implements path_validator)."""
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _resolve_mkdir(value: Path) -> Path:
    """Create a directory if it does not exist (implements mkdir_validator)."""
    p = value.resolve()
    p.mkdir(exist_ok=False, parents=True)
    return p.absolute()


def path_validator(field: str) -> classmethod:
    """Pydantic validator to check if a path exists."""
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


def mkdir_validator(field: str) -> classmethod:
    """Pydantic validator to create a directory if it does not exist."""
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_mkdir)
    return _validator


class BaseModel(_BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models"""

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, "w") as fp:
            json.dump(self.dict(), fp, indent=2)

    @classmethod
    def from_json(cls: Type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseModel instance.
        """
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseModel instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore


three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "SEC": "U",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
    "XLE": "J",
}


def build_logger(debug=0):
    logger_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logger_level, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)
    return logger


def run_and_save(cmd, cwd=None, output_file=None):
    print(cmd, file=output_file)
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=output_file,
        stderr=subprocess.STDOUT,
    )
    process.wait()
    return process


def pdb2seq(pdb_file):
    mda_u = mda.Universe(pdb_file)
    protein = mda_u.select_atoms("protein")
    seq = ""
    for res in protein.residues:
        if res.resname in three_to_one:
            seq += three_to_one[res.resname]
        else:
            seq += "-"
    return seq


def get_pdblabel(pdb_file) -> str:
    return os.path.basename(pdb_file)[:-4]


def comb_pdb(prot_pdb, lig_pdb, comp_pdb=None):
    prot_u = mda.Universe(prot_pdb)
    lig_u = mda.Universe(lig_pdb)
    merged = mda.Merge(prot_u.atoms, lig_u.atoms)
    if comp_pdb is None:
        comp_pdb = f"{os.path.dirname(lig_pdb)}/{get_pdblabel(prot_pdb)}_{get_pdblabel(lig_pdb)}.pdb"
    merged.atoms.write(comp_pdb)
    return comp_pdb


def sdf2pdb(sdf_file, pdb_file=None):
    if pdb_file is None:
        pdb_file = sdf_file[:-3] + "pdb"
    cmd = f"obabel -isdf {shlex.quote(sdf_file)} -opdb > {shlex.quote(pdb_file)}"
    p = run_and_save(cmd)
    p.wait()
    return pdb_file


def clean_pdb(pdb_file, output_pdb: str) -> str:
    mda_u = mda.Universe(pdb_file)
    protein = mda_u.select_atoms("protein")
    protein.write(output_pdb)
    return output_pdb
