from typing import Literal, Optional, Union

from pydantic import Field

from bvbrc_docking.utils import BaseModel


class fredConfig(BaseModel):
    name: Literal["fred"] = "fred"
    receptor_pdb: Optional[str] = Field(
        ..., description="pdb path for the input protein"
    )
    drug_dbs: str = Field(..., description="smi path for the ligands")
    n_cpus: int = Field(..., description="number of MPI threads")
    fred_path: Optional[str] = Field(..., description="path to fred exec")
    oe_license: Optional[str] = Field(..., description="openeye license")
    hitlist_size: Optional[int] = Field(0, description="Number of top scoring mols")


class fredPartialConfig(BaseModel):
    """used for colmena runs"""

    name: Literal["fred_partial"] = "fred_partial"
    drug_dbs: str = Field(..., description="smi path for the ligands")
    n_cpus: int = Field(..., description="number of MPI threads")
    fred_path: Optional[str] = Field(..., description="path to fred exec")
    oe_license: Optional[str] = Field(..., description="openeye license")
    hitlist_size: Optional[int] = Field(0, description="Number of top scoring mols")


class DiffDockConfig(BaseModel):
    name: Literal["diffdock"] = "diffdock"
    receptor_pdb: str = Field(..., description="pdb path for the input protein")
    drug_dbs: str = Field(..., description="smi path for the ligands")
    diffdock_dir: str = Field(..., description="installed path for diffdock")
    output_dir: str = Field(..., description="output path for the docking result")
    top_n: Optional[int] = Field(
        ...,
        description="number of top N candidates for each protein-ligand pair",
    )


class DiffDock11Config(BaseModel):
    name: Literal["diffdock_1_1"] = "diffdock_1_1"
    receptor_pdb: str = Field(..., description="pdb path for the input protein")
    drug_dbs: str = Field(..., description="smi path for the ligands")
    diffdock_dir: str = Field(..., description="installed path for diffdock")
    output_dir: str = Field(..., description="output path for the docking result")
    top_n: Optional[int] = Field(
        ...,
        description="number of top N candidates for each protein-ligand pair",
    )
    batch_size: Optional[int] = Field(..., description="diffdock batch size")
    num_gnina: Optional[int] = Field(
        ..., description="number of gnina to run in parallel"
    )


class DiffDockPartialConfig(BaseModel):
    """used for colmena runs"""

    name: Literal["diffdock_partial"] = "diffdock_partial"
    drug_dbs: str = Field(..., description="smi path for the ligands")
    diffdock_dir: str = Field(..., description="installed path for diffdock")
    # output_dir: str = Field(..., description="output path for the docking result")
    top_n: Optional[int] = Field(
        ...,
        description="number of top N candidates for each protein-ligand pair",
    )


DockConfig = Union[
    fredConfig,
    DiffDockConfig,
    DiffDock11Config,
    fredPartialConfig,
    DiffDockPartialConfig,
]
