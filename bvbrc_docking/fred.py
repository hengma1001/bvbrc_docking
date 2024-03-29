import glob
import os
import shutil
from typing import Optional

import MDAnalysis as mda
from openeye import oechem

from bvbrc_docking.utils import clean_pdb, run_and_save


def oe_convert(input_file, output_file):
    ifs = oechem.oemolistream()
    ifs.open(input_file)

    ofs = oechem.oemolostream()
    ofs.open(output_file)
    for mol in ifs.GetOEMols():
        oechem.OEWriteMolecule(ofs, mol)


def pdb_split(pdb_file):
    fp = open(pdb_file, "r")
    file_name, file_ext = os.path.splitext(pdb_file)
    i = 0
    output_pdbs = []
    for line in fp.readlines():
        if line.startswith("COMPND"):
            # lig_name = line.split()[1]
            file_output = f"{file_name}_{i}{file_ext}"
            output_pdbs.append(file_output)
            i += 1
            fo = open(file_output, "w")
            fo.write(line)
        elif line.startswith("END"):
            fo.write(line)
            fo.close()
        else:
            fo.write(line)

    return output_pdbs


class fred_dock(object):
    """_summary_

    Parameters
    ----------
    receptor_pdb : string
        Path name for the receptor pdb file
    drug_dbs : string
        Path name for the drug database smile file
    """

    def __init__(
        self,
        receptor_pdb,
        drug_dbs,
        n_cpus: int = 1,
        output_dir="./",
        fred_path="",
        oe_license="",
        hitlist_size: int = 0,
        **kwargs,
    ):
        self.drug_dbs = os.path.abspath(drug_dbs)
        self.n_cpus = int(n_cpus)
        self.fred_path = "" if fred_path is None else fred_path
        if oe_license is not None:
            os.environ["OE_LICENSE"] = oe_license
        self.hitlist_size = hitlist_size
        self.label = os.path.basename(receptor_pdb).split(".")[0]
        self.output_dir = output_dir
        self.run_dir = os.path.abspath(f"{self.output_dir}/run_{self.label}")
        os.makedirs(self.run_dir)

        self.receptor_pdb = f"{self.run_dir}/{os.path.basename(receptor_pdb)}"
        shutil.copy2(receptor_pdb, self.receptor_pdb)
        log_file = f"{self.run_dir}/dock_log"
        self.log_handle = open(log_file, "w")

    def find_pocket(self):
        fpocket_cmd = f"fpocket -f {self.receptor_pdb}"
        run_and_save(fpocket_cmd, cwd=self.run_dir, output_file=self.log_handle)

        pocket_pdbs = glob.glob(
            f"{self.run_dir}/{self.label}_out/pockets/pocket*_atm.pdb"
        )
        reslist = []
        for pdb in pocket_pdbs:
            pocket_u = mda.Universe(pdb)

            for res in pocket_u.residues:
                reslist += [f"{res.resname}:{res.resnum}: :{res.atoms[0].chainID}"]
        return max(reslist, key=reslist.count)

    def prepare_receptor(self):
        if self.receptor_pdb.endswith("oedu"):
            self.oe_receptor = self.receptor_pdb
        else:
            spruce_cmd = f"{self.fred_path}/spruce -in {self.receptor_pdb}"
            run_and_save(spruce_cmd, cwd=self.run_dir, output_file=self.log_handle)

            spruce_out = glob.glob(f"{self.run_dir}/{self.label.upper()}*.oedu")
            if spruce_out == []:
                reslist = self.find_pocket()
                spruce_cmd = f'{self.fred_path}/spruce -site_residue "{reslist}" -in {self.receptor_pdb}'
                run_and_save(spruce_cmd, cwd=self.run_dir, output_file=self.log_handle)

                spruce_out = glob.glob(f"{self.run_dir}/{self.label.upper()}*.oedu")
                if spruce_out == []:
                    raise BaseException(
                        f"spruce run failed. No DU found in {self.run_dir}"
                    )

            self.oe_receptor = f"{self.run_dir}/{self.label}.oedu"
            MKreceptor_cmd = f"{self.fred_path}/receptorindu -in {spruce_out[0]} -out {self.oe_receptor}"
            run_and_save(MKreceptor_cmd, cwd=self.run_dir, output_file=self.log_handle)

    def prepare_lig(self):
        if self.drug_dbs.endswith("oeb.gz"):
            self.oe_dbs = self.drug_dbs
        else:
            self.oe_dbs = f"{self.run_dir}/{self.label}.oeb.gz"
            omega_exe = f"{self.fred_path}/oeomega classic"
            if self.n_cpus > 1:
                omega_exe += f" -mpi_np {self.n_cpus}"
            MKlig_cmd = (
                f"{omega_exe} -in {self.drug_dbs} -out {self.oe_dbs}  -useGPU false"
            )
            run_and_save(MKlig_cmd, cwd=self.run_dir, output_file=self.log_handle)

    def run_fred(self):
        self.oe_docked = f"{self.run_dir}/{self.label}_docked.oeb.gz"
        fred_exec = f"{self.fred_path}/fred"
        if self.n_cpus > 1:
            fred_exec += f" -mpi_np {self.n_cpus}"
        fred_cmd = (
            f"{fred_exec} -receptor {self.oe_receptor} "
            f"-dbase {self.oe_dbs} -docked_molecule_file {self.oe_docked} "
            f"-hitlist_size {self.hitlist_size}"
        )
        run_and_save(fred_cmd, cwd=self.run_dir, output_file=self.log_handle)

    def prepare_report(self):
        report_cmd = (
            f"{self.fred_path}/docking_report -docked_poses {self.oe_docked} "
            f"-receptor {self.oe_receptor} -report_file {self.run_dir}/{self.label}.pdf"
        )
        run_and_save(report_cmd, cwd=self.run_dir, output_file=self.log_handle)

    def prepare_output(self):
        output_pdb = f"{self.run_dir}/{self.label}_ligs.pdb"
        oe_convert(self.oe_docked, output_pdb)
        lig_pdbs = pdb_split(output_pdb)

        pro_u = mda.Universe(self.receptor_pdb)
        proteins = pro_u.select_atoms("protein")
        for i, lig_pdb in enumerate(lig_pdbs):
            lig_u = mda.Universe(lig_pdb)
            comp_u = mda.Merge(proteins.atoms, lig_u.atoms)
            save_pdb = f"{self.run_dir}/{self.label}_{i}.pdb"
            comp_u.atoms.write(save_pdb)

    def run(self):
        self.prepare_receptor()
        self.prepare_lig()

        self.run_fred()

        self.prepare_report()
        if self.receptor_pdb.endswith("pdb"):
            self.prepare_output()

        self.log_handle.close()
