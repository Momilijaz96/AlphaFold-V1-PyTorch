from Bio.PDB import *
import tensorflow as tf
import numpy as np

RESIDUE_ONE_LETTER = {
    'ALA':'A',
    'ARG':'R',
    'ASN':'N',
    'ASP':'D',
    'CYS':'C',
    'GLN':'Q',
    'GLU':'E',
    'GLY':'G',
    'HIS':'H',
    'ILE':'I',
    'LEU':'L',
    'LYS':'K',
    'MET':'M',
    'PHE':'F',
    'PRO':'P',
    'SER':'S',
    'THR':'T',
    'TRP':'W',
    'TYR':'Y',
    'VAL':'V',
    'XAA':'X',
}

class FeatureConstructor():
    def __init__(self, dataset, pdb_file):
        self.dataset = dataset
        parser = PDBParser(PERMISSIVE = True, QUIET = True) 
        self.pdb_data = parser.get_structure(self.dataset.domain, pdb_file)
        
        residues = [r.resname for r in self.pdb_data.get_residues()]
        self.sequence = ''.join([RESIDUE_ONE_LETTER[name] for name in residues])
        self.sequence_length = len(self.sequence)

    def add_domain_name(self):
        self.dataset.add_feature('domain_name', self.dataset.domain)

    def add_sequence(self):
        self.dataset.add_feature('sequence', self.sequence)

    def add_sequence_length(self):
        sequence_length_tf = tf.constant(self.sequence_length, dtype=tf.int64, shape=[1, 1])
        sequence_length_tf = tf.tile(sequence_length_tf, [self.sequence_length, 1])
        self.dataset.add_feature('sequence_length', sequence_length_tf)

    def add_aatype(self):
        """Maps the given sequence into a one-hot encoded matrix."""
        mapping = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX')}
        num_entries = max(mapping.values()) + 1
        one_hot_arr = np.zeros((self.sequence_length, num_entries), dtype=np.float32)

        for aa_index, aa_type in enumerate(self.sequence):
            aa_id = mapping[aa_type]
            one_hot_arr[aa_index, aa_id] = 1

        self.dataset.add_feature('aatype', one_hot_arr)

    def add_chain_name(self):
        chains = [c.id for c in self.pdb_data.get_chains()]
        assert len(chains) == 1
        self.dataset.add_feature('chain_name', chains[0])

    def get_atom_pos(self, atom_type):
        atom_mask = np.zeros((self.sequence_length), dtype=np.int64)
        atom_positions = np.zeros((self.sequence_length, 3), dtype=np.int64)
        
        for index, residue in enumerate(self.pdb_data.get_residues()):
            atoms = [a for a in residue.get_atoms() if a.name == atom_type]
            if len(atoms) == 0:
                continue
            assert len(atoms) == 1
            atom_mask[index] = 1
            pos = np.array([p for p in atoms[0].get_vector()])
            atom_positions[index] = pos

        return atom_positions, atom_mask

    def add_alpha_positions(self):
        alpha_mask, alpha_positions = self.get_atom_pos('CA')

        self.dataset.add_feature('alpha_mask', alpha_mask)
        self.dataset.add_feature('alpha_positions', alpha_positions)

    def add_beta_positions(self):
        beta_mask, beta_positions = self.get_atom_pos('CB')

        self.dataset.add_feature('beta_mask', beta_mask)
        self.dataset.add_feature('beta_positions', beta_positions)

    def add_phi_psi_angles(self): 
        phi_mask = np.zeros((self.sequence_length), dtype=np.int64)
        phi_angles = np.zeros((self.sequence_length), dtype=np.float32)
        psi_mask = np.zeros((self.sequence_length), dtype=np.int64)
        psi_angles = np.zeros((self.sequence_length), dtype=np.float32)

        angles = []
        ppb=PPBuilder()
        for pp in ppb.build_peptides(self.pdb_data):
            angles += pp.get_phi_psi_list()

        assert len(angles) == self.sequence_length

        for index, (phi, psi) in enumerate(angles):
            if phi is not None:
                phi_mask[index] = 1
                phi_angles[index] = phi
            if psi is not None:
                psi_mask[index] = 1
                psi_angles[index] = psi
        
        self.dataset.add_feature('psi_mask', psi_mask)
        self.dataset.add_feature('psi_angles', psi_angles)
        self.dataset.add_feature('phi_mask', phi_mask)
        self.dataset.add_feature('phi_angles', phi_angles)

    def add_placeholders(self):
        # We do not need to predict secondary structure
        # (This would be done for us by DSSP)
        self.dataset.add_feature('sec_structure', np.zeros((self.sequence_length, 8), dtype=np.int64))
        self.dataset.add_feature('sec_structure_mask', np.zeros((self.sequence_length), dtype=np.int64))
        
        # We do not need to predict solvent accessible area
        # (This would be done for us by DSSP)
        self.dataset.add_feature('solv_surf', np.zeros((self.sequence_length), dtype=np.float32))
        self.dataset.add_feature('solv_surf_mask', np.zeros((self.sequence_length), dtype=np.int64))
        
        # These values should be completely unused during both training and evaluation
        self.dataset.add_feature('superfamily', 'Unknown')
        self.dataset.add_feature('resolution', np.zeros((1), dtype=np.float32))
            
