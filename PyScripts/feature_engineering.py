import sys
import os.path
import os
import urllib.request
import tensorflow as tf
from feature_dataset import FeatureDataset
from feature_constructor import FeatureConstructor

pdb_dir = 'out/pdb'
tfrec_dir = 'out/tfrec'

def download_domain(domain):
    pdb_file = os.path.join(pdb_dir, domain) +  '.pdb'
    if os.path.isfile(pdb_file):
        return pdb_file

    url = f'http://www.cathdb.info/version/v4_1_0/api/rest/id/{domain}.pdb'
    print(f'Downloading {pdb_file} from {url}')
    urllib.request.urlretrieve(url, pdb_file)
    return pdb_file

def create_out_folders():
    dirs = [pdb_dir, tfrec_dir]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def domain_getter(filename):
    with open(filename, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if not line:
                continue

            yield line

def feature_engineer(domain_list_file):
    domain_count = sum(1 for _ in domain_getter(domain_list_file))
    print(f'Will process {domain_count} domain(s)')
    create_out_folders()
    for domain in domain_getter(domain_list_file) :
        pdb_file = download_domain(domain)
        dataset = FeatureDataset(domain)

        feature_constructor = FeatureConstructor(dataset, pdb_file)

        print(f'Sequence: {feature_constructor.sequence}')

        feature_constructor.add_domain_name()
        feature_constructor.add_sequence()
        feature_constructor.add_sequence_length()
        feature_constructor.add_aatype()
        feature_constructor.add_chain_name()
        feature_constructor.add_alpha_positions()
        feature_constructor.add_beta_positions()
        feature_constructor.add_phi_psi_angles()

        feature_constructor.add_placeholders()

        tfrec_file = os.path.join(tfrec_dir, domain) + '.tfrec'
        dataset.save_file(tfrec_file)

def main():
    if len(sys.argv) <= 1:
        domain_file = 'domains/single_domain.txt'
        print(f'No domain file given. Defaulting to: {domain_file}')
    else:
        domain_file = sys.argv[2]

    if not os.path.isfile(domain_file):
        print(f'File does not exist: {domain_file}')
        sys.exit(1)
        return

    feature_engineer(domain_file)
    sys.exit(0)

if __name__ == '__main__':
    main()
