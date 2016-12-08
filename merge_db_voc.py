import argparse
import sys
import os

import h5py

from vsim_common import load_vocabulary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('database')
    parser.add_argument('vocabulary')
    args = parser.parse_args()

    database_file = os.path.expanduser(args.database)
    voc_file = os.path.expanduser(args.vocabulary)

    with h5py.File(database_file, 'a') as dbf:
        if 'vocabulary' in dbf.keys():
            print('Database already has a vocabulary')
        else:
            vocabulary = load_vocabulary(voc_file)
            voc_size = vocabulary.shape[0]
            db_word_size = dbf[list(dbf.keys())[0]].size
            if not voc_size == db_word_size:
                print('Vocabulary size {:d} did not match database word vector length {:d}'.format(voc_size, db_word_size))
                sys.exit(-1)

            dbf['vocabulary'] = vocabulary
