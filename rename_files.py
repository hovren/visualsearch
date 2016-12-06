#!/usr/bin/env python3

import argparse
import os
import shutil
import datetime

date_format = '%Y%m%d_%H%M%S_000.jpg'
time_delta = datetime.timedelta(hours=-9) # From 18:00 -> 09:00 (GMT+00)
error_prefix = '20000101_'


def matching_files(files):
    return [f for f in files if f.startswith(error_prefix)]
    

def new_filename(fname):
    assert fname.startswith(error_prefix)
    dt = datetime.datetime.strptime(fname, date_format)
    fmt = "20161129_%H%M%S_000.jpg"
    return (dt + time_delta).strftime(fmt)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    
    directory = os.path.expanduser(args.directory)
    all_files = os.listdir(directory)
    mfiles = matching_files(all_files)
    
    if not mfiles:
        print('Nothing to rename')
    
    for f in sorted(mfiles):
        new_f = new_filename(f)
        p1 = os.path.join(directory, f)
        p2 = os.path.join(directory, new_f)
        if args.write:
            os.rename(p1, p2)
        print('{} -> {}'.format(p1, p2))
        
        
    if not args.write:
        print('Nothing written. To rename pass --write to script')                
        
    
