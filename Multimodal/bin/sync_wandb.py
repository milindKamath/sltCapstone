import os, sys, time, argparse
import subprocess
import pdb

import config as conf

def sync( out ):
    dir = [ os.path.join( f'{out}/wandb', f ) for f in os.listdir( f'{out}/wandb' ) if os.path.isdir( os.path.join( f'{out}/wandb', f ) ) ][0]
    env = '/'.join( sys.executable.split('/')[:-1] )
    print( f'Syncing dir: {dir}...')
    res = subprocess.run( f'{env}/wandb sync {dir}', shell=True, capture_output=True )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dir', help='expt dir', \
                         default=conf.dir
                       )
    opt = parser.parse_args()
    sync( opt.dir )

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print( f'Finished in {round( end-start, 2 ) } second(s) or {round( (end-start)/60, 2 ) } min(s).' )
    sys.exit(0)