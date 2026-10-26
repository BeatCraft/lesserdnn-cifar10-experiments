"""Short native CLI runs checking warm-up restoration and timed checkpoint resume."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE=Path(__file__).resolve().parent


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with (HERE.parent/'results/timed_codebooks_80runs/per_run_results.csv').open() as stream:
        recorded=list(csv.DictReader(stream))
    results=[]
    for quant in [8,9,10,11]:
        for method in ['gradient','sa']:
            folder=args.output/f'{method}_q{quant}'
            folder.mkdir()
            command=[sys.executable,'-u',str(HERE/'timed_run.py'),'--method',method,'--quant-set',str(quant),
                '--seed','0','--budget',str(.4 if method=='gradient' else 12.),'--interval','.1',
                '--warmup-steps','2','--data-root',str(args.data_root),'--output',str(folder)]
            with (folder/'console.log').open('w') as log:
                subprocess.run(command+['--stop-after-checkpoint','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
                first=json.loads((folder/'latest.json').read_text())
                assert first['status']=='running'
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            last=json.loads((folder/'latest.json').read_text())
            trace=json.loads((folder/'trace.json').read_text())
            expected=next(row for row in recorded if row['method']==method and int(row['quant_set'])==quant and int(row['seed'])==0)
            assert last['status']=='complete' and np.isfinite(last['val_loss'])
            assert last['checkpoint_sequence']>first['checkpoint_sequence']
            assert last['initial_weights_hash']==expected['initial_weights_hash']
            assert trace[0]['val_loss']==float(expected['initial_val_loss'])
            result={'quant_set':quant,'method':method,'status':'passed','resume_verified':True,
                    'warmup_initial_weights_and_validation_exact':True}
            results.append(result);print(json.dumps(result),flush=True)
    (args.output/'verification.json').write_text(json.dumps({'status':'passed','results':results},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=HERE.parent/'data')
    parser.add_argument('--output',type=Path,default=HERE.parent/'runs/timed_runner_verification')
    main(parser.parse_args())
