"""Sequential, resumable driver for the 80 time-budget comparison runs."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'results/timed_codebooks_80runs/protocol.json'


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main(args):
    if args.budget<=0 or args.interval<=0 or args.interval>args.budget:
        raise ValueError('Require 0 < interval <= budget')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.quant_sets))!=len(args.quant_sets):
        raise ValueError('Seeds and codebooks must be unique')
    protocol=json.loads(PROTOCOL.read_text())
    order=[x for x in protocol['run_order'] if x['seed'] in args.seeds and x['quant_set'] in args.quant_sets]
    jobs=[]
    for item in order:
        jobs.append({**item,'command':[sys.executable,'-u',str(Path(__file__).with_name('timed_run.py')),
            '--method',item['method'],'--quant-set',str(item['quant_set']),'--seed',str(item['seed']),
            '--budget',str(args.budget),'--interval',str(args.interval),
            '--data-root',str(args.data_root.resolve()),'--output',str(args.output.resolve()/'runs'/item['run_name'])]})
    if args.dry_run:
        print(json.dumps({'runs':len(jobs),'net_training_hours':len(jobs)*args.budget/3600,'jobs':jobs},indent=2))
        return 0
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    data=args.data_root.resolve()/'cifar-10-batches-py'
    files=list(Path(__file__).resolve().parent.glob('*.py'))+list((ROOT/'lesserdnn_cifar10_ablation_200').rglob('*.py'))
    files+=list((ROOT/'lesserdnn_cifar10_rigorous_eval').glob('*.py'))+[PROTOCOL]
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    dataset={f'data_batch_{i}':hashlib.sha256((data/f'data_batch_{i}').read_bytes()).hexdigest() for i in range(1,6)}
    config={'run_order':order,'budget_s':args.budget,'interval_s':args.interval,
            'data_root':str(args.data_root.resolve()),'source_sha256':frozen,'dataset_sha256':dataset}
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'manifest.json').exists():
            assert json.loads((root/'manifest.json').read_text())['configuration']==config,'Queue configuration changed'
        else:atomic(root/'manifest.json',{'configuration':config})
        child=None;stopped=False
        def interrupt(signum,frame):
            nonlocal stopped
            stopped=True
            if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        def status(value,**extra):
            atomic(root/'queue_status.json',{'status':value,'planned_runs':len(jobs),
                'updated_at':datetime.now().astimezone().isoformat(),'queue_pid':os.getpid(),**extra})
        for number,job in enumerate(jobs,1):
            if stopped or (root/'stop_after_current.json').exists():
                status('interrupted',next_run=job['run_name']);return 2
            folder=root/'runs'/job['run_name']
            if (folder/'latest.json').exists():
                previous=json.loads((folder/'latest.json').read_text())
                if previous['status']=='complete':continue
                if previous['status']=='early_stop':
                    status('needs_attention',run=job['run_name']);return 2
            for name,digest in frozen.items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Source changed: {name}'
            for name,digest in dataset.items():
                assert hashlib.sha256((data/name).read_bytes()).hexdigest()==digest,f'Dataset changed: {name}'
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'console.log').open('a') as log:
                child=subprocess.Popen(job['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status('running',run_index=number,run=job['run_name'],child_pid=child.pid)
                code=child.wait()
            child=None
            if code:
                status('interrupted' if stopped else 'failed',run=job['run_name'],returncode=code);return code
            latest=json.loads((folder/'latest.json').read_text())
            peer_method='gradient' if job['method']=='sa' else 'sa'
            peer=root/'runs'/f'{peer_method}_q{job["quant_set"]}_seed{job["seed"]}'
            if (peer/'latest.json').exists():
                other=json.loads((peer/'latest.json').read_text())
                assert latest['initial_weights_hash']==other['initial_weights_hash']
                assert latest['split_hash']==other['split_hash']
                first=json.loads((folder/'trace.json').read_text())[0]
                second=json.loads((peer/'trace.json').read_text())[0]
                assert first['val_accuracy']==second['val_accuracy'] and first['val_loss']==second['val_loss']
            if latest['status']!='complete':
                status('interrupted' if stopped else 'needs_attention',run=job['run_name']);return 2
        status('complete',completed_runs=len(jobs))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/timed_codebooks_80runs')
    parser.add_argument('--seeds',nargs='+',type=int,choices=range(10),default=list(range(10)))
    parser.add_argument('--quant-sets',nargs='+',type=int,choices=[8,9,10,11],default=[8,9,10,11])
    parser.add_argument('--budget',type=float,default=1800.)
    parser.add_argument('--interval',type=float,default=300.)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:raise SystemExit(main(args))
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic(args.output/'queue_status.json',{'status':'failed','error':str(error),'queue_pid':os.getpid()})
        raise
