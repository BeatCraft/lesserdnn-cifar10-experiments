"""One timed strict-search/Gradient run for the 8/9/10/11-level comparison."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
import pickle
import platform
import signal
import time
from pathlib import Path

import numpy as np

import comparison_runtime as r
from batched_paper_cnn_metal import BatchedPaperCNNForward
from strict_search import PaperBatchedSA


def source_hashes():
    paths = list(Path(__file__).resolve().parent.glob('*.py')) + [
        r.p.SOURCE/'ldnn_cifar10_model.py', r.p.SOURCE/'train_ldnn_cifar10.py',
        r.p.SOURCE/'verify_pytorch_batch_ldnn.py',
        r.p.SOURCE/'experiments/ldnn_cifar10_native_same_sign/train.py',
        r.p.SOURCE/'ldnn_runtime/core.py', r.p.SOURCE/'ldnn_runtime/lmetal.py',
        r.p.RELEASE/'lesserdnn_cifar10_rigorous_eval/train_one.py']
    return {str(path.relative_to(r.p.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def environment(ctx):
    return {'platform':platform.platform(),'gpu':str(ctx.model._gpu.device.name()),
        'python':platform.python_version(),'torch_threads':r.torch.get_num_threads(),
        'dependencies':{name:importlib.metadata.version(name)
            for name in ['numpy','torch','torchvision','pyobjc-core','pyobjc-framework-Metal']}}


def gradient_step(ctx,state,shadow,iterator,seed):
    try:
        images,labels=next(iterator)
    except StopIteration:
        state['epoch']+=1
        state['batch_position']=0
        if state['epoch']>300:
            return iterator,False
        iterator=r.train_iterator(ctx,state,seed)
        images,labels=next(iterator)
    batch=r.p.gradients_for_batch(ctx.model,ctx.layers,images,labels,1.,ctx.buffers)
    r.p.update_weights(ctx.model,ctx.layers,shadow,batch[3],None,
        r.learning_rate(state['epoch']),0.,'single')
    assert all(np.isfinite(value).all() for value in shadow.values())
    state['optimizer_steps']+=1
    state['batch_position']+=1
    state['seen_train_samples']+=batch[2]
    state['train_loss_sum']+=batch[0]*batch[2]
    if state['optimizer_steps']%100==0:
        gc.collect()
    return iterator,True


def warmup(ctx,method,seed,quant,gradient_steps=20):
    indexes=r.model_indexes(ctx)
    rng=r.rng_state()
    before=time.perf_counter()
    if method=='sa':
        engine=PaperBatchedSA(ctx,seed,quant)
        engine.prepare_baseline()
        engine.step()
    else:
        ctx.model.set_backpropagation(True,0.)
        state={'epoch':1,'batch_position':0,'optimizer_steps':0,
               'seen_train_samples':0,'train_loss_sum':0.}
        shadow={layer._index:layer._weight_matrix.copy() for layer in ctx.layers}
        iterator=r.train_iterator(ctx,state,seed)
        for _ in range(gradient_steps):
            iterator,_=gradient_step(ctx,state,shadow,iterator,seed)
    r.load_model_indexes(ctx,indexes)
    r.restore_rng(rng)
    for key,value in r.model_indexes(ctx).items():
        np.testing.assert_array_equal(value,indexes[key])
    ctx.model.reset()
    return time.perf_counter()-before


def run(args):
    r.DATA_ROOT = args.data_root.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    config={'method':args.method,'quant_set':args.quant_set,'seed':args.seed,
        'budget_s':args.budget,'interval_s':args.interval,'model_weights':174944,
        'metal_execution':'batched_paper_cnn_forward_native_backward',
        'gradient_batches_per_epoch':350,'forward_chunk_size':128,
        'sa_mutation_count':8,'sa_max_candidates_per_search':100,
        'sa_acceptance':'strict_full_training_ce','source_sha256':source_hashes(),
        'warmup_gradient_steps':args.warmup_steps,'warmup_sa_candidates':1,
        'test_used':False,'version':2,
        'initialization':'independent_uniform_indexes_max_abs_0.125',
        'codebook':r.CODEBOOKS[args.quant_set].tolist(),
        'data_root':str(r.DATA_ROOT), 'git_commit':r.comparison_commit()}
    if (output/'config.json').exists():
        assert json.loads((output/'config.json').read_text())==config,'Run configuration/source changed'
    else:
        r.atomic_json(output/'config.json',config)
    started=time.perf_counter()
    ctx=r.initialize('gradient',args.quant_set,args.seed)
    optimized=BatchedPaperCNNForward(ctx).install()
    assert optimized is not None and ctx.model.count_weight()==174944
    initial_hash=r.p.digest({layer._index:layer._weight_matrix for layer in ctx.layers})
    split_hash=r.p.published.indexes_digest(ctx.tr)+':'+r.p.published.indexes_digest(ctx.va)
    np.savez(output/'split_indices.npz',train=ctx.tr,validation=ctx.va)
    if not (output/'initial_indexes.npz').exists():
        np.savez(output/'initial_indexes.npz',**{str(k):v for k,v in r.model_indexes(ctx).items()})
    r.atomic_json(output/'environment.json',environment(ctx))
    state={'epoch':1,'batch_position':0,'optimizer_steps':0,'seen_train_samples':0,
        'train_loss_sum':0.,'training_time_s':0.,'evaluation_time_s':0.,
        'save_time_s':0.,'warmup_time_s':0.,'setup_time_s':time.perf_counter()-started,
        'initial_weights_hash':initial_hash,'split_hash':split_hash,
        'next_target_s':0.,'checkpoint_sequence':0,'evaluation_forward_calls':0,
        'rows':[],'prior_wall_time_s':0.}
    shadow={layer._index:layer._weight_matrix.copy() for layer in ctx.layers} if args.method=='gradient' else None
    engine=None
    checkpoint=output/'latest.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as stream:
            saved=pickle.load(stream)
        if saved['state']['rows'][-1]['status'] in ('complete','early_stop'):
            print('already complete',flush=True)
            return
        state=saved['state']
        latest=json.loads((output/'latest.json').read_text())
        assert latest['checkpoint_sequence']==state['checkpoint_sequence']
        state['rows'][-1]=latest
        state['save_time_s']=latest['save_time_s']
        state['prior_wall_time_s']=latest['total_wall_time_s']
        state['setup_time_s']+=time.perf_counter()-started
        assert state['initial_weights_hash']==initial_hash and state['split_hash']==split_hash
        if args.method=='sa':
            engine=PaperBatchedSA(ctx,args.seed,args.quant_set,
                payload={'state':saved['sa_state'],'indexes':saved['indexes'],
                         'rng':saved['rng'],'cached_loss':saved['sa_cached_loss']})
            journal=output/'candidates.jsonl'
            if journal.exists():
                committed=[]
                for line in journal.read_text().splitlines():
                    try:
                        item=json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item['candidate_number']<=engine.state['num_proposals']:
                        committed.append(line)
                journal.write_text('\n'.join(committed)+('\n' if committed else ''))
        else:
            r.load_model_indexes(ctx,saved['indexes'])
            r.restore_rng(saved['rng'])
            shadow=saved['shadow']
    else:
        distribution={str(layer._index):{'mean':float(layer._weight_matrix.mean()),
            'variance':float(layer._weight_matrix.var()),
            'nonzero_fraction':float(np.count_nonzero(layer._weight_matrix)/layer._weight_matrix.size),
            'value_counts':{str(float(v)):int(np.count_nonzero(layer._weight_matrix==v)) for v in ctx.levels}}
            for layer in ctx.layers}
        r.atomic_json(output/'initialization.json',{'initial_weights_hash':initial_hash,
            'split_hash':split_hash,'distribution':distribution})
        state['warmup_time_s']=warmup(ctx,args.method,args.seed,args.quant_set,args.warmup_steps)
        if args.method=='sa':
            engine=PaperBatchedSA(ctx,args.seed,args.quant_set)
    iterator=r.train_iterator(ctx,state,args.seed) if args.method=='gradient' else None
    stop_requested=False
    def interrupt(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)

    def counters():
        if engine:
            s=engine.state
            return {'sa_candidates':s['num_proposals'],'sa_accepted':s['num_accepted'],
                'sa_rejected':s['num_rejected'],'sa_searches_completed':s['batches_completed'],
                'sa_attempts_in_search':s['attempts_in_search'],
                'sa_hit_rate':s['num_accepted']/s['num_proposals'] if s['num_proposals'] else None,
                'sa_train_ce':engine._cached_loss['loss'] if engine._cached_loss else None,
                'training_forward_calls':s['num_forward_calls'],'training_backward_calls':0,
                'loss_evaluations':s['num_loss_evals'],'seen_train_samples':s['seen_train_samples']}
        return {'optimizer_steps':state['optimizer_steps'],'epoch':state['epoch'],
            'batch_position':state['batch_position'],'learning_rate':r.learning_rate(state['epoch']),
            'training_forward_calls':state['optimizer_steps'],
            'training_backward_calls':state['optimizer_steps'],'seen_train_samples':state['seen_train_samples']}

    def observe(status):
        tick=time.perf_counter()
        rng=r.rng_state()
        try:
            val=r.p.published.evaluate(ctx.model,ctx.data,ctx.va,128,ctx.buffers)
        finally:
            r.restore_rng(rng)
        state['evaluation_time_s']+=time.perf_counter()-tick
        state['evaluation_forward_calls']+=40
        state['checkpoint_sequence']+=1
        sequence=state['checkpoint_sequence']
        snapshot=output/f'checkpoint_{sequence:03d}.pkl'
        row={'method':args.method,'quant_set':args.quant_set,'seed':args.seed,
            'initial_weights_hash':initial_hash,'split_hash':split_hash,
            'checkpoint_sequence':sequence,'checkpoint_target_s':state['next_target_s'],
            'training_time_s':state['training_time_s'],
            'overshoot_s':state['training_time_s']-state['next_target_s'],
            'val_accuracy':val['accuracy'],'val_loss':val['loss'],
            'evaluation_time_s':state['evaluation_time_s'],
            'setup_time_s':state['setup_time_s'],'warmup_time_s':state['warmup_time_s'],
            'save_time_s':state['save_time_s'],'status':status,
            'checkpoint_path':str(snapshot),**counters()}
        state['rows'].append(row)
        if state['training_time_s']>=state['next_target_s']:
            state['next_target_s']=min(state['next_target_s']+args.interval,args.budget)
        tick=time.perf_counter()
        payload={'state':copy.deepcopy(state),'indexes':r.model_indexes(ctx),
            'rng':r.rng_state(),'shadow':shadow,
            'sa_state':copy.deepcopy(engine.state) if engine else None,
            'sa_cached_loss':copy.deepcopy(engine._cached_loss) if engine else None}
        r.atomic_pickle(snapshot,payload)
        pointer=output/'latest.tmp'
        if pointer.exists(): pointer.unlink()
        os.link(snapshot,pointer)
        os.replace(pointer,checkpoint)
        state['save_time_s']+=time.perf_counter()-tick
        row['save_time_s']=state['save_time_s']
        row['total_wall_time_s']=state['prior_wall_time_s']+time.perf_counter()-started
        r.atomic_json(output/'latest.json',row)
        r.atomic_json(output/'trace.json',state['rows'])
        r.atomic_json(output/'progress.json',{'status':status,'method':args.method,
            'quant_set':args.quant_set,'seed':args.seed,'training_time_s':state['training_time_s'],
            'target_time_s':args.budget,**counters()})
        print(json.dumps(row),flush=True)

    while True:
        finished=not engine and state['epoch']>300
        due=state['training_time_s']>=state['next_target_s']
        if due or finished or stop_requested:
            status=('interrupted' if stop_requested else 'complete' if state['training_time_s']>=args.budget
                    else 'early_stop' if finished else 'running')
            observe(status)
            if status!='running' or (args.stop_after_checkpoint and state['checkpoint_sequence']>=args.stop_after_checkpoint):
                return
        tick=time.perf_counter()
        proposal=None
        if engine:
            if engine._cached_loss is None:
                engine.prepare_baseline()
            else:
                proposal=engine.step()
        else:
            iterator,_=gradient_step(ctx,state,shadow,iterator,args.seed)
        assert all(not np.any(layer.bias_array) for layer in ctx.layers)
        state['training_time_s']+=time.perf_counter()-tick
        # Journal/progress I/O is saving overhead, outside training timing.
        if engine or state['optimizer_steps']%100==0:
            tick=time.perf_counter()
            if proposal is not None:
                with (output/'candidates.jsonl').open('a') as stream:
                    stream.write(json.dumps(proposal,separators=(',',':'))+'\n')
            r.atomic_json(output/'progress.json',{'status':'running','method':args.method,
                'quant_set':args.quant_set,'seed':args.seed,
                'training_time_s':state['training_time_s'],'target_time_s':args.budget,**counters()})
            state['save_time_s']+=time.perf_counter()-tick


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['sa','gradient'],required=True)
    parser.add_argument('--quant-set',type=int,choices=[8,9,10,11],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--budget',type=float,default=1800.)
    parser.add_argument('--interval',type=float,default=300.)
    parser.add_argument('--warmup-steps',type=int,default=20)
    parser.add_argument('--stop-after-checkpoint',type=int)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=r.DATA_ROOT)
    args=parser.parse_args()
    assert args.budget>=args.interval>0 and args.warmup_steps>=0
    run(args)
