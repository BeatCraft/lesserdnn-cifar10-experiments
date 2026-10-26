"""Compare the portable runtime against recorded original-implementation fixtures."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE=Path(__file__).resolve().parent


def one(args):
    import comparison_runtime as r
    from batched_paper_cnn_metal import BatchedPaperCNNForward
    from strict_search import PaperBatchedSA
    r.DATA_ROOT=args.data_root.resolve()
    fixtures=json.loads((HERE/'reference_fixtures.json').read_text())['fixtures']
    expected=next(x for x in fixtures if x['quant_set']==args.quant_set)
    with (HERE.parent/'results/timed_codebooks_80runs/per_run_results.csv').open() as stream:
        recorded=next(x for x in csv.DictReader(stream) if x['method']=='gradient' and
                      int(x['quant_set'])==args.quant_set and int(x['seed'])==0)
    ctx=r.initialize('gradient',args.quant_set,0)
    backend=BatchedPaperCNNForward(ctx).install()
    initial=r.model_indexes(ctx);rng=r.rng_state()
    assert r.digest({layer._index:layer._weight_matrix for layer in ctx.layers})==recorded['initial_weights_hash']
    validation=r.p.published.evaluate(ctx.model,ctx.data,ctx.va,128,ctx.buffers)
    r.restore_rng(rng)
    assert validation['loss']==float(recorded['initial_val_loss'])
    assert validation['accuracy']==float(recorded['initial_val_accuracy'])
    shadow={layer._index:layer._weight_matrix.copy() for layer in ctx.layers}
    iterator=r.train_iterator(ctx,{'epoch':1,'batch_position':0},0)
    for fixture in expected['gradient_first_three_updates']:
        images,labels=next(iterator)
        batch=r.p.gradients_for_batch(ctx.model,ctx.layers,images,labels,1.,ctx.buffers)
        assert r.digest(batch[3])==fixture['gradient_sha256'],f'Gradient mismatch at step {fixture["step"]}'
        r.p.update_weights(ctx.model,ctx.layers,shadow,batch[3],None,.01,0.,'single')
        assert r.digest(shadow)==fixture['shadow_sha256']
        assert r.digest(r.model_indexes(ctx))==fixture['indexes_sha256']
    r.load_model_indexes(ctx,initial);r.restore_rng(rng)
    engine=PaperBatchedSA(ctx,0,args.quant_set)
    engine.prepare_baseline()
    saved=None
    for i,fixture in enumerate(expected['sa_first_two_proposals']):
        if i==1:
            saved={'state':copy.deepcopy(engine.state),'indexes':r.model_indexes(ctx),
                   'rng':r.rng_state(),'cached_loss':copy.deepcopy(engine._cached_loss)}
        actual=engine.step()
        assert json.loads(json.dumps(actual))==fixture,f'SA candidate {i+1} differs'
    resumed=PaperBatchedSA(ctx,0,args.quant_set,payload=saved)
    assert json.loads(json.dumps(resumed.step()))==expected['sa_first_two_proposals'][1]
    before=r.model_indexes(ctx)
    engine=resumed
    mutations=[]
    layer=ctx.layers[0]
    for _ in range(3):
        old=int(layer._weight_index_matrix[0,0]);new=(old+1)%args.quant_set
        mutations.append((0,0,0,old,new));layer.set_weight_index(0,0,new)
    engine._upload_mutations(mutations);engine._rollback(mutations)
    for key,value in before.items():np.testing.assert_array_equal(r.model_indexes(ctx)[key],value)
    engine.full_training_loss=lambda:dict(engine._cached_loss)
    engine.state.update(attempts_in_search=0,batches_completed=0,epoch=1)
    for i in range(100):
        row=engine.step()
        assert not row['accepted'] and row['search_finished']==(i==99)
    assert engine.state['attempts_in_search']==0 and engine.state['batches_completed']==1
    for layer in ctx.layers:
        np.testing.assert_array_equal(layer._weight_index_matrix,before[layer._index])
        actual=ctx.model._gpu.read_mbuf_to_numpy(layer._gpu_weight,np.float32,layer._weight_matrix.shape)
        np.testing.assert_array_equal(actual,layer._weight_matrix)
        assert not np.any(layer.bias_array)
    result={'quant_set':args.quant_set,'status':'passed','initial_validation_exact':True,
        'native_gradient_shadow_indexes_exact_three_updates':True,
        'sa_proposals_exact':True,'sa_resume_exact':True,'duplicate_rollback_verified':True,
        'strict_tie_rejection_100_candidates_verified':True}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f'q{args.quant_set}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main(args):
    if args.quant_set is not None:
        one(args);return
    for quant in [8,9,10,11]:
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--quant-set',str(quant),
                        '--data-root',str(args.data_root),'--output',str(args.output)],check=True)
    print('native_verification=ok codebooks=8,9,10,11')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quant-set',type=int,choices=[8,9,10,11])
    parser.add_argument('--data-root',type=Path,default=HERE.parent/'data')
    parser.add_argument('--output',type=Path,default=HERE.parent/'runs/native_verification')
    main(parser.parse_args())
