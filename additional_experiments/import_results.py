"""Export completed local runs as portable, lightweight public records.

Only the 80 time-budget runs are accepted. No pickle/checkpoints, images,
predictions, exploratory runs, or 300-epoch Q11 result are imported.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def clean(value):
    if isinstance(value,dict):
        return {clean(k):clean(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clean(v) for v in value]
    if isinstance(value,str):
        for folder,label in [('soict_camera_ready','experiment_workspace'),
                             ('lesserdnn_cifar10_paper_release','public_repository'),
                             ('LDNN-cifar-10','data')]:
            marker = '/'+folder+'/'
            if value.startswith('/') and marker in value:
                return label+'/'+value.split(marker,1)[1]
        if value.startswith('/'):
            return 'local_path/'+Path(value).name
    return value


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def export(args):
    args.output.mkdir(parents=True,exist_ok=True)
    records,curves,schedule,provenance,initials = [],[],[],[],{}
    all_environments = {}
    scalar_fields = ['method','quant_set','seed','git_commit','training_time_s','val_accuracy','val_loss',
        'optimizer_steps','epoch','batch_position','learning_rate','sa_candidates','sa_accepted',
        'sa_rejected','sa_searches_completed','sa_attempts_in_search','sa_hit_rate','sa_train_ce',
        'training_forward_calls','training_backward_calls','loss_evaluations','seen_train_samples',
        'evaluation_time_s','save_time_s','warmup_time_s','setup_time_s','total_wall_time_s',
        'initial_weights_hash','split_hash']
    curve_fields = ['checkpoint_sequence','checkpoint_target_s','overshoot_s','status']+scalar_fields
    roots = [args.run_root] if args.run_root else [args.q9q10_root,args.q11_root,args.q8_root]
    if any(root is None for root in roots):
        raise ValueError('Use --run-root or all three --q9q10-root/--q11-root/--q8-root arguments')
    for root in roots:
        manifest_path = root/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        config = manifest['configuration']
        assert json.loads((root/'queue_status.json').read_text())['status']=='complete'
        provenance.append({'source_series':root.name,'original_manifest_sha256':
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'manifest':clean(manifest)})
        for item in config['run_order']:
            folder = root/'runs'/item['run_name']
            final = json.loads((folder/'latest.json').read_text())
            trace = json.loads((folder/'trace.json').read_text())
            assert final['status']=='complete' and final['training_time_s']>=1800
            assert final['quant_set'] in [8,9,10,11] and len(trace)==7
            assert [x['checkpoint_target_s'] for x in trace]==list(range(0,1801,300))
            run_id = f"{final['method']}_q{final['quant_set']}_seed{final['seed']}"
            schedule.append({'method':final['method'],'quant_set':final['quant_set'],
                             'seed':final['seed'],'run_name':run_id})
            environment = json.loads((folder/'environment.json').read_text())
            env_key = hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()
            all_environments[env_key] = environment
            row = {key:final.get(key,'NA') for key in scalar_fields}
            run_config=json.loads((folder/'config.json').read_text())
            row['git_commit']=run_config.get('git_commit','NA')
            row.update(initial_val_accuracy=trace[0]['val_accuracy'],initial_val_loss=trace[0]['val_loss'],
                       environment_id=env_key,source_series=root.name,
                       source_trace_sha256=hashlib.sha256((folder/'trace.json').read_bytes()).hexdigest())
            records.append(row)
            curves += [{**{key:x.get(key,'NA') for key in curve_fields},'git_commit':row['git_commit']} for x in trace]
            key = (final['quant_set'],final['seed'])
            if key in initials:
                assert initials[key]==(final['initial_weights_hash'],final['split_hash'],
                                      trace[0]['val_accuracy'],trace[0]['val_loss'])
            initials[key]=(final['initial_weights_hash'],final['split_hash'],
                           trace[0]['val_accuracy'],trace[0]['val_loss'])
    assert len(records)==80 and len({(x['method'],x['quant_set'],x['seed']) for x in records})==80
    records.sort(key=lambda x:(x['quant_set'],x['method'],x['seed']))
    curves.sort(key=lambda x:(x['quant_set'],x['method'],x['seed'],x['checkpoint_target_s']))
    write_csv(args.output/'per_run_results.csv',records)
    write_csv(args.output/'learning_curves.csv',curves)
    (args.output/'provenance.json').write_text(json.dumps({
        'path_policy':'Local absolute paths replaced with logical artifact-relative labels; hashes are original file hashes.',
        'source_manifests':provenance,'environments':all_environments},indent=2)+'\n')
    protocol = {'protocol':'timed_codebooks_v1','planned_runs':80,'completed_runs':80,
        'methods':['sa','gradient'],'quant_sets':[8,9,10,11],'seeds':list(range(10)),
        'model_weights':174944,'budget_s':1800,'interval_s':300,'initial_max_abs':.125,
        'initialization':'independent uniform draws from codebook entries with abs(value)<=0.125',
        'gradient_batch_size':128,'gradient_batches_per_epoch':350,
        'gradient_max_epochs':300,'gradient_lr_segments':[
            {'start':1,'end':100,'learning_rate':.01},{'start':101,'end':200,'learning_rate':.005},
            {'start':201,'end':300,'learning_rate':.001}],
        'gradient_update_rule':'single','sa_mutation_count':8,'sa_max_candidates_per_search':100,
        'sa_position_sampling':'uniform_with_replacement','sa_acceptance':'strict_full_training_ce_improvement',
        'sa_temperature':None,'sa_logical_batch_size':45000,'forward_chunk_size':128,
        'split_seed':20260826,'train_size':45000,'validation_size':5000,'test_used':False,
        'selection':'30-minute endpoint; no best-validation selection',
        'timing':'validation, setup, warm-up and saving excluded; initial SA baseline included; stop at candidate/update boundary',
        'run_order':schedule,'codebook_series_order':'Q9/Q10, then Q11, then Q8; not jointly randomized',
        'excluded':'Exploratory trials and the rejected single Q11 300-epoch run are not part of this dataset.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(f'Exported {len(records)} completed runs and {len(curves)} observations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,help='Unified output of run_experiment.py (80 completed runs)')
    parser.add_argument('--q9q10-root',type=Path)
    parser.add_argument('--q11-root',type=Path)
    parser.add_argument('--q8-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args())
