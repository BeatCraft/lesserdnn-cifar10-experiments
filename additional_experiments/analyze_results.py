"""Recompute the 80-run tables, paired contrasts and optional learning curves."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


DEFAULT = Path(__file__).resolve().parents[1]/'results/timed_codebooks_80runs'


def estimate(values):
    values=np.asarray(values,dtype=np.float64)
    n=len(values)
    mean=float(values.mean())
    sd=float(values.std(ddof=1))
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    return {'n':n,'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}


def paired(values):
    values=np.asarray(values,dtype=np.float64)
    observed=abs(float(values.mean()))
    shuffled=[abs(float(np.mean(values*np.asarray(signs))))
              for signs in itertools.product([-1,1],repeat=len(values))]
    return {**estimate(values),'exact_sign_flip_p_two_sided':
            sum(x>=observed-1e-12 for x in shuffled)/len(shuffled)}


def analyze(folder,plot=False,write=True):
    with (folder/'per_run_results.csv').open() as stream:
        records=list(csv.DictReader(stream))
    keys=[(x['method'],int(x['quant_set']),int(x['seed'])) for x in records]
    expected={(m,q,s) for m in ['sa','gradient'] for q in [8,9,10,11] for s in range(10)}
    assert len(keys)==80 and set(keys)==expected and len(set(keys))==80
    bykey=dict(zip(keys,records))
    aggregate=[]
    for method in ['sa','gradient']:
        for q in [8,9,10,11]:
            rows=[bykey[method,q,s] for s in range(10)]
            aggregate.append({'method':method,'quant_set':q,
                'validation_accuracy':estimate([float(x['val_accuracy']) for x in rows]),
                'validation_ce':estimate([float(x['val_loss']) for x in rows]),
                'training_seconds':estimate([float(x['training_time_s']) for x in rows])})
    contrasts={}
    for method in ['sa','gradient']:
        for a,b in itertools.combinations([8,9,10,11],2):
            contrasts[f'q{b}_minus_q{a}_{method}']={
                'accuracy_difference_pp':paired([100*(float(bykey[method,b,s]['val_accuracy'])-
                    float(bykey[method,a,s]['val_accuracy'])) for s in range(10)]),
                'ce_difference':paired([float(bykey[method,b,s]['val_loss'])-
                    float(bykey[method,a,s]['val_loss']) for s in range(10)])}
    interactions={}
    for method in ['sa','gradient']:
        differences=[]
        for seed in range(10):
            acc={q:float(bykey[method,q,seed]['val_accuracy']) for q in [8,9,10,11]}
            differences.append(100*((acc[11]-acc[10])-(acc[9]-acc[8])))
        interactions[method]={'description':'(Q11-Q10)-(Q9-Q8), accuracy percentage points; representation and initialization both vary',
                              **paired(differences)}
    summary={'runs':80,'metric':'validation at 30-minute endpoint; official test not evaluated',
             'aggregate':aggregate,'paired_codebook_contrasts':contrasts,
             'descriptive_interactions':interactions,
             'inference_note':'Paired seed-wise intervals and exact sign-flip p-values are descriptive, unadjusted for multiple comparisons. Codebook series were collected sequentially, not jointly randomized; initial distributions also differ.'}
    if write:
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Time-budget codebook comparison: 80 completed runs','',
           'Validation metrics at the 30-minute endpoint, 10 seeds per condition. No test-set evaluation.', '',
           '| Method | Levels | Validation CE mean ± SD | Validation accuracy mean ± SD |',
           '|---|---:|---:|---:|']
    for row in aggregate:
        ce=row['validation_ce'];acc=row['validation_accuracy']
        lines.append(f"| {row['method']} | {row['quant_set']} | {ce['mean']:.4f} ± {ce['sd']:.4f} | {acc['mean']*100:.2f}% ± {acc['sd']*100:.2f} pp |")
    lines+=['','The `sa` label denotes strict first-improvement random search: no temperature or worsening-loss acceptance.',
            'All codebooks use uniform initialization restricted to |w| ≤ 0.125. These initial distributions differ.',
            'See `summary.json` for paired contrasts, SDs and 95% intervals; p-values are exploratory and unadjusted.',
            'The three codebook series ran sequentially (Q9/Q10, Q11, Q8), so wall-time comparisons have an execution-order limitation.',
            '', 'Recompute: `python3 additional_experiments/analyze_results.py --plot` from the repository root.']
    if write:
        (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    if plot and write:
        draw(folder)
    print('timed_results_analysis=ok runs=80')
    return summary


def draw(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (folder/'learning_curves.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==560
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,layout='constrained')
    colors={8:'#666666',9:'#2672b1',10:'#d28113',11:'#17846a'}
    for i,method in enumerate(['sa','gradient']):
        for q in [8,9,10,11]:
            subset=[x for x in rows if x['method']==method and int(x['quant_set'])==q]
            for j,metric,factor in [(0,'val_loss',1),(1,'val_accuracy',100)]:
                x=[];y=[];sd=[]
                for target in range(0,1801,300):
                    points=[p for p in subset if float(p['checkpoint_target_s'])==target]
                    assert len(points)==10
                    x.append(np.mean([float(p['training_time_s']) for p in points])/60)
                    values=np.asarray([float(p[metric])*factor for p in points])
                    y.append(values.mean());sd.append(values.std(ddof=1))
                x=np.asarray(x);y=np.asarray(y);sd=np.asarray(sd)
                ax=axes[i,j]
                ax.plot(x,y,'o-',ms=3,color=colors[q],label=f'Q{q}')
                ax.fill_between(x,np.maximum(y-sd,.001),y+sd,color=colors[q],alpha=.13,linewidth=0)
        axes[i,0].set_yscale('log');axes[i,0].set_ylabel('Validation CE (log scale)')
        axes[i,1].set_ylabel('Validation accuracy (%)')
        title='Strict random search (SA label)' if method=='sa' else 'Gradient (single-batch)'
        axes[i,0].set_title(title);axes[i,1].set_title(title)
        for ax in axes[i]:
            ax.grid(alpha=.2,which='both');ax.set_xlim(0,30.3);ax.set_xticks(range(0,31,5))
            ax.legend(frameon=False,ncol=2)
    for ax in axes[-1]:ax.set_xlabel('Net training time (minutes)')
    fig.suptitle('8/9/10/11-level comparison | 10 seeds per condition\nMean and sample SD; validation only')
    fig.savefig(folder/'learning_curves.png',dpi=180)
    fig.savefig(folder/'learning_curves.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    analyze(args.input,args.plot)
