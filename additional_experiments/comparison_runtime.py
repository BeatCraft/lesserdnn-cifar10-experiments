"""Repository-local runtime for the time-budget comparison; no external checkout."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'lesserdnn_cifar10_ablation_200'
sys.path[:0] = [str(ROOT/'lesserdnn_cifar10_rigorous_eval'),
                str(SOURCE/'experiments/ldnn_cifar10_native_same_sign'), str(SOURCE)]
import train_one as published
from train import gradients_for_batch, update_weights, weighted_layers
from ldnn_cifar10_model import build_model, load_ldnn
from train_ldnn_cifar10 import track_temporary_metal_buffers, release_temporary_metal_buffers
from verify_pytorch_batch_ldnn import pytorch_batch_to_ldnn

DATA_ROOT = ROOT/'data'
from codebooks import CODEBOOKS, SHAPES, initial_indexes, digest


p = types.SimpleNamespace(ROOT=ROOT, RELEASE=ROOT, SOURCE=SOURCE, published=published,
    gradients_for_batch=gradients_for_batch, update_weights=update_weights,
    pytorch_batch_to_ldnn=pytorch_batch_to_ldnn,
    release_temporary_metal_buffers=release_temporary_metal_buffers, digest=digest)


def initialize(method, quant_set, seed):
    assert method == 'gradient'  # Both runners construct the same 128-image context.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = published.normalized_dataset(DATA_ROOT, train=True)
    tr, va = published.stratified_split(data.targets,5000,20260826)
    core, platform = load_ldnn(SOURCE/'ldnn_runtime',with_platform=True)
    # Each timed run uses a fresh process; construct the published Q9 template
    # first, then install the requested codebook and overwrite all initial weights.
    model = build_model(core,gpu=platform.getGpu(0),batch_size=128,seed=seed)
    model.set_backpropagation(True,0.)
    model.output.set_softmax_scale(1.)
    layers = weighted_layers(model,core)
    levels = CODEBOOKS[quant_set]
    for prefix in ['WEIGHT','CNN_WEIGHT']:
        setattr(core,prefix+'_SET',levels.tolist())
        setattr(core,prefix+'_INDEX_SIZE',len(levels))
        setattr(core,prefix+'_INDEX_MIN',0)
        setattr(core,prefix+'_INDEX_MAX',len(levels)-1)
        setattr(core,prefix+'_INDEX_ZERO',len(levels)//2)
    def zero_bias(layer):
        layer.bias_array.fill(0)
    ctx = types.SimpleNamespace(data=data,tr=tr,va=va,model=model,layers=layers,
        core=core,levels=levels,batch_size=128,
        buffers=track_temporary_metal_buffers(model._gpu))
    for layer in layers:
        layer.sync_bias_array_from_index = types.MethodType(zero_bias,layer)
    load_model_indexes(ctx,initial_indexes(seed,quant_set))
    assert model.count_weight() == 174944
    assert all(not np.any(layer.bias_array) for layer in layers)
    return ctx


def model_indexes(ctx):
    return {layer._index:layer._weight_index_matrix.copy() for layer in ctx.layers}


def load_model_indexes(ctx,indexes):
    for layer in ctx.layers:
        idx = indexes[layer._index].copy()
        layer._weight_index_matrix = idx
        layer._weight_matrix = np.ascontiguousarray(ctx.levels[idx],dtype=np.float32)
    ctx.model.update_weight()


def rng_state():
    return random.getstate(),np.random.get_state(),torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def train_iterator(ctx,state,seed):
    loader = published.train_loader(ctx.data,ctx.tr,128,seed,state['epoch'])
    return itertools.islice(loader,state['batch_position'],350)


def learning_rate(epoch):
    return .01 if epoch <= 100 else .005 if epoch <= 200 else .001


def comparison_commit():
    process = subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    return process.stdout.strip() if process.returncode == 0 else 'NA'


def atomic_json(path,value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def atomic_pickle(path,value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp,path)
