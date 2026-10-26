"""The measured strict first-improvement search (called SA in the results).

No temperature or worse-loss acceptance. A proposal changes eight independently
drawn positions with replacement, evaluated on all 45,000 training examples.
"""
import copy
import hashlib
import random

import numpy as np

import comparison_runtime as r


class PaperBatchedSA:
    KIND = 'paper_cnn_full_training_strict_search_v1'

    def __init__(self,ctx,seed,quant_set,payload=None):
        self.ctx = ctx
        ctx.model_spec = 'paper_cnn_174944_standardized_no_bn_v1'
        ctx.model.set_backpropagation(False,0.)
        self.seed = seed
        self.sizes = np.asarray([layer._weight_index_matrix.size for layer in ctx.layers])
        self.ends = np.cumsum(self.sizes)
        self.training_indices = list(map(int,ctx.tr))
        training_hash = hashlib.sha256(np.asarray(self.training_indices,dtype='<i8').tobytes()).hexdigest()
        self.state = {'kind':self.KIND,'seed':seed,'quant_set':quant_set,'architecture':'cnn',
            'model_spec':ctx.model_spec,'mutation_count':8,'epoch':1,'batch_index':0,
            'batches_completed':0,'num_proposals':0,'num_accepted':0,'num_rejected':0,
            'num_ties':0,'num_loss_evals':0,'seen_train_samples':0,'training_time_s':0.,
            'initial_weights_hash':r.digest({layer._index:layer._weight_matrix for layer in ctx.layers}),
            'training_indices_sha256':training_hash,'logical_batch_size':45000,
            'forward_chunk_size':128,'num_forward_calls':0,'max_candidates':100,
            'position_sampling':'uniform_with_replacement','execution_mode':'cached_input_and_baseline',
            'attempts_in_search':0}
        self._cached_loss = None
        if payload is not None:
            state = payload['state']
            for key in ['kind','seed','quant_set','mutation_count','model_spec','logical_batch_size',
                        'max_candidates','training_indices_sha256']:
                assert state[key] == self.state[key], f'Checkpoint mismatch: {key}'
            self.state = copy.deepcopy(state)
            r.load_model_indexes(ctx,payload['indexes'])
            r.restore_rng(payload['rng'])
            self._cached_loss = copy.deepcopy(payload['cached_loss'])
        assert len(self.training_indices) == 45000 and not set(self.training_indices)&set(ctx.va)

    def full_training_loss(self):
        rng = r.rng_state()
        try:
            result = r.p.published.evaluate(self.ctx.model,self.ctx.data,self.training_indices,
                                           128,self.ctx.buffers)
        finally:
            r.restore_rng(rng)
        assert result['samples'] == 45000 and np.isfinite(result['loss'])
        self.state['num_forward_calls'] += (len(self.training_indices)+127)//128
        return result

    def prepare_baseline(self):
        assert self._cached_loss is None
        result = self.full_training_loss()
        self._cached_loss = dict(result)
        self.state['num_loss_evals'] += 1
        self.state['seen_train_samples'] += 45000
        return result

    def _upload_mutations(self,mutations):
        for li,ni,ii in sorted({tuple(x[:3]) for x in mutations}):
            layer = self.ctx.layers[li]
            offset = 4*(ni*layer._num_input+ii)
            view = layer._gpu_weight.contents().as_buffer(layer._gpu_weight.length())
            view[offset:offset+4] = np.float32(layer._weight_matrix[ni,ii]).tobytes()

    def _rollback(self,mutations):
        for li,ni,ii,old,new in reversed(mutations):
            self.ctx.layers[li].set_weight_index(ni,ii,old)
        self._upload_mutations(mutations)

    def step(self):
        assert self._cached_loss is not None
        state = self.state
        baseline = self._cached_loss
        mutations = []
        try:
            selected = [random.randrange(int(self.ends[-1])) for _ in range(8)]
            for flat in selected:
                li = int(np.searchsorted(self.ends,flat,side='right'))
                offset = flat-(int(self.ends[li-1]) if li else 0)
                layer = self.ctx.layers[li]
                ni,ii = divmod(offset,layer._num_input)
                old = int(layer._weight_index_matrix[ni,ii])
                new = random.randrange(len(self.ctx.levels)-1)
                if new >= old:
                    new += 1
                mutations.append((li,ni,ii,old,new))
                layer.set_weight_index(ni,ii,new)
            self._upload_mutations(mutations)
            candidate = self.full_training_loss()
        except BaseException:
            self._rollback(mutations)
            raise
        accepted = candidate['loss'] < baseline['loss']
        if not accepted:
            self._rollback(mutations)
        self._cached_loss = dict(candidate if accepted else baseline)
        state['num_proposals'] += 1
        state['num_accepted'] += int(accepted)
        state['num_rejected'] += int(not accepted)
        state['num_ties'] += int(candidate['loss'] == baseline['loss'])
        state['num_loss_evals'] += 1
        state['seen_train_samples'] += 45000
        completed = state['batches_completed']
        attempted = state['attempts_in_search']+1
        finished = accepted or attempted == 100
        state['batches_completed'] = completed+int(finished)
        state['epoch'] = state['batches_completed']+1
        state['batch_index'] = 0
        state['attempts_in_search'] = 0 if finished else attempted
        return {'baseline_loss':baseline['loss'],'candidate_loss':candidate['loss'],
            'retained_loss':candidate['loss'] if accepted else baseline['loss'],
            'accepted':bool(accepted),'mutations':mutations,'baseline_samples':baseline['samples'],
            'candidate_samples':candidate['samples'],'candidate_number':state['num_proposals'],
            'search_number':completed+1,'candidate_in_search':attempted,'search_finished':finished}
