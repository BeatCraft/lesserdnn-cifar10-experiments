"""Batch the published CNN's unchanged Metal kernels into one forward command.

Backward and SGD/quantization stay on the published code path. Parameters have
separate persistent buffers: the legacy shared parameter buffer cannot be
rewritten while multiple encoders are waiting to execute.
"""
import Metal
import numpy as np


class BatchedPaperCNNForward:
    def __init__(self,ctx):
        self.ctx=ctx
        model=ctx.model
        gpu=model._gpu
        assert gpu.type==2 and model.count_weight()==174944
        assert not model.conv_batchnorm and not model.conv_scale_layer
        assert len(model.layers)==7 and model._batch_size==128
        assert all(not np.any(layer.bias_array) for layer in ctx.layers)
        self.original=model.propagate
        self.ops=[]
        self.parameter_buffers=[]
        def add(pipeline,buffers,params,grid,group=None):
            params=np.ascontiguousarray(params)
            buf=gpu.device.newBufferWithBytes_length_options_(params.tobytes(),params.nbytes,gpu.opts)
            assert buf is not None
            self.parameter_buffers.append(buf)
            if group is None:
                group=(min(int(pipeline.maxTotalThreadsPerThreadgroup()),256),1,1)
            self.ops.append((pipeline,tuple(buffers)+(buf,),Metal.MTLSize(*grid),Metal.MTLSize(*group)))
        previous=model.input._gpu_output
        for layer in model.layers[1:]:
            kind=layer.get_type()
            batch=layer._batch_size
            if kind==ctx.core.LAYER_TYPE_CONV:
                w,h,ch,filters=layer._w,layer._h,layer._ch,layer._filter
                add(gpu.pipe_padding_float,[previous,layer._gpu_padded],
                    np.asarray([w,h,ch],dtype=np.uint32),(batch,w,h))
                pipeline=gpu.pipe_conv_float_bias
                ty,tz=16,16
                while ty*tz>int(pipeline.maxTotalThreadsPerThreadgroup()):
                    if tz>1: tz//=2
                    elif ty>1: ty//=2
                    else: break
                add(pipeline,[layer._gpu_padded,layer._gpu_weight,layer.bias_gpu,layer._gpu_output],
                    np.asarray([w,h,ch,filters,0],dtype=np.uint32),
                    (batch,w,h),(1,min(ty,w),min(tz,h)))
                count=batch*filters*w*h
                add(gpu.pipe_relu,[layer._gpu_output],np.asarray([count],dtype=np.int32),(count,1,1))
            elif kind==ctx.core.LAYER_TYPE_MAX:
                w,h=layer._x,layer._y
                pipeline=gpu.pipe_max_float
                ty,tz=16,16
                while ty*tz>int(pipeline.maxTotalThreadsPerThreadgroup()):
                    if tz>1: tz//=2
                    elif ty>1: ty//=2
                    else: break
                add(pipeline,[previous,layer._gpu_output,layer._gpu_mask],
                    np.asarray([layer._ch,w,h],dtype=np.uint32),(batch,h,w),(1,min(ty,h),min(tz,w)))
            elif kind==ctx.core.LAYER_TYPE_OUTPUT:
                assert layer.smax and layer.softmax_scale==1.
                add(gpu.pipe_calc_mac_relu_bias,[previous,layer._gpu_weight,layer.bias_gpu,layer._gpu_output],
                    np.asarray([layer._num_node,layer._num_input,0],dtype=np.uint32),(batch,layer._num_node,1))
                params=np.zeros(1,dtype=np.dtype([('num',np.uint32),('scale',np.float32)],align=True))
                params['num']=layer._num_node; params['scale']=layer.softmax_scale
                add(gpu.pipe_softmax,[layer._gpu_output,layer._gpu_softmax],params,(batch,1,1))
            else:
                raise ValueError(f'Unsupported published-CNN layer: {kind}')
            previous=layer._gpu_output

    def forward(self,debug=0):
        if debug:
            return self.original(debug)
        model=self.ctx.model
        gpu=model._gpu
        command=gpu.queue.commandBuffer()
        for pipeline,buffers,grid,group in self.ops:
            encoder=command.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            for index,buffer in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buffer,0,index)
            encoder.dispatchThreads_threadsPerThreadgroup_(grid,group)
            encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        assert command.error() is None,str(command.error())
        # Materialize native backward's CPU views only after the entire forward
        # has completed. Pooling retains the legacy copy semantics.
        for layer in model.layers[1:]:
            if not layer.backprop:
                continue
            output=gpu.read_mbuf_to_numpy(layer._gpu_output,np.float32,layer._output_array.shape)
            layer._output_array=output.copy() if layer.get_type()==self.ctx.core.LAYER_TYPE_MAX else output
            if layer.get_type()==self.ctx.core.LAYER_TYPE_MAX:
                layer._mask_array=gpu.read_mbuf_to_numpy(layer._gpu_mask,np.float32,layer._mask_array.shape).copy()
            elif layer.get_type()==self.ctx.core.LAYER_TYPE_OUTPUT:
                layer._softmax_array=gpu.read_mbuf_to_numpy(layer._gpu_softmax,np.float32,layer._softmax_array.shape)

    def install(self):
        self.ctx.model.propagate=self.forward
        return self
