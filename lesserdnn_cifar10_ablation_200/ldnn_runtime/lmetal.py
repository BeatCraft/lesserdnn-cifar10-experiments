#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import numpy as np

import Metal

import gpu

MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void double_float(device const float* a [[buffer(0)]],
                        device float*       b [[buffer(1)]],
                        uint idx [[thread_position_in_grid]]) {
    b[idx] = a[idx] * (float)2.0;
}

struct Params_mac {
    uint  xsize; // node
    uint  wsize; // input
    uint  act;
};

kernel void calc_mac_relu(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params_mac& P [[buffer(3)]],    
    uint2 gid [[thread_position_in_grid]])
{
    uint bi = gid.x;
    uint xi = gid.y;
    
    uint x_start = P.wsize * bi;
    uint w_start = P.wsize * xi;
    uint y_start = (P.xsize * bi) + xi;
    float temp = 0.0;

    for (uint i=0;i<P.wsize;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }
    
    if (P.act==0){ // no
        y[y_start] = (float)temp;
    } else {
        if (temp>=0){
            y[y_start] = (float)temp;
        }else{
            y[y_start] = (float)0.0;
        }
    }
}

kernel void calc_mac_relu_bias(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant Params_mac& P [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint bi = gid.x;
    uint xi = gid.y;

    uint x_start = P.wsize * bi;
    uint w_start = P.wsize * xi;
    uint y_start = (P.xsize * bi) + xi;
    float temp = bias[xi];

    for (uint i=0;i<P.wsize;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }

    if (P.act==0){
        y[y_start] = (float)temp;
    } else {
        if (temp>=0){
            y[y_start] = (float)temp;
        }else{
            y[y_start] = (float)0.0;
        }
    }
}

struct Params_scale {
    uint  size;
    float  scale;
};

kernel void scale_layer(
    device float* data [[buffer(0)]],
    constant Params_scale& P [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    uint bi = idx;
    uint start = bi * P.size;
    float max = 0.0;
    
    for (uint i=0;i<P.size;i++){
        float k = fabs(data[start+i]);
        if (k>max){
            max = k;
        }
    }
    
    if (max>0.0){
        max = max / P.scale;
        for (uint i=0;i<P.size;i++){
            data[start+i] = (data[start+i]/max);
        }
    }
}

struct Params_softmax {
    uint num;
    float scale;
};

kernel void softmax(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant Params_softmax& P [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    int bi = idx;
    float temp = 0.0;
    float total = 0.0;
    uint start = bi*P.num;

    for (uint i=0;i<P.num;i++){
        temp = (float)(in[start+i] / P.scale);
        //if (temp>11.0){ // fix overflow
        //    temp = 11.0;
        //}
        temp = exp(temp);
        if (isinf(temp)){
            temp = 3.402823e+38;
        }else if (isnan(temp)){
            temp = 0;
        }
        //printf("exp=%f\n", (float)temp);
        out[start+i] = (float)temp;
        total += temp;
    }

    for (uint i=0;i<P.num;i++){
        out[start+i] = out[start+i]/(float)total;
    }
}

struct Params_ce {
    uint  num;
};

kernel void cross_entropy(
    device const float* infs [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Params_ce& P [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    const float eps = 1e-7f;
    float sum = 0.0f;
    const uint base = idx * P.num;
    
    for (uint i = 0; i < P.num; ++i) {
        float t = (float)labels[base + i];
        float p = (float)infs[base + i];
        p = fmax(p, eps);
        sum += t * log(p);
    }
    
    output[idx] = (float)(-sum);
}

kernel void cross_entropy16(
    device const float* infs [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Params_ce& P [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    int bi = idx;
    /*
    float delta;
    float k;
    float t;
    float sum;
    
    delta = 0.0000001;
    sum = 0.0;
    
    for (uint i=0;i<P.num;i++){
        t = labels[bi*P.num + i];
        k = infs[bi*P.num + i] + delta;
        sum += t * log(k);
    }
    
    output[bi] = (-1.0)*sum;
    */
    
    const float eps = 1e-7f;
    float sum = 0.0f;

    const uint base = idx * P.num;
    for (uint i = 0; i < P.num; ++i) {
        float t = (float)labels[base + i];
        float p = (float)infs  [base + i];
        p = fmax(p, eps);
        sum += t * log(p);
    }

    output[idx] = (float)(-sum);
}

struct Params_rs {
    uint  n;
};

kernel void reduce_sum_pass(
    device const float*  in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant Params_rs&    P   [[buffer(2)]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  gtid [[thread_position_in_grid]],
    uint  tgsz [[threads_per_threadgroup]],
    uint  gidg [[threadgroup_position_in_grid]]
){
    threadgroup float sdata[1024];
    float v = 0.0f;
    if (gtid < P.n) v = (float)in[gtid];
    sdata[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { out[gidg] = sdata[0]; }
}

struct Params_padding_float {
    uint w;
    uint h;
    uint ch;
};

kernel void padding_float(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant Params_padding_float& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
)
{
    uint bi = gid.x;
    uint xi = gid.y;
    uint yi = gid.z;

    if (xi >= p.w || yi >= p.h) return;

    uint b_stride  = p.w * p.h * p.ch;
    uint ch_stride = p.w * p.h;
    uint y_stride  = yi * p.w;

    uint out_w = p.w + 2;
    uint out_h = p.h + 2;

    uint out_b_stride  = out_w * out_h * p.ch;
    uint out_ch_stride = out_w * out_h;
    uint out_y_stride  = (yi + 1) * out_w;

    for (uint i = 0; i < p.ch; i++) {
        uint index     = b_stride * bi + ch_stride * i + y_stride + xi;
        uint out_index = out_b_stride * bi
                       + out_ch_stride * i
                       + out_y_stride
                       + (xi + 1);

        output[out_index] = input[index];
    }
}

struct ConvParams {
    int w;
    int h;
    int ch;
    int filter;
    int activation; // 0: linear, else: relu
};

kernel void conv_float(
    device const float* input   [[buffer(0)]],  // padded input: (w+2)*(h+2)*ch per batch
    device const float* weight  [[buffer(1)]],  // weights: 3*3*filter*ch
    device float*       output  [[buffer(2)]],  // output: w*h*filter per batch
    constant ConvParams& p      [[buffer(3)]],
    uint3 gid                   [[thread_position_in_grid]]
)
{
    int bi = (int)gid.x;
    int xi = (int)gid.y;
    int yi = (int)gid.z;

    if (xi >= p.w || yi >= p.h) return;

    int in_w = p.w + 2;
    int in_h = p.h + 2;

    int ch_stride = in_w * in_h;        // (w+2)*(h+2)
    int b_stride  = ch_stride * p.ch;   // per-batch input stride
    int y_stride  = yi * in_w;

    int i_start = b_stride * bi + y_stride;

    int out_b_stride = p.w * p.h * p.filter; // per-batch output stride
    int out_yx = yi * p.w + xi;

    for (int fi = 0; fi < p.filter; fi++) {
        float sum = 0.0f;
        int f_start = 3 * 3 * fi * p.ch; // per-filter start (over channels)

        for (int ci = 0; ci < p.ch; ci++) {
            int start   = i_start + ch_stride * ci;     // top-left row base for this channel
            int w_start = f_start + ci * 3 * 3;         // 3x3 weights for (fi, ci)

            // row 0
            sum += input[start + xi + 0] * weight[w_start + 0];
            sum += input[start + xi + 1] * weight[w_start + 1];
            sum += input[start + xi + 2] * weight[w_start + 2];

            // row 1
            int r1 = start + in_w;
            sum += input[r1 + xi + 0] * weight[w_start + 3];
            sum += input[r1 + xi + 1] * weight[w_start + 4];
            sum += input[r1 + xi + 2] * weight[w_start + 5];

            // row 2
            int r2 = start + in_w * 2;
            sum += input[r2 + xi + 0] * weight[w_start + 6];
            sum += input[r2 + xi + 1] * weight[w_start + 7];
            sum += input[r2 + xi + 2] * weight[w_start + 8];
        }

        // activation
        if (p.activation != 0) { // relu
            sum = max(sum, 0.0f);
        }

        output[out_b_stride * bi + (p.w * p.h) * fi + out_yx] = sum;
    }
}


kernel void conv_float_bias(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant ConvParams& p      [[buffer(4)]],
    uint3 gid                   [[thread_position_in_grid]]
)
{
    int bi = (int)gid.x;
    int xi = (int)gid.y;
    int yi = (int)gid.z;

    if (xi >= p.w || yi >= p.h) return;

    int in_w = p.w + 2;
    int ch_stride = in_w * (p.h + 2);
    int b_stride  = ch_stride * p.ch;
    int y_stride  = yi * in_w;
    int i_start = b_stride * bi + y_stride;
    int out_b_stride = p.w * p.h * p.filter;
    int out_yx = yi * p.w + xi;

    for (int fi = 0; fi < p.filter; fi++) {
        float sum = bias[fi];
        int f_start = 3 * 3 * fi * p.ch;

        for (int ci = 0; ci < p.ch; ci++) {
            int start   = i_start + ch_stride * ci;
            int w_start = f_start + ci * 3 * 3;

            sum += input[start + xi + 0] * weight[w_start + 0];
            sum += input[start + xi + 1] * weight[w_start + 1];
            sum += input[start + xi + 2] * weight[w_start + 2];

            int r1 = start + in_w;
            sum += input[r1 + xi + 0] * weight[w_start + 3];
            sum += input[r1 + xi + 1] * weight[w_start + 4];
            sum += input[r1 + xi + 2] * weight[w_start + 5];

            int r2 = start + in_w * 2;
            sum += input[r2 + xi + 0] * weight[w_start + 6];
            sum += input[r2 + xi + 1] * weight[w_start + 7];
            sum += input[r2 + xi + 2] * weight[w_start + 8];
        }

        if (p.activation != 0) {
            sum = max(sum, 0.0f);
        }

        output[out_b_stride * bi + (p.w * p.h) * fi + out_yx] = sum;
    }
}

struct MaxPoolParams {
    int ch;
    int w;  // output w
    int h;  // output h
};

kernel void max_float(
    device const float* input  [[buffer(0)]], // input: (w*2)*(h*2)*ch per batch
    device float*       output [[buffer(1)]], // output: w*h*ch per batch
    device float*       mask [[buffer(2)]],
    constant MaxPoolParams& p  [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
)
{
    int bi = (int)gid.x;
    int y  = (int)gid.y;
    int x  = (int)gid.z;

    // 出力範囲外ガード
    if (x >= p.w || y >= p.h) return;

    int input_w = p.w * 2;
    int input_h = p.h * 2;

    int ich_stride    = input_w * input_h;
    int input_stride  = ich_stride * p.ch;
    int input_offset  = input_stride * bi;

    int och_stride    = p.w * p.h;
    int output_stride = och_stride * p.ch;
    int output_offset = output_stride * bi;

    // 出力(x,y)に対応する入力左上 (2x2)
    int base_xy = (input_w * (y * 2)) + (x * 2);

    for (int c = 0; c < p.ch; c++) {
        int k = input_offset + ich_stride * c + base_xy;
        int idx = 0;
        float m = input[k];
        float m_next = input[k + 1];
        //m = max(m, input[k + 1]);
        //m = max(m, input[k + input_w]);
        //m = max(m, input[k + input_w + 1]);
        
        //
        if (m_next>m){
            m = m_next;
            idx=1;
        }
        m_next = input[k + input_w];
        if (m_next>m){
            m = m_next;
            idx=2;
        }
        m_next = input[k + input_w+1];
        if (m_next>m){
            m = m_next;
            idx=3;
        }
        
        switch (idx) {
            case 0:
                mask[k] = 1.0;
                mask[k + 1] = 0.0;
                mask[k + input_w] = 0.0;
                mask[k + input_w + 1] = 0.0;
                break;
            case 1:
                mask[k] = 0.0;
                mask[k + 1] = 1.0;
                mask[k + input_w] = 0.0;
                mask[k + input_w + 1] = 0.0;
                break;
            case 2:
                mask[k] = 0.0;
                mask[k + 1] = 0.0;
                mask[k + input_w] = 1.0;
                mask[k + input_w + 1] = 0.0;
                break;
            case 3:
                mask[k] = 0.0;
                mask[k + 1] = 0.0;
                mask[k + input_w] = 0.0;
                mask[k + input_w + 1] = 1.0;
                break;
        }
        //
        output[output_offset + och_stride * c + (p.w * y + x)] = m;
    }
}

struct MaxBPParams {
    int ch; // channels
    int w;  // output w (= self._x)
    int h;  // output h (= self._y)
};

kernel void max_bp_float(
    device const float* delta [[buffer(0)]], // (B, ch, h, w) もしくは (B,ch,h*w) のフラット
    device const float* mask  [[buffer(1)]], // (B, ch, 2h, 2w) のフラット（0/1）
    device float*       grad  [[buffer(2)]], // (B, ch, 2h, 2w) のフラット 出力
    constant MaxBPParams& p   [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
)
{
    int bi = (int)gid.x;
    int y  = (int)gid.y;
    int x  = (int)gid.z;

    if (x >= p.w || y >= p.h) return;

    int in_w = p.w * 2;
    int in_h = p.h * 2;

    int out_hw = p.w * p.h;
    int in_hw  = in_w * in_h;

    // base offsets
    int delta_b = bi * (p.ch * out_hw);
    int in_b    = bi * (p.ch * in_hw);

    // 対応する入力 2x2 の左上
    int in_xy = (y * 2) * in_w + (x * 2);
    int out_xy = y * p.w + x;

    for (int c = 0; c < p.ch; c++) {
        int delta_idx = delta_b + c * out_hw + out_xy;
        float d = delta[delta_idx];

        int base = in_b + c * in_hw + in_xy;

        // mask は入力と同じ並びで 0/1 が入っている想定
        float m0 = mask[base];
        float m1 = mask[base + 1];
        float m2 = mask[base + in_w];
        float m3 = mask[base + in_w + 1];

        grad[base]             = d * m0;
        grad[base + 1]         = d * m1;
        grad[base + in_w]      = d * m2;
        grad[base + in_w + 1]  = d * m3;
    }
}

struct ConvBackParams {
    int batch;
    int w;
    int h;
    int ch;
    int filter;
};

kernel void conv4_relu_back_input(
    device const float* dY    [[buffer(0)]], // (B,F,H,W)
    device const float* out   [[buffer(1)]], // (B,F,H,W) after ReLU
    device const float* weight[[buffer(2)]], // (F,C,3,3)
    device float*       dX    [[buffer(3)]], // (B,C,H,W)
    constant ConvBackParams& p[[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
)
{
    int bi = (int)gid.x;
    int xi = (int)gid.y;
    int yi = (int)gid.z;

    if (bi >= p.batch || xi >= p.w || yi >= p.h) return;

    int hw = p.w * p.h;
    int dy_b_stride = p.filter * hw;
    int dx_b_stride = p.ch * hw;

    for (int ci = 0; ci < p.ch; ci++) {
        float sum = 0.0f;

        for (int fi = 0; fi < p.filter; fi++) {
            int dy_base = bi * dy_b_stride + fi * hw;

            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int oy = yi - ky + 1;
                    int ox = xi - kx + 1;
                    if (oy < 0 || oy >= p.h || ox < 0 || ox >= p.w) continue;

                    int out_idx = dy_base + oy * p.w + ox;
                    float g = dY[out_idx];
                    if (out[out_idx] <= 0.0f) g = 0.0f;

                    int w_idx = fi * (p.ch * 9) + ci * 9 + ky * 3 + kx;
                    sum += weight[w_idx] * g;
                }
            }
        }

        int dx_idx = bi * dx_b_stride + ci * hw + yi * p.w + xi;
        dX[dx_idx] = sum;
    }
}

kernel void conv4_relu_back_weight(
    device const float* xpad  [[buffer(0)]], // (B,C,H+2,W+2)
    device const float* dY    [[buffer(1)]], // (B,F,H,W)
    device const float* out   [[buffer(2)]], // (B,F,H,W) after ReLU
    device float*       dW    [[buffer(3)]], // (F,C,3,3)
    constant ConvBackParams& p[[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
)
{
    int fi = (int)gid.x;
    int ci = (int)gid.y;
    int kk = (int)gid.z;

    if (fi >= p.filter || ci >= p.ch || kk >= 9) return;

    int ky = kk / 3;
    int kx = kk % 3;

    int hw = p.w * p.h;
    int xpad_w = p.w + 2;
    int xpad_h = p.h + 2;
    int xpad_hw = xpad_w * xpad_h;

    int dy_b_stride   = p.filter * hw;
    int xpad_b_stride = p.ch * xpad_hw;

    float acc = 0.0f;

    for (int bi = 0; bi < p.batch; bi++) {
        int dy_base   = bi * dy_b_stride   + fi * hw;
        int xpad_base = bi * xpad_b_stride + ci * xpad_hw;

        for (int y = 0; y < p.h; y++) {
            for (int x = 0; x < p.w; x++) {
                int out_idx = dy_base + y * p.w + x;
                float g = dY[out_idx];
                if (out[out_idx] <= 0.0f) g = 0.0f;

                int x_idx = xpad_base + (y + ky) * xpad_w + (x + kx);
                acc += xpad[x_idx] * g;
            }
        }
    }

    dW[fi * (p.ch * 9) + ci * 9 + kk] = acc / (float)p.batch;
}


// =========================
// FC backward kernels
// =========================

struct FCBackDeltaParams {
    int batch;
    int cur_nodes;
    int next_nodes;
};

kernel void fc_hidden_delta_relu(
    device const float* next_delta [[buffer(0)]],
    device const float* next_w     [[buffer(1)]],
    device const float* cur_out    [[buffer(2)]],
    device float*       cur_delta  [[buffer(3)]],
    constant FCBackDeltaParams& p  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    int bi = (int)gid.x;
    int ci = (int)gid.y;

    if (bi >= p.batch || ci >= p.cur_nodes) return;

    float acc = 0.0f;
    int nd_base = bi * p.next_nodes;

    for (int nj = 0; nj < p.next_nodes; nj++) {
        acc += next_delta[nd_base + nj] * next_w[nj * p.cur_nodes + ci];
    }

    float y = cur_out[bi * p.cur_nodes + ci];
    if (y <= 0.0f) acc = 0.0f;

    cur_delta[bi * p.cur_nodes + ci] = acc;
}

struct FCGradParams {
    int batch;
    int in_nodes;
    int out_nodes;
};

kernel void fc_weight_grad(
    device const float* x_pre [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float*       dW    [[buffer(2)]],
    constant FCGradParams& p  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    int ii = (int)gid.x;
    int oi = (int)gid.y;

    if (ii >= p.in_nodes || oi >= p.out_nodes) return;

    float acc = 0.0f;
    for (int b = 0; b < p.batch; b++) {
        acc += x_pre[b * p.in_nodes + ii] * delta[b * p.out_nodes + oi];
    }

    dW[ii * p.out_nodes + oi] = acc / (float)p.batch;
}

struct FCOutputDeltaParams {
    int n;
};

kernel void fc_output_delta(
    device const float* softmax [[buffer(0)]],
    device const float* label   [[buffer(1)]],
    device float*       delta   [[buffer(2)]],
    constant FCOutputDeltaParams& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if ((int)idx >= p.n) return;
    delta[idx] = softmax[idx] - label[idx];
}

struct BN2DParams {
    int batch;
    int ch;
    int hw;
    float eps;
};

kernel void bn2d_mean_var(
    device const float* x [[buffer(0)]],   // (B,C,HW)
    device float* mean [[buffer(1)]],      // (C)
    device float* var [[buffer(2)]],       // (C)
    constant BN2DParams& p [[buffer(3)]],
    uint c [[thread_position_in_grid]]
)
{
    if ((int)c >= p.ch) return;

    float sum = 0.0f;
    float sum2 = 0.0f;
    int n = p.batch * p.hw;

    for (int b = 0; b < p.batch; b++) {
        int base = b * p.ch * p.hw + c * p.hw;
        for (int i = 0; i < p.hw; i++) {
            float v = x[base + i];
            sum += v;
            sum2 += v * v;
        }
    }

    float m = sum / (float)n;
    float v = sum2 / (float)n - m * m;
    if (v < 0.0f) v = 0.0f;

    mean[c] = m;
    var[c] = v;
}

kernel void bn2d_apply(
    device float* x [[buffer(0)]],          // in-place (B,C,HW)
    device const float* mean [[buffer(1)]],
    device const float* var [[buffer(2)]],
    constant BN2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
)
{
    int b = (int)gid.x;
    int c = (int)gid.y;
    int i = (int)gid.z;

    if (b >= p.batch || c >= p.ch || i >= p.hw) return;

    int idx = b * p.ch * p.hw + c * p.hw + i;

    float m = mean[c];
    float v = var[c];

    x[idx] = (x[idx] - m) * rsqrt(v + p.eps);
}

kernel void relu_float(
    device float* x [[buffer(0)]],
    constant int& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
)
{
    if ((int)gid >= size) return;

    float v = x[gid];

    if (v < 0.0f) {
        x[gid] = 0.0f;
    }
}

"""

def ceil_div(a, b): return (a + b - 1) // b

class LMetal(gpu.Gpu):
    def __init__(self):
        super(gpu.Gpu, self).__init__()
        self.name = "Metal"
        self.type = 2
        # -1:unknown, 0:OpenCL, 1:CuPy/GDX, 2:Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        assert self.device is not None, "No Metal device found"
        
        self.queue = self.device.newCommandQueue()
        
        #opts = Metal.MTLCompileOptions.alloc().init()
        #opts.setLanguageVersion_(Metal.MTLLanguageVersion2_1)
        self.lib, err = self.device.newLibraryWithSource_options_error_(MSL, None, None)
        if err:
            raise RuntimeError(err)
        #
        self.opts = Metal.MTLResourceOptions(Metal.MTLResourceStorageModeShared)
        self.params_buf = self.device.newBufferWithLength_options_(32, self.opts)

    def prepare(self):
        self.init_calc_mac_relu()
        self.init_calc_mac_relu_bias()
        self.init_scale_layer()
        self.init_softmax()
        self.init_cross_entropy()
        self.init_padding_float()
        self.init_conv_float()
        self.init_conv_float_bias()
        self.init_max_float()
        self.init_max_bp_float()
        self.init_conv4_back_input()
        self.init_conv4_back_weight()
        self.init_fc_hidden_delta_relu()
        self.init_fc_weight_grad()
        self.init_fc_output_delta()
        self.init_bn2d()
        self.init_relu()
        
    def init_relu(self):
        fn = self.lib.newFunctionWithName_("relu_float")
        self.pipe_relu, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        
    def relu(self, mbuf_x, size):
        params = np.array([size], dtype=np.int32)

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()

        enc.setComputePipelineState_(self.pipe_relu)

        enc.setBuffer_offset_atIndex_(mbuf_x, 0, 0)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 1)

        grid = Metal.MTLSize(size, 1, 1)

        t = min(
            self.pipe_relu.maxTotalThreadsPerThreadgroup(),
            256
        )

        tpg = Metal.MTLSize(t, 1, 1)

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    
    def init_bn2d(self):
        fn = self.lib.newFunctionWithName_("bn2d_mean_var")
        self.pipe_bn2d_mean_var, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

        fn = self.lib.newFunctionWithName_("bn2d_apply")
        self.pipe_bn2d_apply, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

    def bn2d_forward(self, batch_size, mbuf_x, ch, hw, eps=1e-5):
        mean = np.zeros((ch,), dtype=np.float32)
        var = np.zeros((ch,), dtype=np.float32)

        mbuf_mean = self.alloc_buf_from_array(mean)
        mbuf_var = self.alloc_buf_from_array(var)

        params = np.zeros(1, dtype=np.dtype([
            ("batch", np.int32),
            ("ch",    np.int32),
            ("hw",    np.int32),
            ("eps",   np.float32),
        ], align=True))

        params["batch"] = batch_size
        params["ch"] = ch
        params["hw"] = hw
        params["eps"] = np.float32(eps)

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        # mean / var
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_bn2d_mean_var)
        enc.setBuffer_offset_atIndex_(mbuf_x, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_mean, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_var, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSize(ch, 1, 1)
        tpg = Metal.MTLSize(min(self.pipe_bn2d_mean_var.maxTotalThreadsPerThreadgroup(), ch), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        # apply
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_bn2d_apply)
        enc.setBuffer_offset_atIndex_(mbuf_x, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_mean, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_var, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSize(batch_size, ch, hw)

        max_t = int(self.pipe_bn2d_apply.maxTotalThreadsPerThreadgroup())
        tx, ty, tz = 1, 1, min(hw, 256)
        while tx * ty * tz > max_t and tz > 1:
            tz //= 2
        #
        tpg = Metal.MTLSize(tx, ty, max(1, tz))
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
    def alloc_buf(self, nbytes):
        return self.device.newBufferWithLength_options_(nbytes, self.opts)
        
    def alloc_buf_from_array(self, abuf):
        return self.device.newBufferWithBytes_length_options_(memoryview(abuf).tobytes(), abuf.nbytes, self.opts)
        
    def write_np_to_mbuf(self, arr, mbuf):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        #
        mv = mbuf.contents().as_buffer(mbuf.length())
        mv[:arr.nbytes] = memoryview(arr).tobytes()
    
    def read_mbuf_to_numpy(self, mbuf, dtype, shape=None):
        count = mbuf.length() // np.dtype(dtype).itemsize
        arr = np.frombuffer(mbuf.contents().as_buffer(mbuf.length()), dtype=dtype, count=count)
        if shape is not None:
            arr = arr.reshape(shape, order="C")
        #
        return arr
    
    def copy_mbuf_to_numpy(self, mbuf, dtype, shape=None):
        view = mtlbuffer_to_numpy_view(mbuf, dtype, shape)
        return view.copy()
        
    def init_test_func(self):
        fn = self.lib.newFunctionWithName_("double_float")
        self.pipe, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        
    def test_func(self, mbuf0, mbuf1, n):
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)

        grid = Metal.MTLSize(n, 1, 1)
        tpg = Metal.MTLSize(min(self.pipe.threadExecutionWidth(), n), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
    
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def init_calc_mac_relu(self):
        fn = self.lib.newFunctionWithName_("calc_mac_relu")
        self.pipe_calc_mac_relu, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        #self.params_buf_mac = self.device.newBufferWithLength_options_(32, self.opts)

    def init_calc_mac_relu_bias(self):
        fn = self.lib.newFunctionWithName_("calc_mac_relu_bias")
        self.pipe_calc_mac_relu_bias, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        
    def calc_mac_relu(self, batch_size, mbuf0, mbuf1, mbuf2, xsize, wsize, act):
        params = np.zeros(1, dtype=np.dtype([
            ("xsize", np.uint32),
            ("wsize", np.uint32),
            ("act", np.uint32),
            ], align=True))
            
        params["xsize"] = xsize # node
        params["wsize"] = wsize # weight
        params["act"] = act
        
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_calc_mac_relu)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf2, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)
        
        grid = Metal.MTLSize(batch_size, xsize, 1)
        #tpg  = Metal.MTLSize(16, 16, 1)
        tpg  = Metal.MTLSize(min(self.pipe_calc_mac_relu.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def calc_mac_relu_bias(self, batch_size, mbuf_x, mbuf_w, mbuf_bias, mbuf_y, xsize, wsize, act):
        params = np.zeros(1, dtype=np.dtype([
            ("xsize", np.uint32),
            ("wsize", np.uint32),
            ("act", np.uint32),
            ], align=True))
        params["xsize"] = xsize
        params["wsize"] = wsize
        params["act"] = act

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_calc_mac_relu_bias)
        enc.setBuffer_offset_atIndex_(mbuf_x, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_w, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_bias, 0, 2)
        enc.setBuffer_offset_atIndex_(mbuf_y, 0, 3)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 4)

        grid = Metal.MTLSize(batch_size, xsize, 1)
        tpg = Metal.MTLSize(min(self.pipe_calc_mac_relu_bias.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def init_scale_layer(self):
        fn = self.lib.newFunctionWithName_("scale_layer")
        self.pipe_scale_layer, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
    
    def scale_layer(self, batch_size, size, scale, mbuf0):
        params = np.zeros(1, dtype=np.dtype([
            ("size", np.uint32),
            ("scale", np.float32),
            ], align=True))
        params["size"] = size
        params["scale"] = scale
        
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_scale_layer)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 1)
        
        grid = Metal.MTLSize(batch_size, 1, 1)
        #tpg  = Metal.MTLSize(4096, 1, 1)
        tpg  = Metal.MTLSize(min(self.pipe_scale_layer.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
    def init_softmax(self):
        fn = self.lib.newFunctionWithName_("softmax")
        self.pipe_softmax, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
    
    def softmax(self, batch_size, num, scale, mbuf0, mbuf1):
        
        params = np.zeros(1, dtype=np.dtype([
            ("num", np.uint32),
            ("scale", np.float32),
            ], align=True))
        params["num"] = num
        params["scale"] = scale
    
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_softmax)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 2)
        
        grid = Metal.MTLSize(batch_size, 1, 1)
        #tpg  = Metal.MTLSize(4096, 1, 1)
        tpg  = Metal.MTLSize(min(self.pipe_softmax.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
    def init_cross_entropy(self):
        fn = self.lib.newFunctionWithName_("cross_entropy")
        self.pipe_cross_entropy, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

    def cross_entropy(self, batch_size, num, mbuf0, mbuf1, mbuf2):
        params = np.zeros(1, dtype=np.dtype([
            ("num", np.uint32),
            ], align=True))
        params["num"] = num
    
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_cross_entropy)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf2, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSize(batch_size, 1, 1)
        #tpg  = Metal.MTLSize(4096, 1, 1)
        tpg  = Metal.MTLSize(min(self.pipe_cross_entropy.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()


    def init_reduce_sum_float(self):
        fn = self.lib.newFunctionWithName_("reduce_sum_float")
        self.pipe_reduce_sum_float, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

    def reduce_sum_float(self, arr_fp16):# np.ndarray):# -> float:
        params = np.zeros(1, dtype=np.dtype([
            ("n", np.uint32),
            ], align=True))
        params["n"] = num
    
        arr = np.ascontiguousarray(arr_fp16.astype(np.float32, copy=False))
        N = arr.size
        
        in_buf  = device.newBufferWithBytes_length_options_(memoryview(arr).tobytes(), arr.nbytes, opts)
        tpg_x = min(pipe.maxTotalThreadsPerThreadgroup(), 256)  # 例: 256
        tpg   = Metal.MTLSize(tpg_x, 1, 1)
        
        blocks = ceil_div(N, tpg_x)
        partial_bytes = blocks * 4  # float32
        out_buf = self.device.newBufferWithLength_options_(partial_bytes, opts)

        #Params_dtype = np.dtype([("n", np.uint32)], align=True)
        #params = np.zeros(1, dtype=Params_dtype); params["n"] = N
        #pbuf = device.newBufferWithBytes_length_options_(memoryview(params).tobytes(), params.nbytes, opts)
        #mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        #mv[:params.nbytes] = memoryview(params).tobytes()

        cur_in_is_float = True
        cur_in_buf = in_buf
        cur_len = N

        while True:
            blocks = ceil_div(cur_len, tpg_x)
            out_buf = self.device.newBufferWithLength_options_(blocks * 4, opts)

            params["n"] = cur_len
            mv = self.params_buf.contents().as_buffer(self.params_buf.length())
            memoryview(mv).cast('B')[:params.nbytes] = memoryview(params).tobytes()

            grid = Metal.MTLSize(blocks * tpg_x, 1, 1)
            cmd = queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipe_reduce_sum_float)

            if cur_in_is_float:
                enc.setBuffer_offset_atIndex_(cur_in_buf, 0, 0)  # float* in
                enc.setBuffer_offset_atIndex_(out_buf,   0, 1)  # float* out
                enc.setBuffer_offset_atIndex_(pbuf,      0, 2)
                enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
                enc.endEncoding()
                cmd.commit(); cmd.waitUntilCompleted()
            else:
                pass
            #
            if blocks == 1:
                res = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), dtype=np.float32, count=1)[0]
                return float(res)
            #
        #
        partial = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), dtype=np.float32, count=blocks)
        return float(partial.sum(dtype=np.float64))

    def init_padding_float(self):
        fn = self.lib.newFunctionWithName_("padding_float")
        self.pipe_padding_float, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        
    def padding_float(self, batch_size, mbuf_in, mbuf_out, w, h, ch):
        """
        mbuf_in  : float32 input buffer
        mbuf_out : float32 output buffer (should be zero-initialized)
        """

        # Params struct (uint x3)
        params = np.array([w, h, ch], dtype=np.uint32)
        mbuf_params = self.alloc_buf_from_array(params)

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_padding_float)
        
        enc.setBuffer_offset_atIndex_(mbuf_in, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_out, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_params, 0, 2)

        grid = Metal.MTLSize(batch_size, w, h)
        tpg  = Metal.MTLSize(min(self.pipe_padding_float.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def init_conv_float(self):
        fn = self.lib.newFunctionWithName_("conv_float")
        self.pipe_conv_float, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

    def init_conv_float_bias(self):
        fn = self.lib.newFunctionWithName_("conv_float_bias")
        self.pipe_conv_float_bias, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        
    def conv_float(self, batch_size, mbuf_in, mbuf_w, mbuf_out, w, h, ch, filter, activation):
        # params (uint32 x5), align=True to match MSL struct layout
        params = np.zeros(1, dtype=np.dtype([
            ("w", np.uint32),
            ("h", np.uint32),
            ("ch", np.uint32),
            ("filter", np.uint32),
            ("activation", np.uint32),
        ], align=True))
        params["w"] = w
        params["h"] = h
        params["ch"] = ch
        params["filter"] = filter
        params["activation"] = activation

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_conv_float)
        enc.setBuffer_offset_atIndex_(mbuf_in, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_w, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_out, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSize(batch_size, w, h)

        # threads per threadgroup: start from (1,16,16) and shrink to fit device limit
        max_t = int(self.pipe_conv_float.maxTotalThreadsPerThreadgroup())
        tx, ty, tz = 1, 16, 16
        while tx * ty * tz > max_t:
            if tz > 1:
                tz //= 2
            elif ty > 1:
                ty //= 2
            else:
                break
        #
        
        tpg = Metal.MTLSize(tx, min(ty, w if w > 0 else 1), min(tz, h if h > 0 else 1))
        #tpg  = Metal.MTLSize(min(max_t, 256), 1, 1)
                
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
    def conv_float_bias(self, batch_size, mbuf_in, mbuf_w, mbuf_bias, mbuf_out, w, h, ch, filter, activation):
        params = np.zeros(1, dtype=np.dtype([
            ("w", np.uint32),
            ("h", np.uint32),
            ("ch", np.uint32),
            ("filter", np.uint32),
            ("activation", np.uint32),
        ], align=True))
        params["w"] = w
        params["h"] = h
        params["ch"] = ch
        params["filter"] = filter
        params["activation"] = activation

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_conv_float_bias)
        enc.setBuffer_offset_atIndex_(mbuf_in, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_w, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_bias, 0, 2)
        enc.setBuffer_offset_atIndex_(mbuf_out, 0, 3)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 4)

        grid = Metal.MTLSize(batch_size, w, h)
        max_t = int(self.pipe_conv_float_bias.maxTotalThreadsPerThreadgroup())
        tx, ty, tz = 1, 16, 16
        while tx * ty * tz > max_t:
            if tz > 1:
                tz //= 2
            elif ty > 1:
                ty //= 2
            else:
                break
        tpg = Metal.MTLSize(tx, min(ty, w if w > 0 else 1), min(tz, h if h > 0 else 1))
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def init_max_float(self):
        fn = self.lib.newFunctionWithName_("max_float")
        self.pipe_max_float, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)

    def max_float(self, batch_size, mbuf_in, mbuf_out, mbuf_mask, ch, w, h):
        # params (uint32 x3), align=True to match MSL struct layout
        params = np.zeros(1, dtype=np.dtype([
            ("ch", np.uint32),
            ("w",  np.uint32),
            ("h",  np.uint32),
        ], align=True))
        params["ch"] = ch
        params["w"]  = w
        params["h"]  = h

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_max_float)
        enc.setBuffer_offset_atIndex_(mbuf_in, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_out, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_mask, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        # OpenCL: (bi, y, x) -> Metal gid.x=bi, gid.y=y, gid.z=x
        grid = Metal.MTLSize(batch_size, h, w)

        max_t = int(self.pipe_max_float.maxTotalThreadsPerThreadgroup())
        tx, ty, tz = 1, 16, 16
        while tx * ty * tz > max_t:
            if tz > 1:
                tz //= 2
            elif ty > 1:
                ty //= 2
            else:
                break
        tpg = Metal.MTLSize(tx, min(ty, h if h > 0 else 1), min(tz, w if w > 0 else 1))

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    
    def init_max_bp_float(self):
        #self.pipe_max_bp_float = self._make_pipeline("max_bp_float")
        fn = self.lib.newFunctionWithName_("max_bp_float")
        self.pipe_max_bp_float, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
    
    def max_bp_float(self, batch_size, mbuf_delta, mbuf_mask, mbuf_grad, ch, w, h):
        # params (int32 x3)
        p = np.array([ch, w, h], dtype=np.int32)

        # params_buf に書き込み（あなたの実装が params_buf 方式なら）
        mv = self.params_buf.contents().as_buffer(16)
        mv[:12] = p.tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_max_bp_float)

        enc.setBuffer_offset_atIndex_(mbuf_delta, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_mask,  0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_grad,  0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSizeMake(batch_size, h, w)
        #grid = Metal.MTLSize(batch_size, h, w)
        
        # threadgroup は無難に (1, 8, 8) など（maxTotalThreadsPerThreadgroup 内で）
        #tg = self.mtl.MTLSizeMake(1, 8, 8)
        tg = Metal.MTLSizeMake(1, 8, 8)

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()
        
    def init_conv4_back_input(self):
        fn = self.lib.newFunctionWithName_("conv4_relu_back_input")
        self.pipe_conv4_back_input, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        
    def init_conv4_back_weight(self):
        fn = self.lib.newFunctionWithName_("conv4_relu_back_weight")
        self.pipe_conv4_back_weight, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

    def init_fc_hidden_delta_relu(self):
        fn = self.lib.newFunctionWithName_("fc_hidden_delta_relu")
        self.pipe_fc_hidden_delta_relu, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)

    def init_fc_weight_grad(self):
        fn = self.lib.newFunctionWithName_("fc_weight_grad")
        self.pipe_fc_weight_grad, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)

    def init_fc_output_delta(self):
        fn = self.lib.newFunctionWithName_("fc_output_delta")
        self.pipe_fc_output_delta, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)

    def conv4_back_input(self, batch_size, mbuf_dy, mbuf_out, mbuf_w, mbuf_dx, w, h, ch, filter):
        params = np.zeros(1, dtype=np.dtype([
            ("batch",  np.int32),
            ("w",      np.int32),
            ("h",      np.int32),
            ("ch",     np.int32),
            ("filter", np.int32),
        ], align=True))
        params["batch"] = batch_size
        params["w"] = w
        params["h"] = h
        params["ch"] = ch
        params["filter"] = filter

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_conv4_back_input)
        enc.setBuffer_offset_atIndex_(mbuf_dy,  0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_out, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_w,   0, 2)
        enc.setBuffer_offset_atIndex_(mbuf_dx,  0, 3)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 4)

        grid = Metal.MTLSize(batch_size, w, h)

        max_t = int(self.pipe_conv4_back_input.maxTotalThreadsPerThreadgroup())
        tx, ty, tz = 1, 16, 16
        while tx * ty * tz > max_t:
            if tz > 1:
                tz //= 2
            elif ty > 1:
                ty //= 2
            else:
                break
            #
        # while
        
        tpg = Metal.MTLSize(tx, min(ty, w if w > 0 else 1), min(tz, h if h > 0 else 1))

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def conv4_back_weight(self, batch_size, mbuf_xpad, mbuf_dy, mbuf_out, mbuf_dw, w, h, ch, filter):
        params = np.zeros(1, dtype=np.dtype([
            ("batch",  np.int32),
            ("w",      np.int32),
            ("h",      np.int32),
            ("ch",     np.int32),
            ("filter", np.int32),
        ], align=True))
        params["batch"] = batch_size
        params["w"] = w
        params["h"] = h
        params["ch"] = ch
        params["filter"] = filter

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_conv4_back_weight)
        enc.setBuffer_offset_atIndex_(mbuf_xpad, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_dy,   0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_out,  0, 2)
        enc.setBuffer_offset_atIndex_(mbuf_dw,   0, 3)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 4)

        grid = Metal.MTLSize(filter, ch, 9)

        max_t = int(self.pipe_conv4_back_weight.maxTotalThreadsPerThreadgroup())
        tx, ty, tz = 1, min(ch, 8), 9
        while tx * ty * tz > max_t:
            if tz > 1:
                tz //= 2
            elif ty > 1:
                ty //= 2
            else:
                break
            #
        #
        tpg = Metal.MTLSize(tx, max(1, ty), max(1, tz))

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def fc_hidden_delta_relu(self, batch_size, mbuf_next_delta, mbuf_next_w, mbuf_cur_out, mbuf_cur_delta,
                             cur_nodes, next_nodes):
        params = np.zeros(1, dtype=np.dtype([
            ("batch", np.int32),
            ("cur_nodes", np.int32),
            ("next_nodes", np.int32),
        ], align=True))
        params["batch"] = batch_size
        params["cur_nodes"] = cur_nodes
        params["next_nodes"] = next_nodes

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_fc_hidden_delta_relu)
        enc.setBuffer_offset_atIndex_(mbuf_next_delta, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_next_w,     0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_cur_out,    0, 2)
        enc.setBuffer_offset_atIndex_(mbuf_cur_delta,  0, 3)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 4)

        grid = Metal.MTLSize(batch_size, cur_nodes, 1)

        max_t = int(self.pipe_fc_hidden_delta_relu.maxTotalThreadsPerThreadgroup())
        tx, ty = 1, min(cur_nodes if cur_nodes > 0 else 1, 256)
        while tx * ty > max_t and ty > 1:
            ty //= 2
        tpg = Metal.MTLSize(tx, max(1, ty), 1)

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def fc_weight_grad(self, batch_size, mbuf_x_pre, mbuf_delta, mbuf_dw, in_nodes, out_nodes):
        params = np.zeros(1, dtype=np.dtype([
            ("batch", np.int32),
            ("in_nodes", np.int32),
            ("out_nodes", np.int32),
        ], align=True))
        params["batch"] = batch_size
        params["in_nodes"] = in_nodes
        params["out_nodes"] = out_nodes

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_fc_weight_grad)
        enc.setBuffer_offset_atIndex_(mbuf_x_pre,      0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_delta,      0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_dw,         0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSize(in_nodes, out_nodes, 1)

        max_t = int(self.pipe_fc_weight_grad.maxTotalThreadsPerThreadgroup())
        tx, ty = 1, min(out_nodes if out_nodes > 0 else 1, 256)
        while tx * ty > max_t and ty > 1:
            ty //= 2
        tpg = Metal.MTLSize(tx, max(1, ty), 1)

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()


    def fc_output_delta(self, batch_size, num_nodes, mbuf_softmax, mbuf_label, mbuf_delta):
        total = int(batch_size * num_nodes)
        params = np.zeros(1, dtype=np.dtype([
            ("n", np.int32),
        ], align=True))
        params["n"] = total

        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_fc_output_delta)
        enc.setBuffer_offset_atIndex_(mbuf_softmax,    0, 0)
        enc.setBuffer_offset_atIndex_(mbuf_label,      0, 1)
        enc.setBuffer_offset_atIndex_(mbuf_delta,      0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSize(total, 1, 1)
        tpg = Metal.MTLSize(min(self.pipe_fc_output_delta.maxTotalThreadsPerThreadgroup(), max(1, total), 256), 1, 1)

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    
def main():
    m = LMetal()
    m.init_test_func()
    m.init_calc_mac_relu()
    m.init_scale_layer()
    m.init_softmax()
    m.init_cross_entropy()
    
    n = 4096 # 64 * 64
    a = (np.random.uniform(size=n).astype(np.float32))
    
    a_buf = m.alloc_buf_from_array(a)
    b_buf = m.alloc_buf(a.nbytes)
    
    m.test_func(a_buf, b_buf, n)
    out = np.frombuffer(b_buf.contents().as_buffer(b_buf.length()), dtype=np.float32)
    #ok = np.allclose(out, a * 2, rtol=1e-3, atol=1e-3)
    #print("OK:", ok)
    print("in[:8] :", a[:8])
    print("out[:8]:", out[:8])
    
    
    print(type(a.nbytes))
    
    c = (np.random.uniform(size=n).astype(np.float32))
    x = m.alloc_buf_from_array(c)
    d = (np.random.uniform(size=n).astype(np.float32))
    w = m.alloc_buf_from_array(d)
    y = m.alloc_buf(n)
    #m.calc_mac_relu(x, w, y, 16, 16, 1)
    
    #out = np.frombuffer(y.contents().as_buffer(y.length()), dtype=np.float32)
    #print("y:", out[:8])
    
    print(m.device.supportsFeatureSet_(Metal.MTLFeatureSet_macOS_GPUFamily1_v3))
    return 0
    

#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)

