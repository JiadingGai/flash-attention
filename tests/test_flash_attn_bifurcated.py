import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_with_kvcache,
)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import _get_block_size_n
from flash_attn.layers.rotary import apply_rotary_emb
import copy
import time
import statistics
import numpy as np

# consolas 17b
dtype = torch.float16
device = "cuda"

def construct_local_mask_gai(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = seqlen_q
    print("window_size = ", window_size)
    assert window_size[0] == -1, "Only support infinite window size[0]"
    return col_idx > row_idx + sk - sq + window_size[1]

def attention_ref_gai(
    q,
    k,
    v,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    if causal:
        window_size = (window_size[0], 0)
    else:
        window_size = (-1,-1)
    dtype_og = q.dtype
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask_gai(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            window_size=window_size,
            key_padding_mask=key_padding_mask,
            device=q.device
        )
        scores.masked_fill_(local_mask, float("-inf"))
        # print("JiadingGAI: local_mask = \n", local_mask[:,0,:,:].int())
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

def attention_ref_bifur(
    q,
    kc,
    kd,
    vc,
    vd,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    bs = q.shape[0]

    if causal:
        window_size = (window_size[0], 0)
    else:
        window_size = (-1,-1)
    dtype_og = q.dtype
    kc = repeat(kc, "b s h d -> b s (h g) d", g=q.shape[2] // kc.shape[2])
    vc = repeat(vc, "b s h d -> b s (h g) d", g=q.shape[2] // vc.shape[2])
    kd = repeat(kd, "b s h d -> b s (h g) d", g=q.shape[2] // kd.shape[2])
    vd = repeat(vd, "b s h d -> b s (h g) d", g=q.shape[2] // vd.shape[2])
    # k = torch.concat((kc.repeat([bs, 1, 1, 1]), kd), dim=1)
    # v = torch.concat((vc.repeat([bs, 1, 1, 1]), vd), dim=1)
    d = q.shape[-1]
    # scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    scores_context = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), kc)
    scores_decoded = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), kd)
    scores = torch.concat([scores_context, scores_decoded], dim=-1)
    seqlen_q, seqlen_k = scores.shape[2], scores.shape[3]
    seqlen_context = kc.shape[1]

    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask_gai(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            window_size=window_size,
            key_padding_mask=key_padding_mask,
            device=q.device
        )
        scores.masked_fill_(local_mask, float("-inf"))
        # print("JiadingGAI: local_mask = \n", local_mask[:,0,:,:].int())
    attention = torch.softmax(scores, dim=-1).to(vc.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    #output = torch.einsum("bhts,bshd->bthd", attention, v)
    output_context = torch.einsum("bhts,bshd->bthd", attention[:, :, :, :seqlen_context], vc)
    output_decoded = torch.einsum("bhts,bshd->bthd", attention[:, :, :, seqlen_context:], vd)
    output = output_context + output_decoded
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

def plot_latency_csv(latency_csv):
    import matplotlib.pyplot as plt
    for c,b,f in latency_csv:
        print(c, ",", b, ",", f)
    ctx_ = [x[0] for x in latency_csv]
    ba2_ = [x[1] for x in latency_csv]
    fa2_ = [x[2] for x in latency_csv]
    plt.xlabel('Cache Context Length')
    plt.ylabel('Latency in microseconds')
    plt.title('BA+FA vs FA: attention kernel latency comparison in 17B v3')
    plt.plot(ctx_, ba2_, 'r^--', label='BA+FA', markersize=3., linewidth=1)
    plt.plot(ctx_, fa2_, 'g^--', label='FA', markersize=3., linewidth=1)
    plt.legend()
    plt.savefig('__plot_timing.png', dpi=600)

class TestFlashFwdBifurcatedAttention:
    @pytest.mark.parametrize("dtype", [torch.float16, ])
    @pytest.mark.parametrize("rotary_dim", [0, ])
    #@pytest.mark.parametrize("rotary_dim", [0, 128])
    @pytest.mark.parametrize("seqlen", [7,10])
    @pytest.mark.parametrize("context_seqlen", [1536, ])
    #TODO: Reference attention kernel needs rope embedding support.
    @pytest.mark.xfail
    def test_compare_with_reference0(self, dtype, seqlen, context_seqlen, rotary_dim, device="cuda", use_bifurcated_attention=True):
        # sanity check test
        # set seed
        torch.random.manual_seed(123)

        #bs = 5
        #nheads = 2
        #nheads_k = 1
        #seqlen = 3
        #hdim = 32 # flash attention only supports hidden dim that is 32, 64, 96 ...
        #context_seqlen = 6
        #kvcache_seqlen = 13
        #max_gen_seqlen = kvcache_seqlen - context_seqlen
        #cache_seqlens = torch.tensor([6,8,9,9,10], dtype=torch.int32, device=device)
        #IsCausal = True

        bs = 5
        nheads = 64
        nheads_k = 8
        # seqlen = 7
        hdim = 128
        context_seqlen = context_seqlen
        kvcache_seqlen = 2904
        max_gen_seqlen = kvcache_seqlen - context_seqlen
        cache_seqlens = torch.tensor([1796, 1816, 1856, 1916, 1896], device=device, dtype=torch.int32)
        IsCausal = True

        # simulate 16K+ context
        #bs = 30
        #nheads = 64
        #nheads_k = 8
        #seqlen = 7
        #hdim = 128
        #context_seqlen = 16384
        #kvcache_seqlen = 17000
        #max_gen_seqlen = kvcache_seqlen - context_seqlen
        ## cache_seqlens = torch.tensor([16586, 16786, 16865, 16856, 16912], device=device, dtype=torch.int32)
        #cache_seqlens = torch.randint(16586, 16912, (bs,), device=device, dtype=torch.int32)
        #print(cache_seqlens)
        #IsCausal = True

        # parameters
        seqlen_q = seqlen
        seqlen_k = kvcache_seqlen
        seqlen_new = seqlen_q
        causal = True
        rotary_interleaved = True
        new_kv = True

        q = torch.randn([bs,seqlen,nheads,hdim], device=device, dtype=dtype)
        k = torch.randn([bs,seqlen,nheads_k,hdim], device=device, dtype=dtype)
        v = torch.randn([bs,seqlen,nheads_k,hdim], device=device, dtype=dtype)

        ## single context (bs=1)
        Kcontext_cache = torch.randn([1,context_seqlen,nheads_k,hdim], device=device, dtype=dtype)
        Vcontext_cache = torch.randn([1,context_seqlen,nheads_k,hdim], device=device, dtype=dtype)

        ## the decoding cache is batched (preallocated to the max seqlen (i.e., kvcache_seqlen-context_seqlen)
        Kdecode_cache = torch.randn([bs,kvcache_seqlen-context_seqlen,nheads_k,hdim], device=device, dtype=dtype)
        Vdecode_cache = torch.randn([bs,kvcache_seqlen-context_seqlen,nheads_k,hdim], device=device, dtype=dtype)

        Kcache = torch.concat((Kcontext_cache.repeat([bs, 1, 1, 1]), Kdecode_cache), dim=1)
        Vcache = torch.concat((Vcontext_cache.repeat([bs, 1, 1, 1]), Vdecode_cache), dim=1)

        if rotary_dim > 0:
            angle = (
                torch.rand(
                    seqlen_k,
                    rotary_dim // 2,
                    device=device,
                )
                * 2
                * math.pi
            )
            cos = torch.cos(angle).to(dtype=dtype)
            sin = torch.sin(angle).to(dtype=dtype)
            if causal or local:
                q_ro = apply_rotary_emb(
                    q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
                )
            else:
                q_ro = rearrange(
                    apply_rotary_emb(
                        rearrange(q, "b s h d -> b 1 (s h) d"),
                        cos,
                        sin,
                        seqlen_offsets=cache_seqlens,
                        interleaved=rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=seqlen_q,
                )
            # q_ro = q
            k_ro = apply_rotary_emb(
                k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k

        out = flash_attn_with_kvcache(
            q,
            k_cache=Kcache, # will be muted
            v_cache=Vcache, # will be muted
            k_cache_decoded=Kdecode_cache, # will be muted
            v_cache_decoded=Vdecode_cache, # will be muted
            k_cache_context=Kcontext_cache,
            v_cache_context=Vcontext_cache,
            k=k,
            v=v,
            rotary_cos=cos,
            rotary_sin=sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=None,
            cache_leftpad=None,
            block_table=None,
            causal=IsCausal, #False
            window_size=[-1,0], # [-1,-1]
            # causal=False,
            # window_size=[-1,-1],
            rotary_interleaved=True,
            alibi_slopes=None,
            num_splits=0, # Jiading should we support num_splits?
            use_bifurcated_attention=use_bifurcated_attention,
        )

        ## SANITY CHECKS:
        ## 1. knew and vnew are copied to Kdecode_cache, Vdecode_cache
        #for i in range (seqlen):
        #  for b in range(bs):
        #    x = Kcache[b, cache_seqlens[b] + i, :, :]
        #    y = k[b, i, :, :]
        #    assert torch.allclose(x, y)
        #    x = Vcache[b, cache_seqlens[b] + i, :, :]
        #    y = v[b, i, :, :]
        #    assert torch.allclose(x, y)

        #for i in range (seqlen):
        #  for b in range(bs):
        #    x = Kdecode_cache[b, cache_seqlens[b] + i - context_seqlen, :, :]
        #    y = k[b, i, :, :]
        #    assert torch.allclose(x, y)
        #    x = Vdecode_cache[b, cache_seqlens[b] + i - context_seqlen, :, :]
        #    y = v[b, i, :, :]
        #    assert torch.allclose(x, y)
        ## 2. Kcache and Kdecode_cache are identical up to actual_seqlen
        #for b in range(bs):
        #  actual_seqlen = cache_seqlens[b] + seqlen
        #  for i in range (actual_seqlen - context_seqlen):
        #      x = Kcache[b, i + context_seqlen, :, :]
        #      y = Kdecode_cache[b, i, :, :]
        #      assert torch.allclose(x, y)
        #      x = Vcache[b, i + context_seqlen, :, :]
        #      y = Vdecode_cache[b, i, :, :]
        #      assert torch.allclose(x, y)

        # flash attn's attention_ref
        arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
        cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
        key_padding_mask = arange < cache_seqlens_expanded + seqlen_new
        # print("key_padding_mask = \n", key_padding_mask.int())
        if new_kv:
            update_mask = torch.logical_and(
                cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
            )
            print(cache_seqlens_expanded)
            print(arange)
            Kcache[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
            Vcache[update_mask] = rearrange(v, "b s ... -> (b s) ...")

        out_ref, _ = attention_ref_gai(
            q,
            Kcache,
            Vcache,
            causal=IsCausal,
            key_padding_mask=key_padding_mask,
            window_size=[-1,0]
        )
        zzz = torch.isnan(out_ref)
        if torch.any(zzz):
          assert False, "nan present in out_ref"
        zzz = torch.isnan(out)
        if torch.any(zzz):
          assert False, "nan present in out"
        print("===================== out vs out_ref ====================")
        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
        print(torch.allclose(out, out_ref, rtol=1e-08, atol=9.8e-04))
        # manuall inspect the max diff in out vs out_ref
        diff = (out - out_ref).abs()
        xxx = torch.argwhere(diff>=diff.max())
        # FIXME: ugly indexing tensor with another tensor (use gather?)
        print("out_ref@max_diff = ", out_ref[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])
        print("out@max_diff     = ", out[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])

        # bifur's attention_ref
        Kdecode_cache_updated = Kcache[:, context_seqlen:, :, :]
        Vdecode_cache_updated = Vcache[:, context_seqlen:, :, :]
        out_bifur_ref, _ = attention_ref_bifur(
            q,
            Kcontext_cache,
            Kdecode_cache_updated,
            Vcontext_cache,
            Vdecode_cache_updated,
            key_padding_mask=key_padding_mask,
            causal=IsCausal,
            window_size=(-1, 0),  # -1 means infinite window size
        )
        print("===================== out vs out_bifur_ref ====================")
        print(f"Output max diff: {(out - out_bifur_ref).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
        print(torch.allclose(out, out_bifur_ref, rtol=1e-08, atol=9.8e-04))
        # manuall inspect the max diff in out vs out_ref
        diff_bifur = (out - out_bifur_ref).abs()
        xxx = torch.argwhere(diff_bifur >= diff_bifur.max())
        # FIXME: ugly indexing tensor with another tensor (use gather?)
        print("out_bifur_ref@max_diff = ", out_bifur_ref[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]], ", coordinate = ", xxx[0])
        print("out@max_diff           = ", out[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]], ", coordinate = ", xxx[0])

        assert torch.allclose(out, out_ref, rtol=1e-08, atol=9.8e-04)

    # Compare with fa: non-divisible context length
    @pytest.mark.parametrize("dtype", [torch.float16, ])
    @pytest.mark.parametrize("rotary_dim", [128])
    @pytest.mark.parametrize("seqlen", list(range(7,30,5)))
    @pytest.mark.parametrize("bs", [8,])
    @pytest.mark.parametrize("context_seqlen_max", [3980, ])
    @pytest.mark.parametrize("decoded_seqlen_max", [512, ])
    def test_compare_fa_with_bifur(self, dtype, bs, seqlen, context_seqlen_max, decoded_seqlen_max, rotary_dim):
        decoded_high = decoded_seqlen_max - seqlen
        decoded_low = 1
        context_high = context_seqlen_max - 1
        context_low = 1
        cache_seqlens_decoded = torch.randint(low=decoded_low, high=decoded_high, size=(bs,), device=device, dtype=torch.int32)
        # cache_seqlens_decoded = torch.randint(low=110, high=111, size=(1,), device=device, dtype=torch.int32).repeat(bs)
        print(f"cache_seqlens_decoded={cache_seqlens_decoded} randomly generated between 1 and {decoded_seqlen_max-seqlen}")
        cache_seqlens_context = torch.randint(low=context_low, high=context_high, size=(1,), device=device, dtype=torch.int32).repeat(bs)
        # cache_seqlens_context = torch.randint(low=128, high=129, size=(1,), device=device, dtype=torch.int32).repeat(bs)

        # temp workaround to generate divisible context seqlen. remove once non-divisble context lengths are supported.
        import random
        xxx = random.randint(1, context_seqlen_max-256)
        xxx = int(xxx / 128) * 128
        cache_seqlens_context = torch.tensor([xxx,], device=device, dtype=torch.int32).repeat(bs)

        # cache_seqlens_context = torch.randint(low=2688, high=2689, size=(1,), device=device, dtype=torch.int32).repeat(bs)
        print(f"cache_seqlens_context ={cache_seqlens_context} randomly generated between 1 and {context_seqlen_max-1}")
        bifur_time, og_fa_time = self.template_timing_run(dtype=torch.float16,
                                                          device="cuda",
                                                          bs=bs,
                                                          seqlen=seqlen,
                                                          context_seqlen_max=context_seqlen_max,
                                                          decoded_seqlen_max=decoded_seqlen_max,
                                                          rotary_dim=rotary_dim,
                                                          num_runs=1,
                                                          cache_seqlens_decoded=cache_seqlens_decoded,
                                                          cache_seqlens_context=cache_seqlens_context,
                                                          )

    ## Sweep Testing: slow test that may take some time.
    @pytest.mark.xfail
    def test2(self, rotary_dim=128):
        total_cases = []
        failed_cases = []
        for context_seqlen in range(128, 2*16384+1, 128):
          # context_seqlen = 1536
          for seqlen in range(1,135+1,3):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"ZZZZZZZZZZZZZZZZZ==== context_seqlen={context_seqlen}, seqlen={seqlen}, bs=10, rotary_dim={rotary_dim} ====ZZZZZZZZZZZZZZZZZZZZ")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # ret_code = self.test0(dtype=torch.float16, seqlen=seqlen, context_seqlen=context_seqlen, use_bifurcated_attention=use_bifurcated_attention)
            ret_code = self.template_run(dtype=torch.float16, bs=10, seqlen=seqlen, context_seqlen=context_seqlen, kvcache_seqlen=context_seqlen+1024, rotary_dim=rotary_dim)
            assert(ret_code)
            total_cases.append(f"context_seqlen={context_seqlen}, seqlen={seqlen}")
            if ret_code:
                print(f"seqlen={seqlen}: PASS")
            else:
                print(f"context_seqlen={context_seqlen}, seqlen={seqlen}: FAIL")
                failed_cases.append(f"context_seqlen={context_seqlen}, seqlen={seqlen}: FAIL")
                # assert False

        # print("FAILED CASES: \n", failed_cases)
        for failed_case in failed_cases:
            print("FAILED CASE: \n", failed_case)

        print("FAILED CASES: \n", len(failed_cases))
        print("TOTAL CASES: ", len(total_cases))

    # Compare with fa: non-divisible context length
    @pytest.mark.parametrize("dtype", [torch.float16, ])
    @pytest.mark.parametrize("rotary_dim", [128,])
    @pytest.mark.parametrize("seqlen", list(range(5,6)))
    @pytest.mark.parametrize("bs", [50,])
    #@pytest.mark.parametrize("context_seqlen_max", [4096, 8192, 8192*2, 8192*4, 8192*8])
    # @pytest.mark.parametrize("context_seqlen_max", [1024, 2*1024, 4*1024, 8 * 1024, 16*1024, 32 * 1024])
    @pytest.mark.parametrize("context_seqlen_max", [8 * 1024, ])
    # @pytest.mark.parametrize("context_seqlen_max", list(range(128, 8192)))
    @pytest.mark.parametrize("decoded_seqlen_max", [512, ])
    def test_timing(self, dtype, bs, seqlen, context_seqlen_max, decoded_seqlen_max, rotary_dim):
        # set seed
        torch.random.manual_seed(123)

        # For timing measurement, which we will use the maximally allowed lengths for context and decoded.
        decoded_high = decoded_seqlen_max-seqlen
        decoded_low = 1
        context_high = context_seqlen_max-1
        context_low = 1

        latency_csv = []
        __debug_logs = []
        for context_low in range(1, 8192+1, 157):
            cache_seqlens_decoded = torch.randint(low=decoded_low, high=decoded_high, size=(bs,), device=device, dtype=torch.int32)
            # cache_seqlens_decoded = torch.randint(low=255, high=255+1, size=(1,), device=device, dtype=torch.int32).repeat(bs)
            print(f"cache_seqlens_decoded={cache_seqlens_decoded} randomly generated between 1 and {decoded_seqlen_max-seqlen}")
            # cache_seqlens_context = torch.randint(low=context_low, high=context_low+1, size=(1,), device=device, dtype=torch.int32).repeat(bs)
            # cache_seqlens_context = torch.randint(low=127, high=128, size=(1,), device=device, dtype=torch.int32).repeat(bs)
            cache_seqlens_context = context_low
            print(f"cache_seqlens_context ={cache_seqlens_context} randomly generated between 1 and {context_seqlen_max-1}")

            # temp workaround to generate divisible context seqlen. remove once non-divisble context lengths are supported.
            #import random
            ##xxx = random.randint(1, context_seqlen_max-256)
            #xxx = cache_seqlens_context[0]
            #xxx = int(xxx / 128) * 128
            #cache_seqlens_context = torch.tensor([xxx,], device=device, dtype=torch.int32).repeat(bs)

            print(f"\n====== Benchmarking context_seqlen_max={context_seqlen_max}, seqlen={seqlen}, bs={bs}, cache_seqlens_context={context_low} ======\n")
            bifur_time, og_fa_time, log_msg  = self.template_timing_run(dtype=dtype,
                                                              device="cuda",
                                                              bs=bs,
                                                              seqlen=seqlen,
                                                              context_seqlen_max=context_seqlen_max,
                                                              decoded_seqlen_max=decoded_seqlen_max,
                                                              rotary_dim=rotary_dim,
                                                              num_runs=1,
                                                              cache_seqlens_context=cache_seqlens_context,
                                                              cache_seqlens_decoded=cache_seqlens_decoded,
                                                              )
            # convert time from seconds to microseconds:
            bifur_time *= 1000000.0
            og_fa_time *= 1000000.0
            print(f"bifurcated+FA: {bifur_time:.6f} microseconds.")
            print(f"  original FA: {og_fa_time:.6f} microseconds.")
            latency_csv.append((context_low, bifur_time, og_fa_time))
            __debug_logs.append(log_msg)


        plot_latency_csv(latency_csv)
        for _log in __debug_logs:
          print(_log)

    def template_timing_run(self, dtype=torch.float16, device="cuda", bs=5, seqlen=7, context_seqlen_max=1536, decoded_seqlen_max=2904, rotary_dim=128, num_runs=1, cache_seqlens_context=1536, cache_seqlens_decoded=2904):
        """
        """

        # we do not test sliding window attention here.
        local = False

        #nheads = 2
        #nheads_k = 1
        #seqlen = 3
        #hdim = 32 # flash attention only supports hidden dim that is 32, 64, 96 ...
        #context_seqlen = 6
        #kvcache_seqlen = 13
        #max_gen_seqlen = kvcache_seqlen - context_seqlen
        #cache_seqlens = torch.tensor([6,8,9,9,10], dtype=torch.int32, device=device)
        #IsCausal = True

        nheads = 32
        nheads_k = 4
        # seqlen = 7
        hdim = 128
        # context_seqlen = 1536
        # kvcache_seqlen = 2904

        # context_seqlen_max = 3980
        # decoded_seqlen_max = 512
        # TODO: the kvcache_seqlen arg will be replaced by kvcache_seqlen_max.
        kvcache_seqlen_max = context_seqlen_max + decoded_seqlen_max

        IsCausal = True

        # simulate 16K+ context
        #nheads = 64
        #nheads_k = 8
        #seqlen = 7
        #hdim = 128
        #context_seqlen = 16384
        #kvcache_seqlen = 17000
        #max_gen_seqlen = kvcache_seqlen - context_seqlen
        ## cache_seqlens = torch.tensor([16586, 16786, 16865, 16856, 16912], device=device, dtype=torch.int32)
        #cache_seqlens = torch.randint(16586, 16912, (bs,), device=device, dtype=torch.int32)
        #print(cache_seqlens)
        #IsCausal = True

        ## PARAMETERS
        seqlen_q = seqlen
        # seqlen_k is the max kvcache_seqlen (pre-allocated max size).
        seqlen_k = kvcache_seqlen_max
        seqlen_new = seqlen_q
        rotary_interleaved = True
        new_kv = True

        q = torch.randn([bs,seqlen,nheads,hdim], device=device, dtype=dtype)
        k = torch.randn([bs,seqlen,nheads_k,hdim], device=device, dtype=dtype)
        v = torch.randn([bs,seqlen,nheads_k,hdim], device=device, dtype=dtype)

        ## single context (bs=1)
        Kcontext_cache = torch.randn([1,context_seqlen_max,nheads_k,hdim], device=device, dtype=dtype)
        Vcontext_cache = torch.randn([1,context_seqlen_max,nheads_k,hdim], device=device, dtype=dtype)
        # Set oob values to nan to help debug
        oob_val = torch.nan
        Kcontext_cache[:, cache_seqlens_context:, :, :] = oob_val
        Vcontext_cache[:, cache_seqlens_context:, :, :] = oob_val



        ## the decoding cache is batched (preallocated to the max seqlen (i.e., kvcache_seqlen-context_seqlen)
        Kdecode_cache = torch.randn([bs,decoded_seqlen_max,nheads_k,hdim], device=device, dtype=dtype)
        Vdecode_cache = torch.randn([bs,decoded_seqlen_max,nheads_k,hdim], device=device, dtype=dtype)
        # Set oob values to nan to help debug
        for bid in range(0, bs):
            Kdecode_cache[bid, cache_seqlens_decoded[bid]:, :, :] = oob_val
            Vdecode_cache[bid, cache_seqlens_decoded[bid]:, :, :] = oob_val

        cache_seqlens = cache_seqlens_context + cache_seqlens_decoded

        if rotary_dim > 0:
            angle = (
                torch.rand(
                    seqlen_k,
                    rotary_dim // 2,
                    device=device,
                )
                * 2
                * math.pi
            )
            cos = torch.cos(angle).to(dtype=dtype)
            sin = torch.sin(angle).to(dtype=dtype)
            if IsCausal or local:
                q_ro = apply_rotary_emb(
                    q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
                )
            else:
                q_ro = rearrange(
                    apply_rotary_emb(
                        rearrange(q, "b s h d -> b 1 (s h) d"),
                        cos,
                        sin,
                        seqlen_offsets=cache_seqlens,
                        interleaved=rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=seqlen_q,
                )
            # q_ro = q
            k_ro = apply_rotary_emb(
                k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k

        torch.set_printoptions(sci_mode=False)
        # bifurcated+flash attention
        bifur_time_list = []
        for xxx in range(num_runs):
          start_time = time.time()
          out_bf, lse_bf = flash_attn_with_kvcache(
              q,
              k_cache=None, # not needed by bifurcated attention
              v_cache=None, # not needed by bifurcated attention
              k_cache_decoded=Kdecode_cache, # will be muted
              v_cache_decoded=Vdecode_cache, # will be muted
              k_cache_context=Kcontext_cache,
              v_cache_context=Vcontext_cache,
              k=k,
              v=v,
              rotary_cos=cos,
              rotary_sin=sin,
              cache_seqlens=None,
              cache_seqlens_context=cache_seqlens_context,
              cache_seqlens_decoded=cache_seqlens_decoded,
              cache_batch_idx=None,
              cache_leftpad=None,
              block_table=None,
              causal=IsCausal, #False
              window_size=[-1,0], # [-1,-1]
              # causal=False,
              # window_size=[-1,-1],
              rotary_interleaved=True,
              alibi_slopes=None,
              num_splits=2, # Jiading should we support num_splits?
              use_bifurcated_attention=True,
              return_softmax_lse=True,
          )
          # print(out_bf[8,0,45,:])
          torch.cuda.synchronize()
          end_time = time.time()
          bifur_time_list.append(end_time - start_time)
        bifur_time = statistics.median(bifur_time_list)

        print("=== bifur flash attention done: sleep 3 seconds ===")
        time.sleep(3)
        print("=== original flash attention ===")

        # original flash attention
        ## Combine context and decoded caches into an aggregated cache. Kcache and Vcache used by the original FA.
        # Kcache = torch.concat((Kcontext_cache[:,:cache_seqlens_context,:,:].repeat([bs, 1, 1, 1]), Kdecode_cache), dim=1)
        # Vcache = torch.concat((Vcontext_cache[:,:cache_seqlens_context,:,:].repeat([bs, 1, 1, 1]), Vdecode_cache), dim=1)
        Kcache_tmp = torch.concat((Kcontext_cache[:,:cache_seqlens_context,:,:].repeat([bs, 1, 1, 1]), Kdecode_cache), dim=1)
        Vcache_tmp = torch.concat((Vcontext_cache[:,:cache_seqlens_context,:,:].repeat([bs, 1, 1, 1]), Vdecode_cache), dim=1)
        Kcache = torch.full([bs,context_seqlen_max+decoded_seqlen_max,nheads_k,hdim], float('nan'), device=device, dtype=dtype)
        Vcache = torch.full([bs,context_seqlen_max+decoded_seqlen_max,nheads_k,hdim], float('nan'), device=device, dtype=dtype)
        Kcache[:, :cache_seqlens_context + decoded_seqlen_max, :, :] = Kcache_tmp
        Vcache[:, :cache_seqlens_context + decoded_seqlen_max, :, :] = Vcache_tmp

        og_fa_time_list = []
        for xxx in range(num_runs):
            start_time = time.time()
            out_og, lse_og = flash_attn_with_kvcache(
                q=q,
                k_cache=Kcache, # will be muted
                v_cache=Vcache, # will be muted
                k_cache_decoded=None, # only needed by bifurated attn
                v_cache_decoded=None, # only needed by bifurated attn
                k_cache_context=None, # only needed by bifurated attn
                v_cache_context=None, # only needed by bifurated attn
                k=k,
                v=v,
                rotary_cos=cos,
                rotary_sin=sin,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=None,
                cache_leftpad=None,
                block_table=None,
                causal=IsCausal, #False
                window_size=[-1,0], # [-1,-1]
                # causal=False,
                # window_size=[-1,-1],
                rotary_interleaved=True,
                alibi_slopes=None,
                num_splits=0, # Jiading should we support num_splits?
                use_bifurcated_attention=False,
                return_softmax_lse=True,
            )
            # print(out_og[8,0,45,:])
            torch.cuda.synchronize()
            end_time = time.time()
            og_fa_time_list.append(end_time - start_time)
        og_fa_time = statistics.median(og_fa_time_list)

        #print("===================== out vs out_ref ====================")
        #print(f"Output max diff : {(out_bf - out_og).abs().max().item()}")
        #print(f"Output mean diff: {(out_bf - out_og).abs().mean().item()}")
        #print("allcose: ", torch.allclose(out_bf, out_og, rtol=1e-08, atol=9.8e-04))
        # manuall inspect the max diff in out vs out_ref
        diff = (out_bf - out_og).abs()
        xxx = torch.argwhere(diff>=diff.max())

        log_msg = f"[INFO] context length = {cache_seqlens_context}, decoded length = {cache_seqlens_decoded}"
        # check nan
        if torch.isnan(torch.flatten(out_bf)).any():
           log_msg += ": -------------------------------------------> failed with nan"
        elif not torch.allclose(out_bf, out_og, rtol=1e-08, atol=9.8e-04):
           print("[FAILED] cache_seqlens_context = ", cache_seqlens_context, ", cache_seqlens_decoded = ", cache_seqlens_decoded)
           # FIXME: ugly indexing tensor with another tensor (use gather?)
           print("out_og@max_diff = ", out_og[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])
           print("out_bf@max_diff = ", out_bf[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])
           log_msg += ": -------------------------------------------> failed with mismatch"
        else:
           print("out_og@max_diff = ", out_og[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])
           print("out_bf@max_diff = ", out_bf[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])
           log_msg += ". PASS!"

        assert not torch.isnan(torch.flatten(out_bf)).any()
        assert torch.allclose(out_bf, out_og, rtol=1e-08, atol=9.8e-04)
        return (bifur_time, og_fa_time, log_msg)

