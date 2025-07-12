/******************************************************************************
 * Copyright (c) 2024, AWS NGDE.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"
#include "bifurcated_utils.h"

namespace flash {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_bifurcated(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    // bifurcated attention assertion on unsupported options.
    assert(params.block_table == nullptr && "bifurcated attention does not support paged attention yet.");
    assert(Is_even_K && "BA2 does not support non-even K yet.");
    assert(!Is_local && "BA2 does not support sliding window attention yet.");
    //assert(!Split && "BA2 does not support split yet.");

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    /* if (threadIdx.x == 0) { */
    /*    printf("block (%d,%d,%d): seqlen_k_cache = %d, actual_seqlen_k = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, binfo.seqlen_k_cache, binfo.actual_seqlen_k); */
    /* } */
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    //(jiadingg): n_block_max is only used by bifurcated flash attention under the scenario of flash decoding.
    int n_block_max = std::min(cute::ceil_div(binfo.seqlen_k_cache_context, kBlockN) + cute::ceil_div(binfo.seqlen_k_cache_decoded + binfo.actual_seqlen_q, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // bifurcated attention
    const int n_block_max_context = (params.seqlen_k_context + kBlockN - 1) / kBlockN;
    // actual_n_block_max_context denotes the effective number of blocks that are actually used.
    // binfo.seqlen_k_cache_context is the same as binfo.actual_seqlen_k_cache_context (i.e., no
    // new kv tokens are appended to the end of context kv cache)
    const int actual_n_block_max_context = (binfo.seqlen_k_cache_context + kBlockN - 1) / kBlockN;
    const int n_block_max_decoded = (params.seqlen_k_decoded + kBlockN - 1) / kBlockN;
    const int actual_n_block_max_decoded = (binfo.actual_seqlen_k_cache_decoded + kBlockN - 1) / kBlockN;
    // it's important to distinguish actual_n_block_max and n_block_max for rotary embedding.
    const int actual_n_block_max = actual_n_block_max_context + actual_n_block_max_decoded;
    /* n_block_max = n_block_max_context + actual_n_block_max_decoded; */
    const int physical_n_block_max = n_block_max_context + n_block_max_decoded;
    const int logical_n_block_max = actual_n_block_max_context + actual_n_block_max_decoded;

    //sleep((int64_t)(6 * 1000ULL));
    //if (threadIdx.x == 1 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 2) {
    //   sleep((int64_t)((blockIdx.y + 1) * 10000ULL));
    //   printf("\n================== block info =====================\n");
    //   printf("binfo.sum_s_q = %d\n", binfo.sum_s_q);
    //   printf("binfo.sum_s_k = %d\n", binfo.sum_s_k);
    //   printf("binfo.actual_seqlen_q = %d\n", binfo.actual_seqlen_q);
    //   printf("binfo.leftpad_k = %d\n", binfo.leftpad_k);
    //   printf("binfo.seqlen_k_cache = %d\n", binfo.seqlen_k_cache);
    //   printf("binfo.actual_seqlen_k = %d\n", binfo.actual_seqlen_k);
    //   printf("binfo.seqlen_k_cache_context = %d\n", binfo.seqlen_k_cache_context);
    //   printf("binfo.seqlen_k_cache_decoded = %d\n", binfo.seqlen_k_cache_decoded);
    //   printf("binfo.actual_seqlen_k_cache_decoded = %d\n", binfo.actual_seqlen_k_cache_decoded);
    //   printf("binfo.actual_seqlen_q = %d\n", binfo.actual_seqlen_q);
    //   printf("num_n_splits = %d\n", num_n_splits);
    //   printf("n_blocks_per_split = %d\n", n_blocks_per_split);
    //   printf("n_block_max_context = %d\n", n_block_max_context);
    //   printf("actual_n_block_max_context = %d\n", actual_n_block_max_context);
    //   printf("n_block_max_decoded = %d\n", n_block_max_decoded);
    //   printf("actual_n_block_max_decoded = %d\n", actual_n_block_max_decoded);
    //   printf("n_block_min = %d\n", n_block_min);
    //   printf("n_block_max = %d\n", n_block_max);
    //   printf("actual_n_block_max = %d\n", actual_n_block_max);
    //   printf("physical_n_block_max = %d\n", physical_n_block_max);
    //   printf("\n================== block info =====================\n");
    //   sleep((int64_t)((blockIdx.y + 1) * 10000ULL));
    //}
    //sleep((int64_t)(6 * 100ULL));
    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // bifurcated attention
        //assert(false && "BA2 does not support n_block_max <= 0 yet.");

        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;

    // bifurcated attention
    // (jiading:flash decoding) for some split, [n_block_min, n_block_max) might cover only context blocks,
    // in those cases, we want row_offset_kcontext to start from the final context block this thread block
    // covers, not the last block from the entire context cache.
    const index_t row_offset_kcontext = (std::min(n_block_max, actual_n_block_max_context) - 1) * kBlockN * params.kcontext_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.kcontext_head_stride;
    const index_t row_offset_vcontext = (std::min(n_block_max, actual_n_block_max_context) - 1) * kBlockN * params.vcontext_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.vcontext_head_stride;

    assert(n_block_max_decoded > 0);
    const index_t row_offset_kdecoded = binfo.k_offset(params.kdecoded_batch_stride, params.kdecoded_row_stride, bidb_cache) +
                                        (std::min(n_block_max - actual_n_block_max_context, actual_n_block_max_decoded) - 1) * kBlockN * params.kdecoded_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.kdecoded_head_stride;
    const index_t row_offset_vdecoded = binfo.k_offset(params.vdecoded_batch_stride, params.vdecoded_row_stride, bidb_cache) +
                                        (std::min(n_block_max - actual_n_block_max_context, actual_n_block_max_decoded) - 1) * kBlockN * params.vdecoded_row_stride +
                                        (bidh / params.h_h_k_ratio) * params.vdecoded_head_stride;


    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    Tensor gKcontext = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.kcontext_ptr) + row_offset_kcontext),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                   make_stride(params.kcontext_row_stride, _1{}));

    Tensor gKdecoded = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.kdecoded_ptr) + row_offset_kdecoded),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                   make_stride(params.kdecoded_row_stride, _1{}));


    Tensor gVcontext = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vcontext_ptr) + row_offset_vcontext),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                   make_stride(params.vcontext_row_stride, _1{}));

    Tensor gVdecoded = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vdecoded_ptr) + row_offset_vdecoded),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                   make_stride(params.vdecoded_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgKcontext = gmem_thr_copy_QKV.partition_S(gKcontext);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKgKdecoded = gmem_thr_copy_QKV.partition_S(gKdecoded);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgVcontext = gmem_thr_copy_QKV.partition_S(gVcontext);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVgVdecoded = gmem_thr_copy_QKV.partition_S(gVdecoded);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // Copy from Knew to K, optionally apply rotary embedding.
    typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    // knew -> kdecoded copy: to replace the original Append_KV above.
    if constexpr (Append_KV) {
        // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
        // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
        // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.
        //const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
        //const index_t row_offset_cossin = ((logical_n_block_max - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
        //const int xxx = (binfo.seqlen_k_cache_context + binfo.seqlen_k_cache_decoded + kBlockN - 1) / kBlockN;
        //const index_t row_offset_cossin = ((xxx - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
        const index_t row_offset_cossin = (binfo.actual_seqlen_k_cache_context + (actual_n_block_max_decoded - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        // if (cute::thread(0, 0)) { printf("rotary_cos_ptr = %p, gCos.data() = %p, tRgCos.data() = %p, rotary_dim = %d\n", params.rotary_cos_ptr, gCos.data(), tRgCos.data(), params.rotary_dim); }
        // if (cute::thread(8, 0)) { print_tensor(gCos); }
        // if (cute::thread(0, 0)) { print_tensor(tRgCos); }

        // const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb)
        const index_t row_offset_knew = bidb * params.knew_batch_stride
            + ((actual_n_block_max_decoded - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        // const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb)
        const index_t row_offset_vnew = bidb * params.vnew_batch_stride
            + ((actual_n_block_max_decoded - 1) * kBlockN) * params.vnew_row_stride + (bidh / params.h_h_k_ratio) * params.vnew_head_stride;
        // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
        // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
        // This maps to accessing the first 64 rows of knew_ptr.
        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr)
                                                + row_offset_knew - binfo.seqlen_k_cache_decoded * params.knew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.knew_row_stride, _1{}));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data()); }
        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr)
                                                + row_offset_vnew - binfo.seqlen_k_cache_decoded * params.vnew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.vnew_row_stride, _1{}));
        Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
        Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)

        const int n_block_copy_min = std::max(n_block_min, (actual_n_block_max_context * kBlockN + binfo.seqlen_k_cache_decoded) / kBlockN);
        auto tKgK_data = tKgKdecoded.data();
        auto tVgV_data = tVgVdecoded.data();
        for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
            // Note that n_block points to the physical IDs of each block.
            flash::copy_w_min_idx<Is_even_K>(
                tVgVnew, tVgVdecoded, tKVcKV, tKVpKV,
                actual_n_block_max_context * kBlockN + binfo.actual_seqlen_k_cache_decoded - n_block * kBlockN,
                actual_n_block_max_context * kBlockN + binfo.seqlen_k_cache_decoded - n_block * kBlockN
            );
            tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
            if (params.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K>(
                    tKgKnew, tKgKdecoded, tKVcKV, tKVpKV,
                    actual_n_block_max_context * kBlockN + binfo.actual_seqlen_k_cache_decoded - n_block * kBlockN,
                    actual_n_block_max_context * kBlockN + binfo.seqlen_k_cache_decoded - n_block * kBlockN
                );
            } else {
                if (params.is_rotary_interleaved) {
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_interleaved<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgKdecoded, tRgCos, tRgSin, tKVcKV,
                        actual_n_block_max_context * kBlockN + binfo.actual_seqlen_k_cache_decoded - n_block * kBlockN,
                        actual_n_block_max_context * kBlockN + binfo.seqlen_k_cache_decoded - n_block * kBlockN,
                        params.d, params.rotary_dim
                    );
                    tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                    assert(false && "bifurcated attention only supports interleaved rotary embedding.");
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_contiguous<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV,
                        binfo.seqlen_k_cache_context + binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache_context + binfo.seqlen_k_cache - n_block * kBlockN,
                        params.d, params.rotary_dim
                    );
                    tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));
#endif
                }
            }
            tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
            if (block_table == nullptr) {
                tVgVdecoded.data() = tVgVdecoded.data() + (-int(kBlockN * params.v_row_stride));
                tKgKdecoded.data() = tKgKdecoded.data() + (-int(kBlockN * params.k_row_stride));
            } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                if (n_block > n_block_copy_min) {
                    const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                    const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                    const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                    const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                    const int table_diff = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
                    const int offset_diff = block_table_offset_next - block_table_offset_cur;
                    tVgV.data() = tVgV.data() + table_diff * params.v_batch_stride + offset_diff * params.v_row_stride;
                    tKgK.data() = tKgK.data() + table_diff * params.k_batch_stride + offset_diff * params.k_row_stride;
                }
#endif
            }
        }
        // Need this before we can read in K again, so that we'll see the updated K values.
        __syncthreads();
        tKgKdecoded.data() = tKgK_data;
        tVgVdecoded.data() = tVgV_data;
    }

    // Read Q from gmem to smem, optionally apply rotary embedding.
    if (!Append_KV || params.rotary_dim == 0) {
        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        /* const index_t row_offset_cossin = (binfo.seqlen_k_cache + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]) + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2); */
        const index_t row_offset_cossin = (binfo.seqlen_k_cache_context + binfo.seqlen_k_cache_decoded + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]) + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);
        // If not causal, all the queries get the same the cos/sin, taken at location seqlen_k_cache.
        // We do this by setting the row stride of gCos / gSin to 0.
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        if (params.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K>(
                tQgQ, tQsQ, tRgCos, tRgSin, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        } else {
            flash::copy_rotary_contiguous<Is_even_K>(
                tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        }
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    if (n_block < actual_n_block_max_context) {
        // use context block
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgKcontext, tKsK, tKVcKV, tKVpKV,
                                           binfo.actual_seqlen_k_cache_context - n_block * kBlockN);
    } else {
        // use decoded block
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgKdecoded, tKsK, tKVcKV, tKVpKV,
                                           binfo.actual_seqlen_k_cache_decoded + actual_n_block_max_context * kBlockN - n_block * kBlockN);
    }
    cute::cp_async_fence();
    //bifurcated_check_tensor(tKsK, n_block);

    // flash::cp_async_wait<0>();
    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
    // __syncthreads();

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    //TODO(jiadingg): fix up masking
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k_cache_decoded + actual_n_block_max_context * kBlockN, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    flash::Mask</*Is_causal*/false, /*Is_local*/false, /*Has_alibi*/false> mask_context(binfo.seqlen_k_cache_context, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    //
    // FIXME(Jiadingg): n_masking_steps used to be a constexpr (compile-time constant), so that the masking
    //                  step loop below can unroll at compile time. But in bifurcated attention, we need to
    //                  make n_masking_steps a value dependent on n_block_max_decoded so we had to make it
    //                  a runtime constant. Check the performance impact of making n_masking_steps a runtime
    //                  constant!!
    int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    // BIFURCATED ATTENTION: The Padding Approach:
    // context and decoded blocks are separate blocks (the last context block gets padded).
    // masking blocks have to be among the decoded blocks, therefore we might have 2 masking
    // steps but only one decoded block, e.g., when context=128, previously decoded=1, q=1:
    // n_block_max_decoded = 1, n_masking_steps = 2, so take std::min between the two.
    n_masking_steps = std::min(n_masking_steps, actual_n_block_max_decoded);
    //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 2) {
    //  printf("[pre-masking step] n_masking_steps=%d\n",  n_masking_steps);
    //}
    auto __copy_next_v_block = [&] __device__ (bool decrease = true) {
      if (n_block == (actual_n_block_max_context - 1)) {
#if 1
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN*/true>(
#else
        flash::copy</*Is_even_MN=*/true, Is_even_K>(
#endif
            gmem_tiled_copy_QKV, tVgVcontext, tVsV, tKVcKV, tKVpKV,
            binfo.actual_seqlen_k_cache_context - n_block * kBlockN
        );
      } else if (n_block < (actual_n_block_max_context - 1)) {
        if (block_table == nullptr) {
            tVgVcontext.data() = tVgVcontext.data() + (-int(kBlockN * params.v_row_stride));
        } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
                const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
                tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
#endif
        }
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVcontext, tVsV, tKVcKV, tKVpKV);
      } else {
        if (block_table == nullptr) {
            tVgVdecoded.data() = tVgVdecoded.data() + (-int(kBlockN * params.v_row_stride));
        } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
                const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
                tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
#endif
        }
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVdecoded, tVsV, tKVcKV, tKVpKV);
      }
    };

    // k block is prefetched => the n_block-th iteration should fetch the (n_block + 1)-th k block
    // v block is not prefetched => the n_block-th iteration should fetch the n_block-th v block
    auto __copy_next_k_block = [&] __device__ (bool decrease = true) {
        if (n_block > actual_n_block_max_context) {
            // Advance gK
            if (block_table == nullptr) {
                tKgKdecoded.data() = tKgKdecoded.data() + (-int(kBlockN * params.kdecoded_row_stride));
            } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =(n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
#endif
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKdecoded, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        } else if (n_block == actual_n_block_max_context) {
            //assert(n_block == n_block_max_context &&
            //       "[BIFURCATED ATTENTION] You are at the last decoded block, but block index does not match.");
            flash::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgKcontext, tKsK, tKVcKV, tKVpKV,
                binfo.actual_seqlen_k_cache_context - n_block * kBlockN
            );
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        } else {
            // Advance gK context
            if (block_table == nullptr) {
                tKgKcontext.data() = tKgKcontext.data() + (-int(kBlockN * params.kcontext_row_stride));
            } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =(n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
#endif
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKcontext, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }
    };

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        //TODO: documentation: n_block indexing through the actual used blocks from context and decoded.
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 2) {
        //  printf("[masking step] visiting n_block = %d, actual_n_block_max_context = %d,  kBlockN = %d, n_masking_steps=%d\n", n_block, actual_n_block_max_context, kBlockN, n_masking_steps);
        //  print(tVgVdecoded);
        //}
        // Advance gV
        if (masking_step > 0) {
          __copy_next_v_block();
        } else {
            // Clear the smem tiles to account for predicated off loads
            //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 2) {
            //  printf("[masking ----] visiting n_block = %d, actual_n_block_max_context = %d,  kBlockN = %d, n_masking_steps=%d, logical_actual_seqlen_k=%d, max_MN=%d\n",
            //         n_block, actual_n_block_max_context, kBlockN, n_masking_steps, logical_actual_seqlen_k, logical_actual_seqlen_k - n_block * kBlockN);
            //}
            // FIXME(jiadingg::flash decoding) only need conditional copy when n_block == (actual_n_block_max_context - 1)
            if (n_block < actual_n_block_max_context) {
                // copy from context blocks.
                flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tVgVcontext, tVsV, tKVcKV, tKVpKV,
                    binfo.actual_seqlen_k_cache_context - n_block * kBlockN
                );
            } else {
                // copy block from decoded cache.
                flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tVgVdecoded, tVsV, tKVcKV, tKVpKV,
                    actual_n_block_max_context * kBlockN + binfo.actual_seqlen_k_cache_decoded - n_block * kBlockN
                );
            }
        }
        cute::cp_async_fence();

#if 0
         if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) {
             printf("[masking step] visiting n_block = %d, actual_n_block_max_context = %d,  kBlockN = %d, n_masking_steps=%d\n", n_block, actual_n_block_max_context, kBlockN, n_masking_steps);

             sleep((int64_t)(6 * 100000ULL));
             for (int i = 0; i < cute::size<0>(sK); ++i) {
                 for (int j = 0; j < cute::size<1>(sK); ++j) {
                   printf("%3.4f, ", static_cast<float>(sK(i, j)));
                 }
                 printf("\n");
             }
             sleep((int64_t)(6 * 100000ULL));
         }
#endif

        flash::gemm(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        /* if (cute::thread0()) { print(acc_s); } */
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        if (n_block == (actual_n_block_max_context - 1) && (binfo.actual_seqlen_k_cache_context % kBlockN) != 0) {
            //require_context_masking (jiadingg:flash decoding)
            mask_context.template apply_mask</*Causal_mask=*/false, /*Is_even_kcontext*/false>(
               acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );
        } else {
            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tVsV); }
        // __syncthreads();

        // BIFURCATED ATTENTION:
        // (jiadingg) flash decoding:
        // n_block \in [actual_n_block_max_conext, actual_n_block_max_context + actual_n_block_max_decoded) are decoded block ids.
        __copy_next_k_block();

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2);
        // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));


#if 0
         if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) {
             printf("[non-masking stepV] visiting n_block = %d, actual_n_block_max_context = %d,  kBlockN = %d, n_masking_steps=%d\n", n_block, actual_n_block_max_context, kBlockN, n_masking_steps);

             sleep((int64_t)(6 * 100000ULL));
             for (int i = 0; i < cute::size<0>(sV); ++i) {
                 for (int j = 0; j < cute::size<1>(sV); ++j) {
                   printf("%3.4f, ", static_cast<float>(sV(i, j)));
                 }
                 printf("\n");
             }
             sleep((int64_t)(6 * 100000ULL));
         }
#endif

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            // After --n_block, n_block becomes smaller than n_block_min therefore
            // the following loop will not be entered.
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    //bool is_first_nonmask_iter = true;
    for (; n_block >= n_block_min; --n_block) {
        int use_context = 0;
        int require_context_masking = 0;
        if (n_block < actual_n_block_max_context)  {
          use_context = 1;
        }
        // FIXME(jiadingg): need a params.is_even_kcontext to replace (params.seqlen_k_context % kBlockN) != 0
        // is_the_context_block && not divisible by kBlockN => require_context_masking = 1
        if (n_block == (actual_n_block_max_context - 1) && (binfo.actual_seqlen_k_cache_context % kBlockN) != 0) {
          require_context_masking = 1;
        }

        //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 2) {
        //  printf("[non-masking step] visiting n_block = %d, actual_n_block_max_context = %d,  kBlockN = %d, use_context = %d, require_context_masking=%d\n", n_block, actual_n_block_max_context, kBlockN, use_context, require_context_masking);
        //}

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        if (use_context == 1) {
          // Advance gVcontext
          // FIXME(jiadingg): simplify the following logic.
          int __offset = 0;
          //if (!is_first_nonmask_iter)
          //    __offset = -int(kBlockN * params.v_row_stride);
          //is_first_nonmask_iter = false;
          __offset = n_block == (actual_n_block_max_context - 1) ? 0 : -int(kBlockN * params.v_row_stride);


           if (block_table == nullptr) {
               tVgVcontext.data() = tVgVcontext.data() + __offset;
           } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
               const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
               const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
               const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
               const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
               tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
#endif
           }
#if 1
          if (n_block == actual_n_block_max_context - 1) {
              // when copying from the last context block, need to be sure to only copy the remainder entry (not the entire block).
              flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN*/true, true, /*GAI_DEBUG*/false>(gmem_tiled_copy_QKV, tVgVcontext, tVsV, tKVcKV, tKVpKV, binfo.seqlen_k_cache_context - n_block * kBlockN);
          } else {
              assert(n_block < actual_n_block_max_context - 1);
              flash::copy</*Is_even_MN=*/true, Is_even_K, false, true, /*GAI_DEBUG*/false>(gmem_tiled_copy_QKV, tVgVcontext, tVsV, tKVcKV, tKVpKV);
          }
#else
          flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVcontext, tVsV, tKVcKV, tKVpKV);
#endif
        } else {
            // Advance gVdecoded
            if (block_table == nullptr) {
                tVgVdecoded.data() = tVgVdecoded.data() + (-int(kBlockN * params.vdecoded_row_stride));
            } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
                const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
                tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
#endif
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVdecoded, tVsV, tKVcKV, tKVpKV);
            //tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            //flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
            //auto gai_x = static_cast<float>(tVgVdecoded.data()[0]);//cutlass::half -> float
            //auto gai_y = static_cast<float>(tVgV.data()[0]);
            //printf("[VVVVVV] gai_x = %f, gai_y = %f\n", gai_x, gai_y);
            //assert(fabsf(gai_x - gai_y) < 0.001f);
        }
        cute::cp_async_fence();

#if 0
         if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) {
             printf("[non-masking step] visiting n_block = %d, actual_n_block_max_context = %d,  kBlockN = %d, n_masking_steps=%d\n", n_block, actual_n_block_max_context, kBlockN, n_masking_steps);

             sleep((int64_t)(6 * 100000ULL));
             for (int i = 0; i < cute::size<0>(sK); ++i) {
                 for (int j = 0; j < cute::size<1>(sK); ++j) {
                   printf("%3.4f, ", static_cast<float>(sK(i, j)));
                 }
                 printf("\n");
                 break;
             }
             sleep((int64_t)(6 * 100000ULL));
         }
#endif

        flash::gemm(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            if (use_context == 1) {
                // Advance gKcontext
                if (block_table == nullptr) {
                    tKgKcontext.data() = tKgKcontext.data() + (-int(kBlockN * params.k_row_stride));
                } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                    const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                    const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                    const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                    const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                    tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
#endif
                }
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKcontext, tKsK, tKVcKV, tKVpKV);
            } else {
              bool is_last_decoded_block = (n_block == actual_n_block_max_context);
              if (!is_last_decoded_block) {
                  assert(n_block > actual_n_block_max_context);
                  // Advance gKdecoded
                  if (block_table == nullptr) {
                      tKgKdecoded.data() = tKgKdecoded.data() + (-int(kBlockN * params.kdecoded_row_stride));
                  } else {
#ifndef BIFURCATED_ATTENTION_DISABLE_UNSUPPORTED_CODE
                      const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                      const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                      const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                      const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                      tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
#endif
                  }
                  flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKdecoded, tKsK, tKVcKV, tKVpKV);
              } else {
                // transition from the decoded part to the context part.
#if 1
                assert(binfo.seqlen_k_cache_context - (n_block - 1) * kBlockN > 0);
                flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgKcontext, tKsK, tKVcKV, tKVpKV, binfo.seqlen_k_cache_context - (n_block - 1) * kBlockN);
#else
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKcontext, tKsK, tKVcKV, tKVpKV);
#endif
              }
            }
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        if (require_context_masking) {
            //vector_print_tensor_acc_o(acc_s);
            mask_context.template apply_mask</*Causal_mask=*/false, /*Is_even_kcontext*/false>(
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );
            //vector_print_tensor_acc_o(acc_s);
        } else {
            mask.template apply_mask</*Causal_mask=*/false>(
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );
        }

        if (require_context_masking) {
          softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(acc_s, acc_o, params.scale_softmax_log2);
        } else {
          softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        }

        Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));


#if 0
         if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) {
             printf("[non-masking stepV] visiting n_block = %d, actual_n_block_max_context = %d,  kBlockN = %d, n_masking_steps=%d\n", n_block, actual_n_block_max_context, kBlockN, n_masking_steps);

             sleep((int64_t)(6 * 100000ULL));
             for (int i = 0; i < cute::size<0>(sV); ++i) {
                 for (int j = 0; j < cute::size<1>(sV); ++j) {
                   printf("%3.4f, ", static_cast<float>(sV(i, j)));
                 }
                 printf("\n");
                 break;
             }
             sleep((int64_t)(6 * 100000ULL));
         }
#endif

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
#ifdef BIFURCATED_ATTENTION_DEBUG
    if (cute::thread0()) { print(lse); printf("\ncompute_attn_1rowblock_splitkv_bifurcated\n"); }
#endif

    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    if constexpr (Split) { __syncthreads(); }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // (jiadding) Remove this lse write back since it's only required for the backward pass.
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

} // namespace flash
