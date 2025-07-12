/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
          // Bifurcated attention:
        , seqlen_k_cache_context((!Varlen || params.cu_seqlens_k_context == nullptr ? params.seqlen_k_context : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k_context[bidb + 1] - sum_s_k : params.cu_seqlens_k_context[bidb])) - leftpad_k)
        , actual_seqlen_k_cache_context(seqlen_k_cache_context)
        , seqlen_k_cache_decoded((!Varlen || params.cu_seqlens_k_decoded == nullptr ? params.seqlen_k_decoded : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k_decoded[bidb + 1] - sum_s_k : params.cu_seqlens_k_decoded[bidb])) - leftpad_k)
        , actual_seqlen_k_cache_decoded(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache_decoded + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int leftpad_k;
    const int seqlen_k_cache;
    const int actual_seqlen_k;

    // Bifurcated attention
    const int seqlen_k_cache_context;
    // FIXME(jiadingg):
    // No need to introduce actual_seqlen_k_cache_context since there'll be no new
    // tokens appended to the context kv cache; so actual_seqlen_k_cache_context
    // and seqlen_k_cache_context always equal.
    const int actual_seqlen_k_cache_context;
    const int seqlen_k_cache_decoded;
    const int actual_seqlen_k_cache_decoded;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
