#ifndef NNET_EINSUM_STREAMED_H_
#define NNET_EINSUM_STREAMED_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"
#include <limits>
#include <sycl/ext/oneapi/device_global/device_global.hpp>
#include <type_traits>

namespace nnet {

struct config_causal_einsum {
    typedef void tpose_inp0_config;
    typedef void tpose_inp1_config;
    typedef void tpose_out_conf;

    // Layer Sizes
    static const unsigned n_free0;
    static const unsigned n_free1;
    static const unsigned n_contract;
    static const unsigned n_inplace;

    // Resource reuse info
    static const unsigned io_type;
    static const unsigned reuse_factor;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

#define PADDING_TYPES 1
#if PADDING_TYPES
// Padding Helper - Not used anymore since attention layer does not need it
template <typename data_T> constexpr data_T minval() {
    if constexpr (std::numeric_limits<data_T>::is_specialized) {
        return std::numeric_limits<data_T>::lowest();
    } else {
        if (!data_T::sign)
            return data_T(0);
        else
            return data_T(-(1LL << (data_T::i_width - 1)));
    }
}

// ac_fixed<res_T::width, res_T::i_width, res_T::sign, res_T::q_mode, AC_SAT> min_fx =
//  value<AC_VAL_MIN>(ac_fixed<res_T::width, res_T::i_width, res_T::sign, res_T::q_mode, AC_SAT>());
// auto int_part = my_var.slc<res_T::i_width>(res_T::width - res_T::i_width);
// auto frac_part = my_var.slc<res_T::width - res_T::i_width>(0);
#endif

using namespace sycl::ext::oneapi::experimental;

template <typename data_T, typename CONFIG_T> struct CausalState {
    static constexpr unsigned BRAM_SIZE = CONFIG_T::n_ctx * CONFIG_T::n_inplace * CONFIG_T::n_contract * CONFIG_T::n_free1;

    struct State {
        //[[intel::numbanks(CONFIG_T::n_ctx), intel::bankwidth(sizeof(data_T))]] 
	    [[intel::singlepump]] data_T causal_buff[BRAM_SIZE];
        unsigned write_ptrs[CONFIG_T::n_inplace];
        unsigned ctx_cts[CONFIG_T::n_inplace];
    };
};

template <typename data_T, typename CONFIG_T>
device_global<typename CausalState<data_T, CONFIG_T>::State, decltype(properties(device_image_scope, host_access_none))>
    causal_state;

// reads a contraction length unit from stream used for datas of dimension > 2
// only works with per-I streamed data though, careful about what you stream
template <class data_T, class data_pipe, typename CONFIG_T>
void read_causal_pipe(unsigned i, unsigned *write_ptrs, unsigned *ctx_cts, data_T *causal_buffer) {

    using data_buf_T = typename ExtractPipeType<data_pipe>::value_type;
    constexpr std::size_t CAUSAL_PIPE_SIZE = std::tuple_size<data_buf_T>::value;

    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned CTX = CONFIG_T::n_ctx;

    if (!CONFIG_T::contract_dim) {
        if (CAUSAL_PIPE_SIZE == C && L1 == 1) { // case where we have a dot product so stream is not 1xL1 but 1xC instead
            //#pragma unroll
            for (unsigned l1 = 0; l1 < L1; l1++) {
                // assumes stream is vector by vector (1xC each time)
                data_buf_T buff = data_pipe::read();

                for (unsigned c = 0; c < C; c++) {
                    causal_buffer[L1 * C * I * write_ptrs[i] + C * L1 * i + C * l1 + c] = buff[c];
                }
            }
        } else {
            for (unsigned c = 0; c < C; c++) {
                // assumes stream is vector by vector (1xL1 each time)
                data_buf_T buff = data_pipe::read();

                //#pragma unroll
                for (unsigned l1 = 0; l1 < L1; l1++) {
                    causal_buffer[L1 * C * I * write_ptrs[i] + C * L1 * i + C * l1 + c] = buff[l1];
                }
            }
        }
    } else {
        data_buf_T buff = data_pipe::read();
        //#pragma unroll
        for (unsigned l1 = 0; l1 < L1; l1++) {
            causal_buffer[L1 * I * write_ptrs[i] + L1 * i + l1] = buff[l1];
        }
    }
    ctx_cts[i] = (ctx_cts[i] + 1 < CTX)? (ctx_cts[i] + 1) : CTX;
    write_ptrs[i] = (write_ptrs[i] + 1 >= CTX)? (write_ptrs[i] + 1 - CTX) : (write_ptrs[i] + 1);
    //write_ptrs[i] = (write_ptrs[i] + 1) % CTX;
}

//read specific to contraction along the context
template <class data_T, class data_buf_T, class data_pipe, typename CONFIG_T>
void read_causal_pipe_ctx(data_buf_T &data_buff, unsigned i, unsigned *write_ptrs, unsigned *ctx_cts, data_T *causal_buffer) {

    constexpr std::size_t CAUSAL_PIPE_SIZE = std::tuple_size<data_buf_T>::value;

    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned CTX = CONFIG_T::n_ctx;
    
    unsigned wptr = (write_ptrs[i]+CTX-1) % CTX;
    unsigned offset = L1 * I * wptr + L1 * i;    
    #pragma unroll
    for (unsigned l1 = 0; l1 < L1; l1++) {
            causal_buffer[offset + l1] = data_buff[l1];
    }

    //ctx_cts[i] = std::min(ctx_cts[i] + 1, CTX);
    //write_ptrs[i] = (write_ptrs[i] + 1) % CTX;
}

// reads a contraction length unit from stream used for datas of dimension > 2
// only works with per-I streamed data though, careful about what you stream
template <class data_T, class data_pipe, typename CONFIG_T>
void read_stateless_pipe(data_T data_vect_buffer[(CONFIG_T::contract_dim ? CONFIG_T::n_free0 : CONFIG_T::n_contract)]) {

    using data_buf_T = typename ExtractPipeType<data_pipe>::value_type;

    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned L0 = CONFIG_T::n_free0;
    // constexpr unsigned I = CONFIG_T::n_inplace;

    // assumes stream is vector by vector (1xC each time)
    data_buf_T buff = data_pipe::read();

    if (!CONFIG_T::contract_dim) {
        #pragma unroll
        for (unsigned c = 0; c < C; c++) {
            data_vect_buffer[c] = buff[c];
        }
    } else {
        #pragma unroll
        for (unsigned l0 = 0; l0 < L0; l0++) {
            data_vect_buffer[l0] = buff[l0];
        }
    }
}

constexpr unsigned ceil_log2(unsigned x) {
    if (x == 0)
        return 0;
    unsigned res = 0;
    x -= 1;
    while (x > 0) {
        x >>= 1;
        res++;
    }
    return res;
}

// THIS ASSUMES DATA ARRIVES IN {STATELESS,CAUSAL} FASHION
template <class data0_pipe, class data1_pipe, class res_pipe, typename CONFIG_T> void causal_einsum() {

    using data0_buf_T = typename ExtractPipeType<data0_pipe>::value_type;
    using data0_T = typename data0_buf_T::value_type;

    using data1_buf_T = typename ExtractPipeType<data1_pipe>::value_type;
    using data1_T = typename data1_buf_T::value_type;

    using res_buf_T = typename ExtractPipeType<res_pipe>::value_type;
    using res_T = typename res_buf_T::value_type;

    // using accum_T = ac_fixed<2*res_T::width, 2*res_T::i_width, res_T::sign>;
    // typename CONFIG_T::accum_t;

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned CTX = CONFIG_T::n_ctx;

    constexpr unsigned accum_T_accum_bits = (CONFIG_T::contract_dim) ? ceil_log2(CTX) : ceil_log2(C);
    constexpr unsigned accum_T_width = data0_T::width + data1_T::width + accum_T_accum_bits;
    constexpr unsigned accum_T_integer = data0_T::i_width + data1_T::i_width + accum_T_accum_bits;
    using accum_T = ac_fixed<accum_T_width, accum_T_integer, res_T::sign>;

    // initialise the buffers to read into
    [[intel::fpga_register]] data0_T data_vect_buffer[CONFIG_T::contract_dim ? L0 : C];
    [[intel::fpga_register]] res_buf_T res_buffer;

    //######## REQUIRED AS GLOBAL PER LAYER NOT PER FUNC CALL ########
    auto &state = causal_state<data1_T, CONFIG_T>.get();
    auto &causal_buff = state.causal_buff;
    auto &write_ptrs = state.write_ptrs;
    auto &ctx_cts = state.ctx_cts;
    //################################################################

    for (unsigned loop = 0; loop < CTX; loop++) {
        // COMBINE THIS WITH TILED APPROACH FOR A SPEEDUP
        #pragma unroll 4
        for (unsigned i = 0; i < I; i++) {

            if constexpr (!CONFIG_T::contract_dim) { // CONTRACT ALONG THE C DIMENSION

                for (unsigned l0 = 0; l0 < L0; l0++) {

                    read_stateless_pipe<data0_T, data0_pipe, CONFIG_T>(data_vect_buffer);

                    if (l0 == 0)
                        read_causal_pipe<data1_T, data1_pipe, CONFIG_T>(i, write_ptrs, ctx_cts, causal_buff);

                    unsigned offset_ctx = (ctx_cts[i] == CTX) ? write_ptrs[i] : 0;

                    for (unsigned ctx = 0; ctx < CTX; ctx++) {

                        const unsigned ctx_offset = L1 * C * I * ((offset_ctx + ctx) % CTX);
                        const unsigned ctx_buff_offset = L1 * ctx;

                        if (ctx < ctx_cts[i]) {
                            #pragma unroll 4
                            for (unsigned l1 = 0; l1 < L1; l1++) {
                                accum_T tmp = 0;

                                //#pragma unroll 4
                                for (unsigned c = 0; c < C; c++) {
                                    tmp += data_vect_buffer[c] * causal_buff[ctx_offset + C * L1 * i + C * l1 + c];
                                }

                                tmp /= CONFIG_T::sqrt_dk;
                                res_buffer[ctx_buff_offset + l1] = static_cast<res_T>(tmp);
                            }
                        } else {
                            #pragma unroll
                            for (unsigned l1 = 0; l1 < L1; l1++) {
                                res_buffer[ctx_buff_offset + l1] =
                                    minval<res_T>(); // res_T(0);//no need to pad with -INF since V mult has 0's in place
                            }
                        }
                    }
                    res_pipe::write(res_buffer);
                }

            } else { // CONTRACT ALONG THE CONTEXT - In this mode L0 == CTX and C is irrelevant

                read_stateless_pipe<data0_T, data0_pipe, CONFIG_T>(data_vect_buffer);
		
		[[intel::fpga_register]] data1_buf_T causal_data = data1_pipe::read();
		[[intel::fpga_register]] data1_buf_T causal_data_operate = causal_data;
		
		ctx_cts[i] = (ctx_cts[i] + 1 < CTX)? (ctx_cts[i] + 1) : CTX;
        //write_ptrs[i] = (write_ptrs[i] + 1) % CTX;
        write_ptrs[i] = (write_ptrs[i] + 1 >= CTX)? (write_ptrs[i] + 1 - CTX) : (write_ptrs[i] + 1);
		
                unsigned offset_ctx = (ctx_cts[i] == CTX) ? write_ptrs[i] : 0;

                #pragma unroll 4
                for (unsigned l1 = 0; l1 < L1; l1++) {
                    accum_T tmp = 0;
		            tmp = (CTX-1 < ctx_cts[i]) ? (data_vect_buffer[CTX-1] * causal_data_operate[l1]) : (data_vect_buffer[ctx_cts[i]-1] * causal_data_operate[l1]);
                    for (unsigned ctx = 0; ctx < CTX-1; ctx++) {
                        if (ctx < ctx_cts[i]-1){

                            unsigned idx = offset_ctx + ctx;
                            //unsigned offset_to_buffer = (offset_ctx + ctx) % CTX;
                            unsigned offset_to_buffer = (idx >= CTX) ? idx - CTX : idx;
                            tmp += data_vect_buffer[ctx] * causal_buff[L1 * I * offset_to_buffer + L1 * i + l1];
                        }
                    }
                    res_buffer[l1] = static_cast<res_T>(tmp);
                }

                res_pipe::write(res_buffer);
                read_causal_pipe_ctx<data1_T, data1_buf_T, data1_pipe, CONFIG_T>(causal_data, i, write_ptrs, ctx_cts, causal_buff);
            }
        }
    }

    // CLEAN BRAMS AFTER ITERATIONS END
    for (unsigned i = 0; i < I; i++) {
        ctx_cts[i] = 0;
        write_ptrs[i] = 0;
    }
}

} // namespace nnet

#endif
