#ifndef NNET_DATA_MOVEMENT_H
#define NNET_DATA_MOVEMENT_H

#include <sycl/ext/altera/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// This file defines the methods to transfer the data to the kernel. In the HLS flow,
// these are really part of the testbench. However, in the accelerator (BSP) flow, they are
// actual kernels that are deployed in hardware.

namespace nnet {

struct test_DMA {
    unsigned sop_num;
    unsigned eop_num;
    unsigned writect;
};
//////////////////////////////////////////////////////////////////////////////
// These are the simple, testbench and bridge versions for the HLS flow
//////////////////////////////////////////////////////////////////////////////
/*
template <class srcType, class dest_pipe, size_t SIZE> void convert_data(sycl::queue &q, srcType *src) {
    constexpr auto dstTypeSize = std::tuple_size<typename ExtractPipeType<dest_pipe>::value_type>{};
    for (size_t i = 0; i < SIZE / dstTypeSize; i++) {
        typename ExtractPipeType<dest_pipe>::value_type ctype;
        for (size_t j = 0; j < dstTypeSize; j++) {
            ctype[j] = src[i * dstTypeSize + j];
        }
        dest_pipe::write(q, ctype);
    }
}

template <class src_pipe, class dstType, size_t SIZE> void convert_data_back(sycl::queue &q, dstType *dst) {
    constexpr auto srcTypeSize = std::tuple_size<typename ExtractPipeType<src_pipe>::value_type>{};
    for (size_t i = 0; i < SIZE / srcTypeSize; i++) {
        auto ctype = src_pipe::read(q);
        for (size_t j = 0; j < srcTypeSize; j++) {
            dst[i * srcTypeSize + j] = ctype[j].to_double();
        }
    }
}
*/
//////////////////////////////////////////////////////////////////////////////
// The ones below can be used both in testbenches and in the accelerator flow
//////////////////////////////////////////////////////////////////////////////
#if !defined(IS_BSP)
// Definition for buffer locations for Avalon MM host.
inline constexpr unsigned kInputBufferLocation = 0;
inline constexpr unsigned kOutputBufferLocation = 1;
#endif

// Implementation of a direct memory access kernel. Move data from source, convert,
// and send to the sink. Adaptive to SYCL HLS and hardware acceleration flow.
template <class tkn_src_T, class pos_src_T, class tkn_dest_pipe, class pos_dest_pipe> struct DMA_convert_data {
#if !defined(IS_BSP)
    // When targeting a device family, we instantiate an Avalon Memory Mapped Host for
    // data transaction between host and the DMA kernel during emulation and simulation.
    sycl::ext::oneapi::experimental::annotated_arg<
        tkn_src_T *,
        decltype(sycl::ext::oneapi::experimental::properties{
            sycl::ext::altera::experimental::latency<0>, sycl::ext::altera::experimental::dwidth<16>,
            sycl::ext::altera::experimental::buffer_location<kInputBufferLocation>,
            sycl::ext::altera::experimental::read_write_mode_read, sycl::ext::altera::experimental::wait_request_requested})>
#else
    // When targeting oneAPI BSP, we can use USM pointer to access host memory.
    tkn_src_T *const
#endif
        tkn_src;

#if !defined(IS_BSP)
    // When targeting a device family, we instantiate an Avalon Memory Mapped Host for
    // data transaction between host and the DMA kernel during emulation and simulation.
    sycl::ext::oneapi::experimental::annotated_arg<
        pos_src_T *,
        decltype(sycl::ext::oneapi::experimental::properties{
            sycl::ext::altera::experimental::latency<0>, sycl::ext::altera::experimental::dwidth<16>,
            sycl::ext::altera::experimental::buffer_location<kInputBufferLocation>,
            sycl::ext::altera::experimental::read_write_mode_read, sycl::ext::altera::experimental::wait_request_requested})>
#else
    // When targeting oneAPI BSP, we can use USM pointer to access host memory.
    pos_src_T *const
#endif
        pos_src;

    size_t num_iteration;

    [[intel::kernel_args_restrict]] void operator()() const {

#if defined(IS_BSP)
        // Access data using host pointer.
        sycl::ext::altera::host_ptr<tkn_src_T> tkn_src_ptr(tkn_src);
        sycl::ext::altera::host_ptr<pos_src_T> pos_src_ptr(pos_src);
#else
        // Host allocation is not supported when targeting an FPGA family or part number.
        tkn_src_T *tkn_src_ptr(tkn_src);
        pos_src_T *pos_src_ptr(pos_src);
#endif
        // First, extract the PipeDataT from the pipe
        using TokenPipeDataType = typename nnet::ExtractPipeType<tkn_dest_pipe>::value_type;
        using PosPipeDataType = typename nnet::ExtractPipeType<pos_dest_pipe>::value_type;
        // By definition, both must have the same size
	constexpr auto dstTypeSize = std::tuple_size<TokenPipeDataType>{};

        [[intel::fpga_register]] TokenPipeDataType tkn_packet;
        [[intel::fpga_register]] PosPipeDataType pos_packet;

        // Keep sending data to the input layer and keep the kernels running.
        for (size_t i = 0; i < num_iteration; i++) {
            
	    #pragma unroll
            for (size_t j = 0; j < dstTypeSize; j++) {
                tkn_packet[j] = tkn_src_ptr[i * dstTypeSize + j];
            }
            tkn_dest_pipe::write(tkn_packet);
	    
	    #pragma unroll 
	    for (size_t j = 0; j < dstTypeSize; j++) {
                pos_packet[j] = pos_src_ptr[i * dstTypeSize + j];
            }
            pos_dest_pipe::write(pos_packet);
        }
    }
};

// Symmetrical to the DMA_convert_data above, this DMA drains the output pipe and
// writes result to memory.
template <class src_pipe, class dst_T, class ttft_flag_T> struct DMA_convert_data_back {
#if !defined(IS_BSP)
    // Without BSP, instantiate an Avalon Memory Mapped Host to write to host.
    sycl::ext::oneapi::experimental::annotated_arg<
        dst_T *,
        decltype(sycl::ext::oneapi::experimental::properties{
            sycl::ext::altera::experimental::latency<0>, sycl::ext::altera::experimental::dwidth<16>,
            sycl::ext::altera::experimental::buffer_location<kOutputBufferLocation>,
            sycl::ext::altera::experimental::read_write_mode_write, sycl::ext::altera::experimental::wait_request_requested})>
#else
    // USM pointer, otherwise.
    dst_T *const 
#endif
    dst;

    volatile ttft_flag_T *const ttft_flag;

    size_t num_iteration;

    [[intel::kernel_args_restrict]] void operator()() const {
#if defined(IS_BSP)
        sycl::ext::altera::host_ptr<dst_T> dst_ptr(dst);
#else
        dst_T *dst_ptr(dst);
#endif
        // First, extract the PipeDataT from the pipe
        using PipeDataType = typename nnet::ExtractPipeType<src_pipe>::value_type;
        // Then, extract the DataT from StreamingBeat
        using SrcDataType = typename PipeDataType::value_type;
        constexpr auto srcTypeSize = std::tuple_size<PipeDataType>{};

        [[intel::fpga_register]] PipeDataType packet;

        // Drain the output pipe and write result to memory.
        for (size_t i = 0; i < num_iteration; i++) {
            packet = src_pipe::read();

            #pragma unroll 4
            for (size_t j = 0; j < srcTypeSize; j++) {
                dst_ptr[i * srcTypeSize + j] = static_cast<dst_T>(packet[j].to_double());
	    }
	    
	    if(i == 0) *ttft_flag = 1;

        }
    }
};

//////////////////////////////////////////////////////////////////////////////
// These are versions to convert data for the accelerator bridge (using BSP)
//////////////////////////////////////////////////////////////////////////////
/*
template <class srcType, class dest_pipe, size_t SIZE> void DMA_bridge_convert_data(sycl::queue &q, srcType *src) {
    // First, extract the PipeDataT from the pipe
    using PipeDataType = typename nnet::ExtractPipeType<dest_pipe>::value_type;
    // Then, extract the DataT from StreamingBeat
    using DstDataType = typename nnet::ExtractDataType<PipeDataType>::value_type;
    constexpr auto dstTypeSize = std::tuple_size<DstDataType>{};

    constexpr size_t num_iterations = SIZE / dstTypeSize;

    // Allocate host memory
    srcType *vals = sycl::malloc_host<srcType>(SIZE, q);
    if (vals == nullptr) {
        std::cerr << "ERROR: host allocation failed for input\n";
        return;
    }
    // copy to host memory
    for (size_t i = 0; i < SIZE; i++) {
        vals[i] = src[i];
    }
    q.single_task(DMA_convert_data<srcType, dest_pipe>{vals, num_iterations});
}

template <class src_pipe, class dstType, size_t SIZE> void DMA_bridge_convert_data_back(sycl::queue &q, dstType *dst) {
    // First, extract the PipeDataT from the pipe
    using PipeDataType = typename nnet::ExtractPipeType<src_pipe>::value_type;
    // Then, extract the DataT from StreamingBeat
    using SrcDataType = typename nnet::ExtractDataType<PipeDataType>::value_type;
    constexpr auto srcTypeSize = std::tuple_size<SrcDataType>{};

    constexpr size_t num_iterations = SIZE / srcTypeSize;

    // Allocate host memory
    dstType *outputs = sycl::malloc_host<dstType>(SIZE, q);
    if (outputs == nullptr) {
        std::cerr << "ERROR: host allocation failed for output\n";
        return;
    }

    q.single_task(DMA_convert_data_back<src_pipe, dstType>{outputs, num_iterations}).wait();

    // copy the data back
    for (size_t j = 0; j < SIZE; j++) {
        dst[j] = outputs[j];
    }
}
*/

} // namespace nnet

#endif
