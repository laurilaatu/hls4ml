#ifndef NNET_EMBED_STREAM_H_
#define NNET_EMBED_STREAM_H_

namespace nnet {

template <class data_pipe, class res_pipe, typename CONFIG_T>
void embedding_stream() {

    using res_T = typename ExtractPipeType<res_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{};

    constexpr unsigned loopnum = ((CONFIG_T::n_in/datasize) > 0)?  (CONFIG_T::n_in/datasize) : 1;

    if constexpr ((datasize == 1) && (loopnum > 0)){
   	InputSequence2:
    	[[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int j = 0; j < loopnum; j++) {

		auto in_data = data_pipe::read();
        	res_T res_pack;

   	 DenseEmbedding2:
        	#pragma unroll 4 // TODO: Introduce a config parameter
        	for (int i = 0; i < CONFIG_T::n_out; i++) {
           	 	res_pack[i] = CONFIG_T::embeddings[(in_data[0] * CONFIG_T::n_out + i).to_uint()];
       	 	}	

        	res_pipe::write(res_pack);
	}
  	
    }
    else{

    	auto in_data = data_pipe::read();

	InputSequence:
   	 [[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int j = 0; j < datasize; j++) {

	        res_T res_pack;
	
    	DenseEmbedding:
        	#pragma unroll 4 // TODO: Introduce a config parameter
        	for (int i = 0; i < CONFIG_T::n_out; i++) {
           	 res_pack[i] = CONFIG_T::embeddings[(in_data[j] * CONFIG_T::n_out + i).to_uint()];
        	}

        	res_pipe::write(res_pack);
    	}
    }
}

} // namespace nnet

#endif
