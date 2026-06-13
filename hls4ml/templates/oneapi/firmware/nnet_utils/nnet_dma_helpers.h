#ifndef AUTOREGRESSIVE_HELPER_H_
#define AUTOREGRESSIVE_HELPER_H_

/*
Global switch to keep track of where we read from, 
use seperate switches to avoid pipeline stall issues
*/

namespace nnet {

template<class data_pipe, class loopback_pipe, class output_pipe, typename CONFIG_T>
void input_switch() {

    constexpr unsigned N_PREFILL = CONFIG_T::n_prefill;
    constexpr unsigned N_GENERATE = CONFIG_T::n_generate;
    constexpr unsigned N_TOTAL = N_PREFILL + N_GENERATE;

	using data_T = typename ExtractPipeType<data_pipe>::value_type;
	[[intel::fpga_register]] data_T in_data;


	for(unsigned i = 0; i < N_TOTAL; i++){
		if(i >= N_PREFILL) in_data = loopback_pipe::read();
		else in_data = data_pipe::read();
		output_pipe::write(in_data);
    }
}

template<class data_in_pipe, class loopback_emb_pipe, class loopback_pos_pipe,  class output_pipe, typename CONFIG_T>
void output_switch(){

    constexpr unsigned N_PREFILL = CONFIG_T::n_prefill;
    constexpr unsigned N_GENERATE = CONFIG_T::n_generate;
    constexpr unsigned N_TOTAL = N_PREFILL + N_GENERATE;

    using data_in_arr_T = typename ExtractPipeType<data_in_pipe>::value_type;
    using data_in_T = typename data_in_arr_T::value_type;

	constexpr unsigned n_data_in = std::tuple_size<data_in_arr_T>{};
	constexpr unsigned loops = ceil_log2(n_data_in);
	using idx_T = ac_fixed<loops,loops,false>;
	using idx_arr_T = typename nnet::array<ac_fixed<loops,loops,false>,n_data_in>;
	
    //this is fine since we feed a single token
    using emb_arr_T = typename ExtractPipeType<loopback_emb_pipe>::value_type;
    using pos_arr_T = typename ExtractPipeType<loopback_pos_pipe>::value_type;
    using emb_T = typename emb_arr_T::value_type;
    using pos_T = typename pos_arr_T::value_type;
	
    [[intel::fpga_register]] emb_arr_T emb_arr;
    [[intel::fpga_register]] pos_arr_T pos_arr;

	idx_arr_T maxval_idxs;
	#pragma unroll
	for(unsigned it = 0; it < n_data_in; it++){
		maxval_idxs[it] = it;
	}

    idx_T maxval_idx = 0;

    for(unsigned i = 0; i < N_TOTAL; i++){

    	data_in_arr_T data = data_in_pipe::read();

		#pragma unroll
		for(unsigned arr_len = n_data_in; arr_len > 1; arr_len /= 2){
			unsigned curr_len = arr_len/2;
			#pragma unroll
			for(unsigned l = 0; l < curr_len; l++){
				data_in_T data_first = data[2*l];
				data_in_T data_second = data[2*l+1];
				data[l] = (data_first > data_second) ? data_first : data_second;
				maxval_idxs[l] = (data_first > data_second) ? maxval_idxs[2*l] : maxval_idxs[2*l+1];
			}
		}

		maxval_idx = maxval_idxs[0];
		for(unsigned j = 0; j < std::tuple_size<emb_arr_T>{}; j++){
			emb_arr[j] = static_cast<emb_T>(maxval_idx);
			pos_arr[j] = static_cast<pos_T>(i+1);
		}

		// Write the next token ID instead
		output_pipe::write(emb_arr);
		
		if((i >= N_PREFILL-1) && (i < N_TOTAL-1)){
			//guaranteed to be a single item; arrays are for formatting
			loopback_emb_pipe::write(emb_arr);
			loopback_pos_pipe::write(pos_arr);
		}
    }
}


} //namespace nnet
#endif
