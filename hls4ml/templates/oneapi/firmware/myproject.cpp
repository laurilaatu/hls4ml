#include "myproject.h"
#include "parameters.h"
//#include "nnet_utils/nnet_dma_helpers.h"
#include <sycl/ext/altera/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

// hls-fpga-machine-learning insert invocation props

using sycl::ext::altera::experimental::task_sequence;

void MyProject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    // hls-fpga-machine-learning return
}
