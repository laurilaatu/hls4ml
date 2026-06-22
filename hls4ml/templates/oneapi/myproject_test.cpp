#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <immintrin.h>

#include "firmware/myproject.h"
#include "firmware/parameters.h"

#include <sycl/ext/altera/fpga_extensions.hpp>

// hls-fpga-machine-learning use host_reads


// For data collection
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

// This lib is irrelevant to Altera HLS 
#if (__INTEL_CLANG_COMPILER < 20250000)
//#include <sycl/ext/intel/prototype/interfaces.hpp>
#endif

#include "exception_handler.hpp"
// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

int main(int argc, char **argv) {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::altera::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::altera::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::altera::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    // make sure the device supports USM host allocations
    if (!device.has(sycl::aspect::usm_host_allocations)) {
        std::cerr << "This design must either target a board that supports USM "
                     "Host/Shared allocations, or IP Component Authoring. "
                  << std::endl;
        std::terminate();
    }

    std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");
    
    // Create the tb_data folder
    if (mkdir("tb_data", 0755) != 0) {
        if (errno != EEXIST) {
                perror("mkdir");
                return 1;
        }
        else{
                std::cout << "Created the tb_data folder" << std::endl;
        }
    }

    // Create the log file
    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    if (!fout.is_open()) {
        perror("results.log");
        return 1;
    }

    // Set the iterations (number of repeated tests to be performed)
    const unsigned int num_iterations = 10;

    // Pre-define tokens to be filled and generated, REMOVE FOR THE LARGE MODEL
    // This is kept currently since pipes are blocking and models are not guaranteed to predict
    // The EOS token.
    #define PREFILL_TOKENS 32
    #define TOTAL_TOKENS 48
    #define GENERATE_TOKENS (TOTAL_TOKENS - PREFILL_TOKENS)


#if HOST_READS

    // Create the required host memory

    using token_t = typename token_inputs_t::value_type;
    token_t *token_inputs_vals = sycl::malloc_host<token_t>(PREFILL_TOKENS, q);
    if(token_inputs_vals == nullptr){
        std::cerr << "ERROR: host allocation failed for input\n";
        fout.close();
        return 1;
    }

    using pos_t = typename pos_inputs_t::value_type;
    pos_t *pos_inputs_vals = sycl::malloc_host<pos_t>(PREFILL_TOKENS, q);
    if(pos_inputs_vals == nullptr){
        std::cerr << "ERROR: host allocation failed for input\n";
        fout.close();
        return 1;
    }

    using opt_t = typename token_inputs_t::value_type;
    float *outputs = sycl::malloc_host<float>(TOTAL_TOKENS, q);
    if(outputs == nullptr){
        std::cerr << "ERROR: host allocation failed for output\n";
        fout.close();
        return 1;
    }

    volatile uint32_t *ttft_flag = sycl::malloc_shared<uint32_t>(1, q);
    if(ttft_flag == nullptr){
        std::cerr << "ERROR: host allocation failed for output\n";
        fout.close();
        return 1;
    }

    std::cout << "INFO: Using a pre-determined input sequence, performing " << num_iterations
                << " independent tests." << std::endl;

    std::chrono::high_resolution_clock::time_point first_token_time;
    double average_ts = 0;
    double average_ttft = 0;
    double first_ts = 0;
    double first_ttft = 0;

    // Input sequence
    token_t input_seq[PREFILL_TOKENS] = {44,48,8,35,6,33,9,8,34,195,71,1,3,1,72,7,207,846,5,43,18,4,425,585,20,19,63,3,30,24,6,10}; 

    // Fill in the host buffers
    for (int j = 0 ; j < PREFILL_TOKENS ; j++) {
        token_inputs_vals[j] = input_seq[j]; 
        pos_inputs_vals[j] = j; 
    }

    std::cout << "Filled arrays are:" << std::endl;
    std::cout << "Token Input Sequence: ";
    for(int j = 0; j < PREFILL_TOKENS - 1; j++) std::cout << token_inputs_vals[j] << ",";
    std::cout << token_inputs_vals[PREFILL_TOKENS - 1] << std::endl;
    std::cout << "Positional Embeddings: ";
    for(int j = 0; j < PREFILL_TOKENS - 1; j++) std::cout << pos_inputs_vals[j] << ",";
    std::cout << pos_inputs_vals[PREFILL_TOKENS - 1] << std::endl;


    for (int i = 0; i < num_iterations; i++) {

        // Reset the timing flag
        *ttft_flag = 0;
        
        // Start timer
        auto start = std::chrono::high_resolution_clock::now();

        //Create a new CPU thread to independently pool the TTFT
        std::thread ttft_thread([&](){
            while(*ttft_flag == 0){
                _mm_pause();
            }
            first_token_time = std::chrono::high_resolution_clock::now();
        });
                
        q.single_task(nnet::DMA_convert_data<token_t, pos_t, TokenInputsPipe, PosInputsPipe>{token_inputs_vals, pos_inputs_vals, 1});
        q.single_task(Myproject{});
        q.single_task(nnet::DMA_convert_data_back<Layer80OutPipe, float, uint32_t>{outputs, ttft_flag, 48}).wait();
        
        auto end = std::chrono::high_resolution_clock::now();
        ttft_thread.join();

        //Analyse and record data on the host side
        double ttft = std::chrono::duration<double, std::milli>(first_token_time - start).count();
        double total_time = std::chrono::duration<double>(end - start).count();
        double num_tokens = TOTAL_TOKENS;
        double ts = num_tokens/total_time; 
        average_ts = i > 0 ? ((average_ts * i + ts)/(i+1)) : ts;
        average_ttft = i > 0? ((average_ttft * i + ttft)/(i+1)) : ttft;
        std::cout << "Current TTFT (ms): " << ttft << std::endl;
        std::cout << "Current Tokens/s: " << ts << std::endl;

        if(i == 0){
            first_ts = ts;
            first_ttft = ttft;
        }

        int pipeOutSize = std::tuple_size<typename nnet::ExtractPipeType<Layer80OutPipe>::value_type>{};
        for (int i = 0; i < TOTAL_TOKENS; i++) {
            for (int j = 0; j < pipeOutSize; j++) {
                fout << outputs[i * pipeOutSize + j] << " ";
            }
            fout << std::endl;
        }   
    }

    std::cout << num_iterations << " iterations were performed." << std::endl;
    std::cout << "Average TTFT (ms) was: " << average_ttft << std::endl;
    std::cout << "Average Tokens/s was: " << average_ts << std::endl;

    // Ignore the first ttft and t/s
    average_ttft = (average_ttft * num_iterations - first_ttft) / (num_iterations - 1);
    average_ts = (average_ts * num_iterations - first_ts) / (num_iterations - 1);
    std::cout << "Average TTFT (ms) after ignoring the first TTFT: " << average_ttft << std::endl;
    std::cout << "Average Tokens/s after ignoring the first Tokens/s: " << average_ts << std::endl;

    sycl::free(token_inputs_vals, q);
    sycl::free(pos_inputs_vals, q);
    sycl::free(outputs, q);
    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;

#else
    std::string iline;
    std::string pline;

    if (fin.is_open() && fpr.is_open()) {
        std::vector<std::vector<float>> predictions;
        unsigned int iteration = 0;
        for (; std::getline(fin, iline) && std::getline(fpr, pline); iteration++) {
            if (iteration % CHECKPOINT == 0) {
                std::cout << "Processing input " << iteration << std::endl;
            }

            std::vector<float> in;
            std::vector<float> pr;
            float current;

            std::stringstream ssin(iline);
            while (ssin >> current) {
                in.push_back(current);
            }

            std::stringstream sspred(pline);
            while (sspred >> current) {
                pr.push_back(current);
            }

            // hls-fpga-machine-learning insert data
            float token_inputs_vals[48]; 
            for (int j = 0 ; j < 48 ; j++) {
                token_inputs_vals[j] = in[j]; 
            }
            nnet::convert_data<float, TokenInputsPipe, 48>(q, token_inputs_vals);
            float pos_inputs_vals[48]; 
            for (int j = 0 ; j < 48 ; j++) {
                pos_inputs_vals[j] = in[j]; 
            }
            nnet::convert_data<float, PosInputsPipe, 48>(q, pos_inputs_vals);

            q.single_task(Myproject{});

            // hls-fpga-machine-learning convert output
            float outputs[48*1024];
            nnet::convert_data_back<Layer80OutPipe, float, 48*1024>(q, outputs);

            std::copy(pr.cbegin(), pr.cend(), predictions.back().begin());

            for (auto outval : outputs) {
                fout << outval << " ";
            }
            fout << std::endl;
            if (iteration % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for (auto predval : pr) {
                    std::cout << predval << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                for (auto outval : outputs) {
                    std::cout << outval << " ";
                }
                std::cout << std::endl;
            }
        }
        fin.close();
        fpr.close();
    } else {
        const unsigned int num_iterations = 10;
        std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                  << " invocations." << std::endl;

        // hls-fpga-machine-learning insert top-level-function
        for (int i = 0; i < num_iterations; i++) {
            // hls-fpga-machine-learning insert zero
            float token_inputs_vals[48]; 
            for (int j = 0 ; j < 48 ; j++) {
                token_inputs_vals[j] = 0.0; 
            }
            nnet::convert_data<float, TokenInputsPipe, 48>(q, token_inputs_vals);
            float pos_inputs_vals[48]; 
            for (int j = 0 ; j < 48 ; j++) {
                pos_inputs_vals[j] = 0.0; 
            }
            nnet::convert_data<float, PosInputsPipe, 48>(q, pos_inputs_vals);
            q.single_task(Myproject{});
            // hls-fpga-machine-learning convert output
            float outputs[48*1024];
            nnet::convert_data_back<Layer80OutPipe, float, 48*1024>(q, outputs);
            for (auto outval : outputs) {
                std::cout << outval << " ";
            }
            std::cout << std::endl;

            for (auto outval : outputs) {
                fout << outval << " ";
            }
            fout << std::endl;
        }
    }
    q.wait();

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
#endif
}
