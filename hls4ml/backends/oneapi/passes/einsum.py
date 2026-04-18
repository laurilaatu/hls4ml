from math import ceil

from hls4ml.backends.backend import get_backend
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Einsum
from hls4ml.utils.transpose_utils import transpose_config_gen

from .reshaping_templates import transpose_config_template

# Shared Dense template
# Einsum template

einsum_config_template = """
struct config{index} {{
    typedef config{index}_tpose_inp0 tpose_inp0_config;
    typedef config{index}_tpose_inp1 tpose_inp1_config;
    typedef config{index}_tpose_out tpose_out_conf;

    typedef {accum_t.name} accum_t;

    // Context Length
    static const unsigned n_ctx = {n_ctx};

    // BRAM Contraction Axis
    static const bool contract_dim = {contract_dim};

    // Square Root of KEY_DIM
    static constexpr {inp0_t} sqrt_dk = {sqrt_dk};

    // Layer Sizes
    static const unsigned n_free0 = {n_free0};
    static const unsigned n_free1 = {n_free1};
    static const unsigned n_contract = {n_contract};
    static const unsigned n_inplace = {n_inplace};

    // Resource reuse info
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};
"""

einsum_function_template = 'nnet::einsum<{input0_t}, {input1_t}, {output_t}, {config}>({input0}, {input1}, {output});'

einsum_include_list = ['nnet_utils/nnet_einsum.h', 'nnet_utils/nnet_causal_einsum.h']


class EinsumConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Einsum)
        self.template = einsum_config_template

    def format(self, node: Einsum):
        default_params = self._default_config_params(node)

        strategy = node.attributes['strategy']

        assert strategy.lower() == 'latency', 'EinsumDense layer only supports Latency strategy for now'

        # EinsumDense config
        params = default_params.copy()
        params['strategy'] = strategy
        params['n_free0'] = node.attributes['n_free0']
        params['n_free1'] = node.attributes['n_free1']
        params['n_contract'] = node.attributes['n_contract']
        params['n_inplace'] = node.attributes['n_inplace']

        # no effect if we are not using causal einsum for MHA
        params['sqrt_dk'] = 1
        params['inp0_t'] = 'unsigned'

        io_type = node.model.config.get_config_value('IOType')
        streamed = io_type == 'io_stream'

        # This means we are about to do an attention calculation - additional configs required for causal einsum
        if ('contract_dim' in node.attributes) and streamed:
            context_len = node.attributes['context_len']
            if node.attributes['contract_dim'] == 'embedding':
                params['n_free0'] = max(1, params['n_free0'] // context_len)
                params['n_free1'] = max(1, params['n_free1'] // context_len)
                params['sqrt_dk'] = params['n_contract'] ** 0.5
                ip = node.get_input_variable(node.inputs[0]).type.precision
                sign = 'true' if ip.signed else 'false'
                params['inp0_t'] = f'ac_fixed<{ip.width},{ip.integer},{sign},AC_RND,AC_SAT>'
            else:
                params['n_contract'] = max(1, params['n_contract'] // context_len)
            # params['n_inplace'] = 1

        elif (node.model.config.get_config_value('HLSConfig')['context_len'] is not None) and streamed:
            context_len = node.model.config.get_config_value('HLSConfig')['context_len']
            params['n_free0'] = max(1, params['n_free0'] // context_len)
            if 'contract_dim' in node.attributes and node.attributes['contract_dim'] == 'context':
                params['n_contract'] = max(1, params['n_contract'] // context_len)
            else:
                params['n_free1'] = max(1, params['n_free1'] // context_len)

        print(f'params are: {params["n_free0"]},{params["n_free1"]},{params["n_contract"]},{params["n_inplace"]}')

        inp0_t = node.get_input_variable(node.inputs[0]).type.precision
        inp1_t = node.get_input_variable(node.inputs[1]).type.precision
        params['product_type'] = get_backend('oneapi').product_type(inp0_t, inp1_t)

        total_mults = params['n_free0'] * params['n_free1'] * params['n_contract'] * params['n_inplace']
        params['multiplier_limit'] = ceil(total_mults / params['reuse_factor'])

        if (node.model.config.get_config_value('HLSConfig')['context_len'] is not None) and streamed:
            params['n_ctx'] = node.model.config.get_config_value('HLSConfig')['context_len']
        else:
            params['n_ctx'] = node.attributes['context_len'] if (('context_len' in node.attributes) and streamed) else 1

        params['contract_dim'] = (
            1 if ('contract_dim' in node.attributes and node.attributes['contract_dim'] == 'context' and streamed) else 0
        )

        einsum_conf = self.template.format(**params)

        # inp/out transpose config
        inp0_shape = node.attributes['inp0_shape']
        inp1_shape = node.attributes['inp1_shape']

        """if 'contract_dim' in node.attributes:
            if node.attributes['contract_dim'] == 'embedding':
                inp0_shape = (max(1,inp0_shape[0]//node.attributes['context_len']), *inp0_shape[1:])
                inp1_shape = (max(1,inp1_shape[0]//node.attributes['context_len']), *inp1_shape[1:])
            else:
                inp0_shape = (inp0_shape[0], max(1,inp0_shape[1]//node.attributes['context_len']), *inp0_shape[2:])
                inp1_shape = (max(1,inp1_shape[0]//node.attributes['context_len']), *inp1_shape[1:])
        """
        out_interpert_shape = node.attributes['out_interpert_shape']
        inp0_tpose_idxs = node.attributes['inp0_tpose_idxs']
        inp1_tpose_idxs = node.attributes['inp1_tpose_idxs']
        out_tpose_idxs = node.attributes['out_tpose_idxs']
        tpose_inp0_config_name = f'config{node.index}_tpose_inp0'
        tpose_inp1_config_name = f'config{node.index}_tpose_inp1'
        tpose_out_conf_name = f'config{node.index}_tpose_out'

        conf = transpose_config_gen(tpose_inp0_config_name, inp0_shape, inp0_tpose_idxs)
        inp0_tpose_conf = transpose_config_template.format(**conf)
        conf = transpose_config_gen(tpose_inp1_config_name, inp1_shape, inp1_tpose_idxs)
        inp1_tpose_conf = transpose_config_template.format(**conf)
        conf = transpose_config_gen(tpose_out_conf_name, out_interpert_shape, out_tpose_idxs)
        out_tpose_conf = transpose_config_template.format(**conf)
        return '\n\n'.join((inp0_tpose_conf, inp1_tpose_conf, out_tpose_conf, einsum_conf))


einsum_stream_template = 'task_sequence<nnet::causal_einsum<{input0_pipe}, {input1_pipe}, {output_pipe}, {config}>> {name};'


class EinsumFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Einsum, include_header=einsum_include_list)
        self.template = einsum_function_template

    def format(self, node: Einsum):

        io_type = node.model.config.get_config_value('IOType')
        params = {}

        if io_type == 'io_stream':
            self.template = einsum_stream_template
            params['name'] = node.name
            params['input0_pipe'] = node.get_input_variable(node.inputs[0]).pipe_name
            params['input1_pipe'] = node.get_input_variable(node.inputs[1]).pipe_name
            params['output_pipe'] = node.get_output_variable().pipe_name
            params['config'] = f'config{node.index}'

            return self.template.format(**params)
        else:
            params['config'] = f'config{node.index}'
            params['input0_t'] = node.get_input_variable(node.inputs[0]).type.name
            params['input1_t'] = node.get_input_variable(node.inputs[1]).type.name
            params['output_t'] = node.get_output_variable().type.name
            params['input0'] = node.get_input_variable(node.inputs[0]).name
            params['input1'] = node.get_input_variable(node.inputs[1]).name
            params['output'] = node.get_output_variable().name
            return self.template.format(**params)


class EinsumStreamTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Einsum)
        self.template = 'task_sequence<nnet::causal_einsum<{input0_pipe}, {input1_pipe}, {output_pipe}, {config}>> {name};'

    def format(self, node):
        params = self._default_function_params(node)
        params['input0_pipe'] = node.get_input_variable(node.inputs[0]).pipe_name
        params['input1_pipe'] = node.get_input_variable(node.inputs[1]).pipe_name
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        return self.template.format(**params)


class EinsumStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(Einsum)
        self.template = '{name}.async();'

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)
