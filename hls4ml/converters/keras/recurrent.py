from hls4ml.converters.keras_v2_to_hls import (
    KerasModelReader,
    KerasNestedFileReader,
    KerasWrappedLayerFileReader,
    KerasWrappedLayerReader,
    get_layer_handlers,
    get_weights_data,
    keras_handler,
    parse_default_keras_layer,
    parse_keras_model,
)

rnn_layers = ['SimpleRNN', 'LSTM', 'GRU']


@keras_handler(*rnn_layers)
def parse_rnn_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in rnn_layers or keras_layer['class_name'][1:] in rnn_layers

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['return_sequences'] = keras_layer['config']['return_sequences']
    layer['return_state'] = keras_layer['config']['return_state']

    if 'SimpleRNN' not in layer['class_name']:
        layer['recurrent_activation'] = keras_layer['config']['recurrent_activation']

    layer['time_major'] = keras_layer['config']['time_major'] if 'time_major' in keras_layer['config'] else False

    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('Time-major format is not supported by hls4ml')

    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    layer['n_out'] = keras_layer['config']['units']

    layer['weight_data'], layer['recurrent_weight_data'], layer['bias_data'] = get_weights_data(
        data_reader, layer['name'], ['kernel', 'recurrent_kernel', 'bias']
    )

    if 'GRU' in layer['class_name']:
        layer['apply_reset_gate'] = 'after' if keras_layer['config']['reset_after'] else 'before'

        # biases array is actually a 2-dim array of arrays (bias + recurrent bias)
        # both arrays have shape: n_units * 3 (z, r, h_cand)
        biases = layer['bias_data']
        layer['bias_data'] = biases[0]
        layer['recurrent_bias_data'] = biases[1]

    if layer['return_sequences']:
        output_shape = [input_shapes[0][0], layer['n_timesteps'], layer['n_out']]
    else:
        output_shape = [input_shapes[0][0], layer['n_out']]

    if layer['return_state']:
        raise Exception('"return_state" of {} layer is not yet supported.')

    return layer, output_shape


@keras_handler('TimeDistributed')
def parse_time_distributed_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'TimeDistributed'

    layer = parse_default_keras_layer(keras_layer, input_names)

    wrapped_keras_layer = keras_layer['config']['layer']
    handler = get_layer_handlers()[wrapped_keras_layer['class_name']]
    if wrapped_keras_layer['class_name'] in ['Sequential', 'Model', 'Functional']:
        if isinstance(data_reader, KerasModelReader):
            nested_data_reader = KerasModelReader(data_reader.model.get_layer(layer['name']).layer)
        else:
            nested_data_reader = KerasNestedFileReader(data_reader, layer['name'])
        layer_list, input_layers, output_layers, output_shapes = parse_keras_model(wrapped_keras_layer, nested_data_reader)

        wrapped_layer = layer.copy()
        wrapped_layer['name'] = wrapped_keras_layer['config']['name']
        wrapped_layer['class_name'] = 'LayerGroup'

        if output_layers is None:
            last_layer = layer_list[-1]['name']
        else:
            last_layer = output_layers[0]
        layer_output_shape = output_shapes[last_layer]

        wrapped_layer['layer_list'] = layer_list
        wrapped_layer['input_layers'] = input_layers if input_layers is not None else []
        wrapped_layer['output_layers'] = output_layers if output_layers is not None else []
        wrapped_layer['data_reader'] = nested_data_reader
        wrapped_layer['output_shape'] = layer_output_shape

        layer['wrapped_layer'] = wrapped_layer
    else:
        if isinstance(data_reader, KerasModelReader):
            nested_data_reader = KerasWrappedLayerReader(data_reader.model.get_layer(layer['name']).layer)
        else:
            nested_data_reader = KerasWrappedLayerFileReader(data_reader, f"{layer['name']}/{layer['name']}")

        wrapped_layer, layer_output_shape = handler(wrapped_keras_layer, [layer['name']], input_shapes, nested_data_reader)
        wrapped_layer['output_shape'] = layer_output_shape
        layer['wrapped_layer'] = wrapped_layer

    if layer_output_shape[0] is None:
        layer_output_shape = layer_output_shape[1:]
    output_shape = input_shapes[0]
    output_shape[len(layer_output_shape) - 1 :] = layer_output_shape
    layer['output_shape'] = output_shape[1:]  # Remove the batch dimension
    layer['n_time_steps'] = output_shape[1]

    return layer, output_shape
