import numpy as np

def quantize(tensor, delta=0.1):
    return np.round(tensor / delta).astype(int)


def binarize(coefficients, num_bits=4):
    return [format(c, f'0{num_bits}b') for c in coefficients]


def context_modeling(importance_map_prev, level_map_prev, importance_map_current, level_map_current):
    prob_model_importance = np.mean(importance_map_prev)
    prob_model_level = np.mean(level_map_prev)

    prob_model_importance_current = prob_model_importance
    prob_model_level_current = prob_model_level

    return prob_model_importance_current, prob_model_level_current


def arithmetic_coding(bitstream):
    # Simplified version using Python standard library
    byte_array = bytearray()

    # Pad bitstream to be a multiple of 8
    bitstream = bitstream.ljust(((len(bitstream) + 7) // 8) * 8, '0')

    # Pack bits into bytes
    for i in range(0, len(bitstream), 8):
        byte = bitstream[i:i + 8]
        byte_array.append(int(byte, 2))

    return bytes(byte_array)


def oabac(tensor, delta=0.1, num_bits=4):
    n1, r, n3 = tensor.shape

    # Flatten the tensor for processing
    flattened_tensor = tensor.reshape(-1)

    # Step 1: Quantization
    quantized = quantize(flattened_tensor, delta)

    # Split into importance and level maps
    importance_map = quantized > 0
    level_map = quantized[importance_map]

    # Step 2: Binarization
    binarized_levels = binarize(level_map, num_bits)

    # Convert importance map to binary string
    bitstream_importance = ''.join('1' if x else '0' for x in importance_map)

    # Concatenate binarized levels into bitstream
    bitstream_levels = ''.join(binarized_levels)

    # Dummy previous values for context modeling
    importance_map_prev = np.random.randint(0, 2, size=importance_map.shape)
    level_map_prev = np.random.randint(0, 10, size=level_map.shape)

    # Step 3: Context Modeling
    prob_model_importance, prob_model_level = context_modeling(importance_map_prev, level_map_prev, importance_map,
                                                               level_map)

    # Step 4: Arithmetic Coding
    coded_importance = arithmetic_coding(bitstream_importance)
    coded_levels = arithmetic_coding(bitstream_levels)

    return coded_importance, coded_levels, prob_model_importance, prob_model_level

def calculate_size(tensor):
    return tensor.nbytes


def calculate_quantized_size(tensor, delta=0.1):
    quantized = quantize(tensor, delta)
    return quantized.nbytes


def calculate_binarized_size(coefficients, num_bits=4):
    binarized_levels = binarize(coefficients, num_bits)
    # Each binarized coefficient is num_bits long
    return len(binarized_levels) * num_bits


def calculate_encoded_size(encoded_bytes):
    return len(encoded_bytes)


def oabac(tensor, delta=0.1, num_bits=4):
    n1, r, n3 = tensor.shape

    # Flatten the tensor for processing
    flattened_tensor = tensor.reshape(-1)

    # Step 1: Quantization
    quantized = quantize(flattened_tensor, delta)

    # Split into importance and level maps
    importance_map = quantized > 0
    level_map = quantized[importance_map]

    # Step 2: Binarization
    binarized_levels = binarize(level_map, num_bits)

    # Convert importance map to binary string
    bitstream_importance = ''.join('1' if x else '0' for x in importance_map)

    # Concatenate binarized levels into bitstream
    bitstream_levels = ''.join(binarized_levels)

    # Dummy previous values for context modeling
    importance_map_prev = np.random.randint(0, 2, size=importance_map.shape)
    level_map_prev = np.random.randint(0, 10, size=level_map.shape)

    # Step 3: Context Modeling
    prob_model_importance, prob_model_level = context_modeling(importance_map_prev, level_map_prev, importance_map,
                                                               level_map)

    # Step 4: Arithmetic Coding
    coded_importance = arithmetic_coding(bitstream_importance)
    coded_levels = arithmetic_coding(bitstream_levels)

    return (coded_importance, coded_levels, prob_model_importance, prob_model_level)

def get_oabac(input, tag):
    tensor = input
    delta = 0.5
    coded_importance, coded_levels, prob_model_importance, prob_model_level = oabac(tensor, delta)
    # print("Probability Model for Importance Map:", prob_model_importance)
    # print("Probability Model for Level Map:", prob_model_level)
    original_size = calculate_size(tensor)
    quantized = quantize(tensor, delta)
    quantized_size = calculate_size(quantized)

    flattened_tensor = tensor.reshape(-1)
    flattened_size = calculate_size(flattened_tensor)

    coded_importance, coded_levels, prob_model_importance, prob_model_level = oabac(tensor, delta)
    encoded_importance_size = calculate_encoded_size(coded_importance)
    encoded_levels_size = calculate_encoded_size(coded_levels)

    # print("Original Tensor Size (bytes):", original_size)
    # print("Quantized Tensor Size (bytes):", quantized_size)
    # print("Flattened Tensor Size (bytes):", flattened_size)
    print('{} encoded Importance Map Size (bytes): {}'.format(tag, encoded_importance_size))
    print('{} encoded Level Map Size (bytes): {}'.format(tag, encoded_levels_size))