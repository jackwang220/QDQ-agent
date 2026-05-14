# import onnx
# import numpy as np
# from onnx import numpy_helper

# # 載入 ONNX 模型
# # model_path = "best.onnx"
# model_path = "wrap_all_best_temp.onnx"
# model = onnx.load(model_path)

# # 定義要額外處理的指定節點名稱
# # target_nodes = [
# #     "module.model.102.rbr_dense.0.weight",
# #     "module.model.103.rbr_dense.0.weight",
# #     "module.model.104.rbr_dense.0.weight",
# #     "module.model.102.rbr_1x1.0.weight",
# #     "module.model.103.rbr_1x1.0.weight",
# #     "module.model.104.rbr_1x1.0.weight",
# #     "module.model.105.m.0.weight",
# #     "module.model.105.m.1.weight",
# #     "module.model.105.m.2.weight",    
# # ]

# target_nodes = [
#     "model.102.rbr_dense.0.weight",
#     "model.103.rbr_dense.0.weight",
#     "model.104.rbr_dense.0.weight",
#     "model.102.rbr_1x1.0.weight",
#     "model.103.rbr_1x1.0.weight",
#     "model.104.rbr_1x1.0.weight",
#     "model.105.m.0.weight",
#     "model.105.m.1.weight",
#     "model.105.m.2.weight",
    
#     ]


# # 找到並處理所有 Conv 層（包括指定節點和其他 Conv 層）
# for initializer in model.graph.initializer:
#     # print(initializer.name)
#     # 如果名稱在 target_nodes 中，或者名稱包含 "conv"（以處理其他卷積層）
#     if initializer.name in target_nodes or "conv" in initializer.name.lower():
#         # 將 FP32 權重轉換為 numpy array
#         weight_array = numpy_helper.to_array(initializer)
        
#         # 將權重數據轉換為 INT8
#         int8_weights = weight_array.astype(np.int8)
        
#         # 將轉換後的 INT8 權重轉換回 ONNX 格式
#         int8_initializer = numpy_helper.from_array(int8_weights, initializer.name)
        
#         # 在模型中替換原始 FP32 權重為 INT8 權重
#         model.graph.initializer.remove(initializer)
#         model.graph.initializer.append(int8_initializer)
#         print(f"已將節點 {initializer.name} 的權重轉換為 INT8。")

# # 保存轉換後的模型
# int8_model_path = "wrap_all_int8_model.onnx"
# onnx.save(model, int8_model_path)

# print(f"模型已成功保存至: {int8_model_path}")


import onnx
import numpy as np
from onnx import numpy_helper
import re
import os
import argparse

def convert_model_to_int8(input_model_path, output_model_path, target_node_patterns=None, verbose=False):
    """
    Converts a model's weights and biases from FP32 to INT8.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted model
        target_node_patterns: List of regex patterns to match node names for conversion
        verbose: Whether to print detailed information about conversions
    """
    print(f"Loading model from {input_model_path}...")
    model = onnx.load(input_model_path)
    
    # Default patterns to match all conv and gemm operators if none provided
    if target_node_patterns is None:
        target_node_patterns = [r'.*weight', r'.*bias', r'.*implicit']
    
    # Compile regex patterns
    compiled_patterns = [re.compile(pattern) for pattern in target_node_patterns]
    
    # Track conversion statistics
    converted_count = 0
    total_count = len(model.graph.initializer)
    
    # Process each initializer in the model
    for initializer in list(model.graph.initializer):  # Create a copy of the list for safe modification
        name = initializer.name
        
        # Check if this initializer matches any of our patterns
        should_convert = any(pattern.match(name) for pattern in compiled_patterns)
        
        if should_convert:
            # Convert to numpy array
            data = numpy_helper.to_array(initializer)
            
            # Store original data type and shape for logging
            orig_dtype = data.dtype
            orig_shape = data.shape
            
            # Convert to INT8
            int8_data = data.astype(np.int8)
            
            # Create a new initializer with INT8 data
            int8_initializer = numpy_helper.from_array(int8_data, name)
            
            # Replace the initializer in the model
            idx = -1
            for i, init in enumerate(model.graph.initializer):
                if init.name == name:
                    idx = i
                    break
            
            if idx != -1:
                model.graph.initializer.remove(initializer)
                model.graph.initializer.insert(idx, int8_initializer)
                converted_count += 1
                
                if verbose:
                    print(f"Converted: {name}")
                    print(f"  Shape: {orig_shape}, Original type: {orig_dtype}, New type: {int8_data.dtype}")
                    print(f"  Original range: [{data.min()}, {data.max()}], New range: [{int8_data.min()}, {int8_data.max()}]")
    
    # Save the converted model
    print(f"Saving INT8 model to {output_model_path}...")
    onnx.save(model, output_model_path)
    
    print(f"Conversion complete. Converted {converted_count}/{total_count} tensors to INT8.")
    return converted_count

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model weights from FP32 to INT8")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, help="Output ONNX model path")
    parser.add_argument("--patterns", type=str, nargs="+", default=None, 
                        help="Regex patterns to match node names for conversion")
    parser.add_argument("--verbose", action="store_true", help="Print detailed conversion information")
    
    args = parser.parse_args()
    
    # If output path not specified, generate one
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_int8.onnx"
    
    convert_model_to_int8(args.input, args.output, args.patterns, args.verbose)

if __name__ == "__main__":
    main()