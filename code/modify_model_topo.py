import onnx
from onnx import helper, TensorProto
import numpy as np
from collections import defaultdict

def add_input_bias_node(model, bias_value=-0.5):
    """
    Add a -0.5 input bias to the ONNX model by inserting a constant node
    and an addition operation right after the input.
    
    Args:
        model: The ONNX model to modify
        bias_value: The bias value to add (default: -0.5)
    
    Returns:
        The modified ONNX model
    """
    # Get the input tensor name
    input_name = model.graph.input[0].name
    
    # Create a unique name for the bias constant
    bias_name = input_name + "_bias"
    
    # Create a unique name for the addition node output
    biased_input_name = input_name + "_biased"
    
    # Create the constant bias tensor with the same shape as the input
    # We'll use a scalar that will be broadcast to all elements
    bias_tensor = helper.make_tensor(
        name=bias_name,
        data_type=TensorProto.FLOAT,
        dims=[],  # Scalar value for broadcasting
        vals=[bias_value]
    )
    
    # Create a constant node to hold the bias tensor
    bias_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[bias_name],
        name=input_name + "_bias_constant",
        value=bias_tensor
    )
    
    # Create an addition node to add the bias to the input
    add_node = helper.make_node(
        "Add",
        inputs=[input_name, bias_name],
        outputs=[biased_input_name],
        name=input_name + "_add_bias"
    )
    
    # Now we need to update all nodes that consume the original input
    # to use the biased input instead
    for node in model.graph.node:
        for i, input_tensor in enumerate(node.input):
            if input_tensor == input_name:
                node.input[i] = biased_input_name
    
    # Insert the new nodes at the beginning of the graph
    model.graph.node.insert(0, add_node)
    model.graph.node.insert(0, bias_node)
    
    print(f"Added input bias of {bias_value} to the model")
    return model

def add_qdq_after_nodes(model, node_names, fl_values):
    """
    Add quantization and dequantization nodes after specific nodes in an ONNX model.
    
    Args:
        model: The ONNX model to modify
        node_names: List of target node names
        fl_values: List of corresponding FL values for each node
    
    Returns:
        The modified ONNX model with Q/DQ nodes added
    """
    if len(node_names) != len(fl_values):
        raise ValueError("The lengths of node_names and fl_values must match")
        
    graph = model.graph
    used_names = set()
    inserted_after = defaultdict(list)
    # new_nodes = []
    quantized_outputs = set()
    node_name_to_idx = {node.name: i for i, node in enumerate(graph.node)}
    # Helper function to create a unique name
    def get_unique_name(base_name):
        name = base_name
        counter = 0
        while name in used_names:
            counter += 1
            name = f"{base_name}_{counter}"
        used_names.add(name)
        return name
    
    # Helper function to create scale and zero point tensors
    def create_scale_zero_tensors(fl_value, scale_name, zero_name):
        scale_value = float(np.power(2.0, -fl_value))
        scale = helper.make_tensor(scale_name, TensorProto.FLOAT, [], [scale_value])
        zero = helper.make_tensor(zero_name, TensorProto.INT8, [], [0])
        return scale, zero
    
    # Process each target node
    for node_name, fl_value in zip(node_names, fl_values):
        if node_name not in node_name_to_idx:
            print(f"Warning: Node '{node_name}' not found, skipping")
            continue
        target_idx = node_name_to_idx[node_name]
        # Find the target node
        # target_node = None
        target_node = graph.node[target_idx]
        # for node in graph.node:
        #     if node.name == node_name:
        #         target_node = node
        #         break
        
        # if not target_node:
        #     print(f"Warning: Node '{node_name}' not found in the model, skipping")
        #     continue
        
        # Get the output tensor of the target node
        output_tensor = target_node.output[0]
        
        # Skip if this output has already been quantized
        if output_tensor in quantized_outputs:
            print(f"Output {output_tensor} has already been quantized, skipping")
            continue
        
        # Create unique names for new nodes and tensors
        node_id = node_name.replace("/", "_").replace(".", "_")
        scale_name = get_unique_name(f"scale_{node_id}")
        zero_name = get_unique_name(f"zero_{node_id}")
        quant_node_name = get_unique_name(f"quant_{node_id}")
        dequant_node_name = get_unique_name(f"dequant_{node_id}")
        quant_output = get_unique_name(f"{output_tensor}_quantized")
        dequant_output = get_unique_name(f"{output_tensor}_dequantized")
        
        # Create scale and zero point tensors
        scale, zero = create_scale_zero_tensors(fl_value, scale_name, zero_name)
        # graph.initializer.append(scale)
        # graph.initializer.append(zero)
        graph.initializer.extend([scale, zero])
        
        # Create quantize node
        quant_node = helper.make_node(
            "QuantizeLinear",
            inputs=[output_tensor, scale_name, zero_name],
            outputs=[quant_output],
            name=quant_node_name
        )
        
        # Create dequantize node
        dequant_node = helper.make_node(
            "DequantizeLinear",
            inputs=[quant_output, scale_name, zero_name],
            outputs=[dequant_output],
            name=dequant_node_name
        )
        
        # Update all nodes that use the original output to use the dequantized output instead
        for n in graph.node:
            for i, input_name in enumerate(n.input):
                if input_name == output_tensor:
                    n.input[i] = dequant_output
        
        # Add the new nodes to the list
        # new_nodes.extend([quant_node, dequant_node])
        inserted_after[target_idx].extend([quant_node, dequant_node])
        
        # Mark this output as quantized
        quantized_outputs.add(output_tensor)
        
        print(f"Added Q/DQ nodes after '{node_name}' (idx {target_idx}) with FL value {fl_value}")
    
    # Add all new nodes to the graph
    # graph.node.extend(new_nodes)
    # print(f"Added {len(new_nodes)} new Q/DQ nodes to the model")
    final_nodes = []
    for i, node in enumerate(graph.node):
        final_nodes.append(node)
        if i in inserted_after:
            final_nodes.extend(inserted_after[i])
            
    del graph.node[:]
    graph.node.extend(final_nodes)
    print(f"Rebuilt graph with {len(final_nodes)} nodes (Topologically sorted)")
    
    return model

def add_conv_outputs_to_model(model):
    for node in model.graph.node:
        if node.op_type == "Conv":
            output_value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT, None  
            )
            model.graph.output.append(output_value_info)

def add_img_outputs_to_model(model):
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and "images_dequantized" in node.output[0]:
            output_value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT, None 
            )
            model.graph.output.append(output_value_info)

def add_weight_bias_outputs_to_model(model):
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and "model.0.conv.weight_dequantized" in node.output[0]:
            output_value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT, None 
            )
            model.graph.output.append(output_value_info)

        if node.op_type == "DequantizeLinear" and "model.0.conv.bias_dequantized" in node.output[0]:
            output_value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT, None 
            )
            model.graph.output.append(output_value_info)

def add_quantized_conv_outputs_to_model(model):
    """
    Add outputs for all quantized convolution outputs in the ONNX model.
    Specifically targets the dequantized outputs after quantization operations.
    """
    added_outputs = []
    
    for node in model.graph.node:
        # Look for DequantizeLinear nodes that follow Conv nodes
        if node.op_type == "DequantizeLinear" and "Conv_output_0" in node.output[0]:
            # Add this dequantized output to model outputs
            output_name = node.output[0]
            if output_name not in [o.name for o in model.graph.output]:
                output_value_info = helper.make_tensor_value_info(
                    output_name, TensorProto.FLOAT, None  # Shape will be inferred
                )
                model.graph.output.append(output_value_info)
                added_outputs.append(output_name)
    
    print(f"Added {len(added_outputs)} quantized convolution outputs to model")
    return model

def add_1st_conv_outputs_to_model(model):
    for node in model.graph.node:
        if node.op_type == "Conv" and "/model.0/conv/Conv_output_0" in node.output[0]:
            output_value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT, None 
            )
            model.graph.output.append(output_value_info)

def add_specific_outputs_to_model(model, target_nodes):
    for node in model.graph.node:
        if node.name in target_nodes:
            output_value_info = helper.make_tensor_value_info(
                node.output[0], TensorProto.FLOAT, None 
            )
            model.graph.output.append(output_value_info)

def remove_outputs_by_names(model, output_names_to_remove):
    outputs_to_keep = [
        output for output in model.graph.output
        if output.name not in output_names_to_remove
    ]
    del model.graph.output[:]
    model.graph.output.extend(outputs_to_keep)

def remove_nodes_by_names(model, node_names_to_remove):
    remaining_nodes = [
        node for node in model.graph.node
        if node.name not in node_names_to_remove
    ]
    del model.graph.node[:]
    model.graph.node.extend(remaining_nodes)
    print(f"remove node: {node_names_to_remove}")

def add_transpose_after_node(model, target_node_name, perm, transpose_node_name, output_name):
    target_idx = -1
    target_node = None
    # for node in model.graph.node:
    #     if node.name == target_node_name:
    #         target_node = node
    #         break
    for i, node in enumerate(model.graph.node):
        if node.name == target_node_name:
            target_node = node
            target_idx = i
            break
    if not target_node:
        raise ValueError(f"node not found: {target_node_name}")
    
    # 添加 Transpose 節點
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[target_node.output[0]],
        outputs=[output_name],
        name=transpose_node_name,
        perm=perm
    )
    # model.graph.node.append(transpose_node)
    model.graph.node.insert(target_idx + 1, transpose_node)
    
    # 將 Transpose 的輸出設為模型的輸出
    output_value_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, None  # 不指定形狀
    )
    model.graph.output.append(output_value_info)

    print(f"add transpose after {target_node_name} at index {target_idx + 1}, output name: {output_name}")

# Example usage of the combined functionality
if __name__ == "__main__":
    # load ONNX model
    model_path = "output_topo.onnx"
    # model_path = "yolov7-OD-coco-416-w40p50-LR-best.onnx"
    model = onnx.load(model_path)

    # Add the input bias node to the model
    model = add_input_bias_node(model, bias_value=-0.5)

    # Define nodes and FL values for adding Q/DQ nodes
    # node_names = [
    #     "/model.51/Concat",
    #     "/model.51/Concat_1",
    #     "/model.53/Resize",
    #     "/model.65/Resize",
    #     "/model.102/act/LeakyRelu",
    #     "/model.103/act/LeakyRelu",
    #     "/model.104/act/LeakyRelu"
    # ]
    node_names = [
        "/model.51/Concat",
        "/model.51/Concat_1",
        "/model.53/Resize",
        "/model.65/Resize",
        "/model.102/act/Sigmoid",
        "/model.103/act/Sigmoid",
        "/model.104/act/Sigmoid"
    ]
    
    fl_values = [
        4.0,
        4.0,
        4.0,
        4.0,
        3.0,
        3.0,
        3.0,
    ]
    
    # Add Q/DQ nodes after specified nodes
    model = add_qdq_after_nodes(model, node_names, fl_values)
    
    # define Transpose configs
    # transpose_configs = [
    #     {"target_node_name": "out_dequant_DetectionConv_0_132_77", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_0", "output_name": "transposed_output_0"},
    #     {"target_node_name": "out_dequant_DetectionConv_1_135_77", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_1", "output_name": "transposed_output_1"},
    #     {"target_node_name": "out_dequant_DetectionConv_2_138_77", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_2", "output_name": "transposed_output_2"}
    # ]

    # transpose_configs = [
    #     {"target_node_name": "/model.105/m.0/Conv", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_0", "output_name": "transposed_output_0"},
    #     {"target_node_name": "/model.105/m.1/Conv", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_1", "output_name": "transposed_output_1"},
    #     {"target_node_name": "/model.105/m.2/Conv", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_2", "output_name": "transposed_output_2"}
    # ]

    # transpose_configs = [
    #     {"target_node_name": "/model.105/im.0/Mul", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_0", "output_name": "transposed_output_0"},
    #     {"target_node_name": "/model.105/im.1/Mul", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_1", "output_name": "transposed_output_1"},
    #     {"target_node_name": "/model.105/im.2/Mul", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_2", "output_name": "transposed_output_2"}
    # ]

    transpose_configs = [
        {"target_node_name": "implicit_output_dequant_105_im0_mul", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_0", "output_name": "transposed_output_0"},
        {"target_node_name": "implicit_output_dequant_105_im1_mul", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_1", "output_name": "transposed_output_1"},
        {"target_node_name": "implicit_output_dequant_105_im2_mul", "perm": [0, 2, 3, 1], "transpose_node_name": "transpose_2", "output_name": "transposed_output_2"}
    ]
    
    # add Transpose nodes to model
    for config in transpose_configs:
        add_transpose_after_node(
            model,
            config["target_node_name"],
            config["perm"],
            config["transpose_node_name"],
            config["output_name"]
        )

    # keep other functions
    # output_names_to_remove = ['455', '469', 'output']
    # output_names_to_remove = ["298", "286", "output"]
    output_names_to_remove = ['452', '466', 'output']
    node_names_to_remove = ["/model.105/Reshape_2", "/model.105/Transpose_2", "/model.105/Transpose", "/model.105/Reshape", "/model.105/Reshape_1", "/model.105/Transpose_1"]
    # node_names_to_remove = ["/model.77/Reshape_2", "/model.77/Transpose_2", "/model.77/Transpose", "/model.77/Reshape", "/model.77/Reshape_1", "/model.77/Transpose_1"]

    remove_outputs_by_names(model, output_names_to_remove)
    remove_nodes_by_names(model, node_names_to_remove)
    # add_1st_conv_outputs_to_model(model)
    # add_img_outputs_to_model(model)
    # add_weight_bias_outputs_to_model(model)
    # add_quantized_conv_outputs_to_model(model)
    # add_conv_outputs_to_model(model)

    # save modified model
    modified_model_path = "model_bias_topo.onnx"
    onnx.save(model, modified_model_path)

    print(f"success modify model and save to {modified_model_path}")