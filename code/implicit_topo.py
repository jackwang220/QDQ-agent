#!/usr/bin/env python3
"""
ONNX Implicit Function Quantizer

This script specifically addresses quantization of Implicit Function nodes (ImplicitA and ImplicitM) 
in YOLO detection layers which were missed in the original quantization process.

Usage:
    python onnx_implicit_quant_fixer.py --model_path model.onnx --output_path quantized_model.onnx --quant_info fl_values.xlsx
"""

import onnx
import argparse
import pandas as pd
import numpy as np
import re
import logging
from onnx import helper, TensorProto

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Add quantization nodes to ONNX model's Implicit functions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the modified ONNX model")
    parser.add_argument("--quant_info", type=str, required=True, help="Path to the Excel file with FL values")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--detect_layer", type=int, required=True, default=105, help="Layer number of IDetect")
    return parser.parse_args()

def load_fl_values(excel_path):
    """Load FL values from Excel file"""
    logger.info(f"Loading FL values from: {excel_path}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        logger.info(f"Excel columns: {df.columns.tolist()}")
        
        # Make a copy to avoid modifying the original DataFrame
        processed_df = df.copy()
        
        # If FL values aren't in a column called 'fl', try to find it
        if 'fl' not in processed_df.columns:
            for col in processed_df.columns:
                if 'fl' in col.lower():
                    processed_df['fl'] = processed_df[col]
                    logger.info(f"Using column '{col}' for FL values")
                    break
        
        # If layer_number doesn't exist, try to extract it
        if 'layer_number' not in processed_df.columns:
            found_layer_col = False
            
            # First check if there's an existing column with layer numbers
            for col in processed_df.columns:
                if 'layer' in col.lower() and 'number' in col.lower():
                    processed_df['layer_number'] = processed_df[col]
                    found_layer_col = True
                    logger.info(f"Using column '{col}' for layer numbers")
                    break
            
            # If not found, try to extract from layer_name
            if not found_layer_col and 'layer_name' in processed_df.columns:
                # Extract numeric part from layer_name
                def extract_number(x):
                    if pd.isna(x):
                        return None
                    matches = re.findall(r'\d+', str(x))
                    if matches:
                        return int(matches[0])
                    return None
                
                processed_df['layer_number'] = processed_df['layer_name'].apply(extract_number)
                logger.info(f"Extracted layer numbers from 'layer_name' column")
        
        # Ensure we have quantizer roles
        if 'quantizer_role' not in processed_df.columns:
            for col in processed_df.columns:
                if 'role' in col.lower() or 'type' in col.lower():
                    processed_df['quantizer_role'] = processed_df[col]
                    logger.info(f"Using column '{col}' for quantizer roles")
                    break
        
        # Convert data types
        processed_df['fl'] = pd.to_numeric(processed_df['fl'], errors='coerce')
        processed_df['layer_number'] = pd.to_numeric(processed_df['layer_number'], errors='coerce')
        
        # Drop rows with missing essential data
        processed_df = processed_df.dropna(subset=['layer_number', 'fl', 'quantizer_role'])
        
        # Display summary
        logger.info(f"Loaded {len(processed_df)} valid FL values for quantization")
        logger.info(f"Sample data:\n{processed_df[['layer_number', 'quantizer_role', 'fl']].head()}")
        
        return processed_df
    
    except Exception as e:
        logger.error(f"Failed to load FL values: {e}")
        raise

def identify_implicit_nodes(model, layer_num):
    """Identify Implicit function nodes in the model"""
    graph = model.graph
    
    # Find all Add and Mul nodes that are part of the ImplicitA and ImplicitM in layer 118
    implicit_nodes = []
    
    for i, node in enumerate(graph.node):
        # Basic node info
        node_info = {
            'node_idx': i,
            'node_name': node.name,
            'node': node,
            'op_type': node.op_type,
        }
        
        # Check if this is an Add or Mul node in layer 118
        if node.op_type in ['Add', 'Mul']:
            # Look for patterns like /model.118/ia.2/Add or /model.118/im.1/Mul
            # implicit_match = re.search(r'model\.105\/(ia|im)\.(\d+)\/(Add|Mul)', node.name)
            pattern = rf'model\.{layer_num}\/(ia|im)\.(\d+)\/(Add|Mul)'
            implicit_match = re.search(pattern, node.name)
            if implicit_match:
                comp = implicit_match.group(1)  # 'ia' or 'im'
                idx = int(implicit_match.group(2))  # index number
                op = implicit_match.group(3)  # 'Add' or 'Mul'
                
                node_info['component'] = comp
                node_info['index'] = idx
                node_info['layer_number'] = layer_num
                
                # Ensure op_type matches the node name pattern
                if (comp == 'ia' and op == 'Add') or (comp == 'im' and op == 'Mul'):
                    implicit_nodes.append(node_info)
                    logger.info(f"Found Implicit{op} node: {node.name} (component={comp}, index={idx})")
    
    logger.info(f"Identified {len(implicit_nodes)} Implicit function nodes in layer 105")
    # print(implicit_nodes)
    return implicit_nodes
def map_fl_values_to_implicit_nodes(implicit_nodes, fl_df, layer_num):
    """Map FL values to Implicit function nodes"""
    # Extract FL values for layer 118 and specifically for Implicit components
    # layer_105_fl = fl_df[(fl_df['layer_number'] == 105) & 
    #                      (fl_df['layer_name'].str.contains('Implicit', case=False, na=False))]
    target_layer_fl = fl_df[(fl_df['layer_number'] == layer_num) & 
                         (fl_df['layer_name'].str.contains('Implicit', case=False, na=False))]
    
    if len(target_layer_fl) == 0:
        logger.warning("No FL values found for Implicit modules in layer 105")
        return implicit_nodes
    
    # Organize FL values by Implicit component type (ImplicitA, ImplicitM) and index
    fl_values_by_component = {}
    
    for _, row in target_layer_fl.iterrows():
        role = row['quantizer_role']
        fl = float(row['fl'])
        layer_name = row['layer_name']
        full_path = row['full_path']  # Assuming this is the full path to the layer
        
        # Extract component type (ia/im) and index from layer name
        # Assumes format like ".ia.0", ".im.1", etc.

        component_match = re.search(r'\.i([am])\.(\d+)', full_path, re.IGNORECASE)
        if component_match:
            component_type = 'ia' if component_match.group(1).lower() == 'a' else 'im'
            component_index = int(component_match.group(2))
            
            # Create key for this specific component
            component_key = f"{component_type}.{component_index}"
            if component_key not in fl_values_by_component:
                fl_values_by_component[component_key] = {}
            
            fl_values_by_component[component_key][role] = fl
            logger.info(f"Found {role} FL value {fl} for {component_key}")
    
    # Map FL values to each Implicit node
    mapped_nodes = []
    
    for node_info in implicit_nodes:
        # Copy node info
        mapped_node = dict(node_info)

        component = node_info['component']  # 'ia' or 'im'
        index = node_info['index']  # numeric index
        
        # Create component key that matches the one used above
        component_key = f"{component}.{index}"
        
        # Apply FL values if found for this specific component
        if component_key in fl_values_by_component:
            fl_values = fl_values_by_component[component_key]
            
            if 'weight' in fl_values:
                mapped_node['weight_fl'] = fl_values['weight']
            
            if 'output' in fl_values:
                mapped_node['output_fl'] = fl_values['output']
                
            logger.info(f"Mapped FL values for {component_key}: "
                       f"weight_fl={fl_values.get('weight', 'N/A')}, "
                       f"output_fl={fl_values.get('output', 'N/A')}")
        else:
            logger.warning(f"No FL values found for {component_key}")
        
        mapped_nodes.append(mapped_node)
    
    return mapped_nodes

from collections import defaultdict
def add_quantization_to_implicit_nodes(model, mapped_nodes):
    """Add quantization nodes to Implicit function nodes"""
    graph = model.graph
    
    # Keep track of used names to avoid duplicates
    used_names = set()
    # new_nodes = []
    inserted_before = defaultdict(list)
    inserted_after = defaultdict(list)
    
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
    def create_scale_zero_tensors(fl_value, scale_name, zero_name, zero_type=TensorProto.INT8):
        scale_value = float(np.power(2.0, -fl_value))
        scale = helper.make_tensor(scale_name, TensorProto.FLOAT, [], [scale_value])
        zero = helper.make_tensor(zero_name, zero_type, [], [0])
        return scale, zero
    
    # Process each Implicit node
    for node_info in mapped_nodes:

        node = node_info['node']
        comp = node_info['component']  # 'ia' or 'im'
        # idx = node_info['index']       # index number
        idx = node_info['node_idx']       # 真實 Graph Index (例如 150)
        implicit_idx = node_info['index'] # 組件 Index (例如 0)
        op_type = node_info['op_type'] # 'Add' or 'Mul'
        layer_num = node_info.get('layer_number', 'unknown')
        
        # Create a unique identifier for this node
        # node_unique_id = f"105_{comp}{idx}_{op_type.lower()}"
        # node_unique_id = f"{layer_num}_{comp}{idx}_{op_type.lower()}"
        node_unique_id = f"{layer_num}_{comp}{implicit_idx}_{op_type.lower()}"
        # Add input quantization for Implicit functions
        for input_idx, input_tensor in enumerate(node.input):
            if input_idx == 1:
                continue
            # Skip input quantization if no weight_fl value is available
            if node_info.get('weight_fl') is None:
                continue
                
            # Create unique names for tensors and nodes
            input_scale_name = get_unique_name(f"implicit_input{input_idx}_scale_{node_unique_id}")
            input_zero_name = get_unique_name(f"implicit_input{input_idx}_zero_{node_unique_id}")
            quant_node_name = get_unique_name(f"implicit_input{input_idx}_quant_{node_unique_id}")
            dequant_node_name = get_unique_name(f"implicit_input{input_idx}_dequant_{node_unique_id}")
            quant_output = get_unique_name(f"{input_tensor}_quantized")
            dequant_output = get_unique_name(f"{input_tensor}_dequantized")
            
            # Create scale and zero point tensors
            scale, zero = create_scale_zero_tensors(node_info['weight_fl'], input_scale_name, input_zero_name)
            graph.initializer.extend([scale, zero])
            
            # Create quantize and dequantize nodes
            # quant_node = helper.make_node(
            #     "QuantizeLinear",
            #     inputs=[input_tensor, input_scale_name, input_zero_name],
            #     outputs=[quant_output],
            #     name=quant_node_name
            # )
            
            dequant_node = helper.make_node(
                "DequantizeLinear",
                inputs=[input_tensor, input_scale_name, input_zero_name],
                outputs=[dequant_output],
                name=dequant_node_name
            )
            
            # Update the node input to use the dequantized input
            node.input[input_idx] = dequant_output
            inserted_before[idx].append(dequant_node)    ### 標記插入位置
            # new_nodes.extend([dequant_node])
            logger.info(f"Added input quantization for {comp}.{idx} input{input_idx} with FL={node_info['weight_fl']}")
        
        # Add output quantization for Implicit functions
        if node_info.get('output_fl') is not None:
            output_tensor = node.output[0]
            
            # Create unique names for tensors and nodes
            output_scale_name = get_unique_name(f"implicit_output_scale_{node_unique_id}")
            output_zero_name = get_unique_name(f"implicit_output_zero_{node_unique_id}")
            quant_node_name = get_unique_name(f"implicit_output_quant_{node_unique_id}")
            dequant_node_name = get_unique_name(f"implicit_output_dequant_{node_unique_id}")
            quant_output = get_unique_name(f"{output_tensor}_quantized")
            dequant_output = get_unique_name(f"{output_tensor}_dequantized")
            
            # Create scale and zero point tensors
            scale, zero = create_scale_zero_tensors(node_info['output_fl'], output_scale_name, output_zero_name)
            graph.initializer.extend([scale, zero])
            
            # Create quantize and dequantize nodes
            quant_node = helper.make_node(
                "QuantizeLinear",
                inputs=[output_tensor, output_scale_name, output_zero_name],
                outputs=[quant_output],
                name=quant_node_name
            )
            
            dequant_node = helper.make_node(
                "DequantizeLinear",
                inputs=[quant_output, output_scale_name, output_zero_name],
                outputs=[dequant_output],
                name=dequant_node_name
            )
            
            # Update all nodes that consume this output
            for n in graph.node:
                for i, input_name in enumerate(n.input):
                    if input_name == output_tensor:
                        n.input[i] = dequant_output
            inserted_after[idx].extend([quant_node, dequant_node])    ### 標記插入位置
            # new_nodes.extend([quant_node, dequant_node])
            logger.info(f"Added output quantization for {comp}.{idx} with FL={node_info['output_fl']}")
    
    # # Add all new nodes to the graph
    # graph.node.extend(new_nodes)
    # logger.info(f"Added {len(new_nodes)} new quantization nodes to the model")
    # --- 重組 Graph ---
    final_nodes = []
    for i, node in enumerate(graph.node):
        if i in inserted_before:
            final_nodes.extend(inserted_before[i])
        final_nodes.append(node)
        if i in inserted_after:
            final_nodes.extend(inserted_after[i])
            
    del graph.node[:]
    graph.node.extend(final_nodes)
    
    logger.info(f"Rebuilt Implicit graph with {len(final_nodes)} nodes")
    return model

def handle_implicit_function_nodes(model, fl_df, layer_num):
    """Identify and quantize Implicit function nodes"""
    # Identify Implicit function nodes
    implicit_nodes = identify_implicit_nodes(model, layer_num)
    
    if not implicit_nodes:
        logger.warning("No Implicit function nodes found in layer 118")
        return model
    
    # Map FL values to Implicit function nodes
    mapped_nodes = map_fl_values_to_implicit_nodes(implicit_nodes, fl_df, layer_num)
    
    # Add quantization to Implicit function nodes
    model = add_quantization_to_implicit_nodes(model, mapped_nodes)
    
    return model

def main():
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load ONNX model
    logger.info(f"Loading ONNX model from: {args.model_path}")
    try:
        model = onnx.load(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        return 1
    
    # Load FL values from Excel
    fl_df = load_fl_values(args.quant_info)
    
    # Handle Implicit function nodes
    logger.info("Processing Implicit function nodes...")
    quantized_model = handle_implicit_function_nodes(model, fl_df, args.detect_layer)
    
    # Save the quantized model
    logger.info(f"Saving quantized model to: {args.output_path}")
    onnx.save(quantized_model, args.output_path)
    logger.info("Implicit function quantization completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit(main())