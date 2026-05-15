#!/usr/bin/env python3
"""
ONNX Model Quantization Script

This script adds quantization nodes to ONNX models based on FL values from an Excel file.
It handles:
1. Standard Conv layers with input/weight/bias/output quantization
2. LeakyReLU activations
3. Implicit functions in detection layer (layer 105) - both weight and output quantization
4. RepConv Add operations (layers 102-104)
5. Concat operations
"""

import onnx
import argparse
import pandas as pd
import numpy as np
import re
import logging
from onnx import helper, TensorProto

# Defaults — overridden at runtime by CLI args from the agent
SPPSCSPC_LR_LAYER = 51
DETECTION_LAYER = 105
REPCONV_LAYERS = [102, 103, 104]
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Add quantization nodes to ONNX model based on FL values")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the modified ONNX model")
    parser.add_argument("--quant_info", type=str, required=True, help="Path to the Excel file with FL values")
    parser.add_argument("--sppcspc-layer", type=int, default=None, help="SPPCSPC layer index (overrides default)")
    parser.add_argument("--repconv-layers", type=str, default="", help="Comma-separated RepConv layer indices")
    parser.add_argument("--detection-layer", type=int, default=None, help="Detection layer index (overrides default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def load_fl_values(excel_path):
    """Load FL values from Excel file with enhanced RepConv sub-module detection"""
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
        
        # Add a new column for sub-module identification, especially for RepConv
        processed_df['sub_module'] = None
        
        # Identify sub-modules from full_path or layer_name
        for i, row in processed_df.iterrows():
            layer_num = row.get('layer_number')
            if pd.isna(layer_num):
                continue
                
            layer_num = int(layer_num)
            full_path = str(row.get('full_path', ''))
            layer_name = str(row.get('layer_name', ''))
            combined_path = full_path + layer_name
            
            # Identify RepConv sub-modules for layers 102-104
            if layer_num in REPCONV_LAYERS:
                # Look for rbr_dense or rbr_1x1 in the path/name
                if 'rbr_dense' in combined_path.lower():
                    processed_df.at[i, 'sub_module'] = 'rbr_dense'
                elif 'rbr_1x1' in combined_path.lower():
                    processed_df.at[i, 'sub_module'] = 'rbr_1x1'
                elif re.search(r'rbr_(?:dense|1x1)', combined_path, re.IGNORECASE) is None:
                    # This might be the Add operation or another part
                    if 'add' in combined_path.lower():
                        processed_df.at[i, 'sub_module'] = 'add'
            elif layer_num == DETECTION_LAYER:
                # Check for detection layer convolution indices
                m_match = re.search(r'm\.(\d+)', combined_path)
                if m_match:
                    detect_idx = int(m_match.group(1))
                    processed_df.at[i, 'sub_module'] = f"m.{detect_idx}"
                    # This indicates it's one of the detection convolutions
                
                # Check for implicit components
                comp_match = re.search(r'i([am])\.(\d+)', combined_path)
                if comp_match:
                    comp = 'ia' if comp_match.group(1) == 'a' else 'im'
                    idx = comp_match.group(2)
                    processed_df.at[i, 'sub_module'] = f"{comp}.{idx}"

            # Identify SPPCSPC sub-modules for layer 51
            elif layer_num == SPPSCSPC_LR_LAYER:
                cv_match = re.search(r'cv(\d+)', combined_path)
                if cv_match:
                    processed_df.at[i, 'sub_module'] = f"cv{cv_match.group(1)}"
                
                m_match = re.search(r'm\.(\d+)', combined_path)
                if m_match:
                    processed_df.at[i, 'sub_module'] = f"m.{m_match.group(1)}"
        
        # Convert data types
        processed_df['fl'] = pd.to_numeric(processed_df['fl'], errors='coerce')
        processed_df['layer_number'] = pd.to_numeric(processed_df['layer_number'], errors='coerce')
        
        # Drop rows with missing essential data
        processed_df = processed_df.dropna(subset=['layer_number', 'fl', 'quantizer_role'])
        
        # Display summary
        logger.info(f"Loaded {len(processed_df)} valid FL values for quantization")
        logger.info(f"Sample data:\n{processed_df[['layer_number', 'quantizer_role', 'fl', 'sub_module']].head()}")
        
        # Count sub-modules for RepConv layers
        repconv_sub_modules = processed_df[processed_df['layer_number'].isin(REPCONV_LAYERS)]['sub_module'].value_counts()
        if not repconv_sub_modules.empty:
            logger.info(f"RepConv sub-module distribution: {repconv_sub_modules.to_dict()}")
        
        return processed_df
    
    except Exception as e:
        logger.error(f"Failed to load FL values: {e}")
        raise


def analyze_model_structure(model):
    """Analyze the ONNX model structure to identify nodes for quantization"""
    graph = model.graph
    
    # Create dictionaries to track different node types
    conv_nodes = []
    concat_nodes = []
    MaxPool_nodes = []
    special_nodes = {
        'repconv_add': [],    # Add operations in RepConv
        'repconv_dense': [],  # Dense Conv operations in RepConv layers
        'repconv_1x1': [],    # 1x1 Conv operations in RepConv layers
        'implicit_nodes': [], # Implicit nodes in detection layer
        'leakyrelu': [],      # LeakyReLU activations
        'sppcspc_conv': [],   # SPPCSPC Conv sub-modules
        'sppcspc_act': [],    # SPPCSPC activation sub-modules
        'sppcspc_maxpool': [] # SPPCSPC MaxPool nodes
    }
    
    # Process all initializers to identify weights
    weight_initializers = {}
    for initializer in graph.initializer:
        weight_initializers[initializer.name] = initializer
    
    # Process all nodes
    for i, node in enumerate(graph.node):
        # Basic node info
        node_info = {
            'node_idx': i,
            'node_name': node.name,
            'node': node,
            'op_type': node.op_type,
            'node_type': node.op_type,  # Add explicit node_type field for clearer differentiation
        }
        
        # Try to extract layer info from name
        layer_match = re.search(r'model\.(\d+)', node.name)
        if layer_match:
            node_info['layer_number'] = int(layer_match.group(1))
        
        # Special handling for layer 51 (SPPCSPC_LR)
        is_layer_51 = node_info.get('layer_number') == SPPSCSPC_LR_LAYER
        
        # Identify RepConv layers (102-104)
        is_repconv_layer = node_info.get('layer_number') in REPCONV_LAYERS
        
        # Categorize nodes
        if node.op_type == 'Conv':
            # Check if it has bias by examining input length
            has_bias = len(node.input) > 2
            node_info['has_bias'] = has_bias
            node_info['weight_name'] = node.input[1] if len(node.input) > 1 else None
            node_info['bias_name'] = node.input[2] if has_bias else None
            node_info['node_type'] = 'Conv'  # Be explicit about node type
            
            # Special handling for RepConv layers (102-104)
            if is_repconv_layer:
                # Look for rbr_dense or rbr_1x1 in the node name
                rbr_dense_match = re.search(r'rbr_dense', node.name, re.IGNORECASE)
                rbr_1x1_match = re.search(r'rbr_1x1', node.name, re.IGNORECASE)
                
                if rbr_dense_match:
                    node_info['sub_module'] = 'rbr_dense'
                    node_info['node_type'] = f"RepConv_Dense_{node_info['layer_number']}"
                    logger.info(f"Found RepConv rbr_dense in layer {node_info['layer_number']} node {node.name}")
                    special_nodes['repconv_dense'].append(node_info)
                elif rbr_1x1_match:
                    node_info['sub_module'] = 'rbr_1x1'
                    node_info['node_type'] = f"RepConv_1x1_{node_info['layer_number']}"
                    logger.info(f"Found RepConv rbr_1x1 in layer {node_info['layer_number']} node {node.name}")
                    special_nodes['repconv_1x1'].append(node_info)
                else:
                    # If we can't detect from name, try to infer from position and connections
                    patterns = [
                        r'/(\d+)\.rbr_(\w+)/',
                        r'model\.(\d+)\.rbr_(\w+)',
                        r'layer(\d+)\.rbr_(\w+)',
                        r'/rbr_(\w+)/',
                    ]
                    detected = False
                    for pattern in patterns:
                        sub_match = re.search(pattern, node.name, re.IGNORECASE)
                        if sub_match:
                            sub_type = sub_match.group(2) if len(sub_match.groups()) > 1 else sub_match.group(1)
                            if 'dense' in sub_type.lower():
                                node_info['sub_module'] = 'rbr_dense'
                                node_info['node_type'] = f"RepConv_Dense_{node_info['layer_number']}"
                                special_nodes['repconv_dense'].append(node_info)
                            elif '1x1' in sub_type.lower():
                                node_info['sub_module'] = 'rbr_1x1'
                                node_info['node_type'] = f"RepConv_1x1_{node_info['layer_number']}"
                                special_nodes['repconv_1x1'].append(node_info)
                            detected = True
                            logger.info(f"Inferred RepConv sub-module {node_info['sub_module']} for layer {node_info['layer_number']} node {node.name}")
                            break
                    
                    if not detected:
                        node_info['node_type'] = f"RepConv_Unknown_{node_info['layer_number']}"
                        logger.warning(f"Couldn't determine RepConv sub-module type for layer {node_info['layer_number']} node {node.name}")
            
            ## 0518 add final layer repeated conv
            elif node_info.get('layer_number') == DETECTION_LAYER:
                # Look for m.X pattern in node name (detection convolution index)
                detect_match = re.search(rf'model\.{DETECTION_LAYER}\.m\.(\d+)', node.name)
                if detect_match:
                    detect_idx = int(detect_match.group(1))
                    node_info['detect_index'] = detect_idx
                    node_info['sub_module'] = f"m.{detect_idx}"
                    node_info['node_type'] = f'DetectionConv_{detect_idx}'
                    logger.info(f"Found Detection Conv {detect_idx} in layer 105 node {node.name}")
                else:
                    # Try alternative patterns for detection convs
                    alt_detect_match = re.search(rf'{DETECTION_LAYER}.*?m\.(\d+)', node.name)
                    if alt_detect_match:
                        detect_idx = int(alt_detect_match.group(1))
                        node_info['detect_index'] = detect_idx
                        node_info['sub_module'] = f"m.{detect_idx}"
                        node_info['node_type'] = f'DetectionConv_{detect_idx}'
                        logger.info(f"Found Detection Conv {detect_idx} in layer 105 using alternative pattern")


            # Check if this is part of SPPCSPC_LR in layer 51
            elif is_layer_51:
                # Extract the sub-module path more comprehensively
                patterns = [
                    r'model\.51\.(cv\d+)\.conv',  # Pattern for cv1.conv, cv2.conv, etc.
                    r'model\.51\.(cv\d+)',        # Pattern for cv1, cv2, etc.
                    r'51\.(cv\d+)',              # Alternative pattern
                    r'\.51\.(cv\d+)',            # Another alternative
                    r'/(cv\d+)/Conv'             # Pattern often seen in ONNX node names
                ]
                
                for pattern in patterns:
                    spp_match = re.search(pattern, node.name)
                    if spp_match:
                        sub_module = spp_match.group(1)
                        node_info['sub_module'] = sub_module
                        node_info['conv_index'] = int(sub_module[2:])  # Extract X from cvX
                        node_info['node_type'] = f'SPPCSPC_Conv_{sub_module}'  # More specific node type
                        logger.info(f"Found SPPCSPC Conv sub-module {sub_module} (index: {node_info['conv_index']}) in node {node.name}")
                        special_nodes['sppcspc_conv'].append(node_info)
                        break
                
                # If no match found but node is definitely layer 51, try more aggressively
                if 'sub_module' not in node_info:
                    # Try to extract just the number after cv
                    cv_num_match = re.search(r'51.*?cv(\d+)', node.name)
                    if cv_num_match:
                        cv_num = cv_num_match.group(1)
                        sub_module = f"cv{cv_num}"
                        node_info['sub_module'] = sub_module
                        node_info['conv_index'] = int(cv_num)
                        node_info['node_type'] = f'SPPCSPC_Conv_{sub_module}'
                        logger.info(f"Inferred SPPCSPC Conv sub-module {sub_module} from node {node.name}")
                        special_nodes['sppcspc_conv'].append(node_info)
                    else:
                        # If all else fails, use a placeholder and log this issue
                        node_info['node_type'] = 'SPPCSPC_Conv_unknown'
                        logger.warning(f"Layer 51 Conv node without detectable sub-module: {node.name}")
            
            # Add to regular conv_nodes for normal processing
            conv_nodes.append(node_info)
            
        elif node.op_type == 'Concat':
            node_info['node_type'] = 'Concat'
            concat_nodes.append(node_info)
        
        elif node.op_type == 'MaxPool':
            # Store basic information for all MaxPool nodes
            node_info['node_type'] = 'MaxPool'
            MaxPool_nodes.append(node_info)
            
            # Special handling for SPPCSPC MaxPool nodes
            if is_layer_51:
                patterns = [
                    r'model\.51\.m\.(\d+)',  # Standard pattern
                    r'51\.m\.(\d+)',         # Alternative
                    r'\.51\.m\.(\d+)',       # Another alternative
                    r'/m\.(\d+)/'            # Pattern often seen in ONNX
                ]
                
                for pattern in patterns:
                    spp_match = re.search(pattern, node.name)
                    if spp_match:
                        pool_index = int(spp_match.group(1))
                        node_info['pool_index'] = pool_index
                        node_info['sub_module'] = f"m.{pool_index}"  # Add sub_module for consistency
                        node_info['node_type'] = f'SPPCSPC_MaxPool_{pool_index}'
                        logger.info(f"Found SPPCSPC MaxPool {pool_index} in node {node.name}")
                        special_nodes['sppcspc_maxpool'].append(node_info)
                        break
                
                # Similar fallback as with Conv
                if 'pool_index' not in node_info:
                    node_info['node_type'] = 'SPPCSPC_MaxPool_unknown'
                    logger.warning(f"Layer 51 MaxPool node without detectable index: {node.name}")
        
        elif node.op_type in ['LeakyRelu', 'Relu']:
            # Process all activation nodes
            node_info['node_type'] = 'Activation_' + node.op_type
            special_nodes['leakyrelu'].append(node_info)
            
            # Special handling for SPPCSPC activation nodes
            if is_layer_51:
                patterns = [
                    r'model\.51\.(cv\d+)\.act',  # Pattern for cv1.act, cv2.act, etc.
                    r'51\.(cv\d+)\.act',        # Alternative
                    r'\.51\.(cv\d+)\.act',      # Another alternative
                    r'/(cv\d+)/.*?Relu'         # Pattern for activation in ONNX
                ]
                
                for pattern in patterns:
                    spp_match = re.search(pattern, node.name)
                    if spp_match:
                        sub_module = spp_match.group(1)
                        node_info['sub_module'] = sub_module
                        node_info['conv_index'] = int(sub_module[2:])  # Extract X from cvX
                        node_info['node_type'] = f'SPPCSPC_Activation_{sub_module}'
                        logger.info(f"Found SPPCSPC Activation for sub-module {sub_module} in node {node.name}")
                        special_nodes['sppcspc_act'].append(node_info)
                        break
                
                # Similar fallback
                if 'sub_module' not in node_info:
                    cv_num_match = re.search(r'51.*?cv(\d+).*?(Relu|relu|act)', node.name)
                    if cv_num_match:
                        cv_num = cv_num_match.group(1)
                        sub_module = f"cv{cv_num}"
                        node_info['sub_module'] = sub_module
                        node_info['conv_index'] = int(cv_num)
                        node_info['node_type'] = f'SPPCSPC_Activation_{sub_module}'
                        logger.info(f"Inferred SPPCSPC Activation sub-module {sub_module} from node {node.name}")
                        special_nodes['sppcspc_act'].append(node_info)
                    else:
                        node_info['node_type'] = 'SPPCSPC_Activation_unknown'
            
        elif node.op_type == 'Add':
            node_info['node_type'] = 'Add'
            # Check if this is a RepConv Add operation (layers 102-104)
            if is_repconv_layer:
                node_info['node_type'] = f'RepConv_Add_{node_info["layer_number"]}'
                special_nodes['repconv_add'].append(node_info)
        
        # Check for implicit nodes in detection layer (layer 105)
        if node_info.get('layer_number') == DETECTION_LAYER:
            # Check if it's an ia or im node
            implicit_match = re.search(rf'model\.{DETECTION_LAYER}\.(ia|im)\.(\d+)', node.name)
            if implicit_match:
                comp = implicit_match.group(1)  # 'ia' or 'im'
                idx = int(implicit_match.group(2))  # index
                node_info['component'] = comp
                node_info['index'] = idx
                node_info['node_type'] = f'Implicit_{comp}_{idx}'
                special_nodes['implicit_nodes'].append(node_info)
    
    # Log summary of identified nodes
    logger.info(f"Found {len(conv_nodes)} Conv nodes")
    logger.info(f"Found {len(concat_nodes)} Concat nodes")
    logger.info(f"Found {len(MaxPool_nodes)} MaxPool nodes")
    logger.info(f"Found {len(special_nodes['leakyrelu'])} LeakyReLU/Activation nodes")
    logger.info(f"Found {len(special_nodes['repconv_add'])} RepConv Add nodes")
    logger.info(f"Found {len(special_nodes['repconv_dense'])} RepConv Dense Conv nodes")
    logger.info(f"Found {len(special_nodes['repconv_1x1'])} RepConv 1x1 Conv nodes")
    logger.info(f"Found {len(special_nodes['implicit_nodes'])} Implicit nodes in layer 105")
    
    # Log specific info about SPPCSPC components
    logger.info(f"Found {len(special_nodes['sppcspc_conv'])} SPPCSPC Conv sub-modules in layer 51")
    logger.info(f"Found {len(special_nodes['sppcspc_act'])} SPPCSPC Activation sub-modules in layer 51")
    logger.info(f"Found {len(special_nodes['sppcspc_maxpool'])} SPPCSPC MaxPool nodes in layer 51")
    
    # Verify that the node_type field is properly set for all nodes
    node_types = {}
    for node in conv_nodes + concat_nodes + MaxPool_nodes:
        node_type = node.get('node_type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    logger.info(f"Node types distribution: {node_types}")
    
    return {
        'conv_nodes': conv_nodes,
        'concat_nodes': concat_nodes,
        'MaxPool_nodes': MaxPool_nodes,
        'special_nodes': special_nodes,
        'weight_initializers': weight_initializers
    }

def map_fl_values_to_nodes(model_structure, fl_df):
    """Map FL values from dataframe to model nodes with improved RepConv handling"""
    # Create enhanced lookup table for FL values by layer, node type, and role
    fl_lookup = {}
    
    # Process all FL values from the Excel file
    for _, row in fl_df.iterrows():
        layer = int(row['layer_number'])
        role = row['quantizer_role']
        fl = float(row['fl'])
        full_path = row.get('full_path', '')
        layer_name = row.get('layer_name', '')
        sub_module = row.get('sub_module', None)
        
        # Initialize the layer in the lookup table if not present
        if layer not in fl_lookup:
            fl_lookup[layer] = {}
        
        # Try to determine node type from available information
        node_type = None
        
        # Special handling for RepConv layers (102-104)
        if layer in REPCONV_LAYERS:
            if sub_module == 'rbr_dense':
                node_type = f"RepConv_Dense_{layer}"
            elif sub_module == 'rbr_1x1':
                node_type = f"RepConv_1x1_{layer}"
            elif sub_module == 'add':
                node_type = f"RepConv_Add_{layer}"
            else:
                # If no sub_module specified, try to infer from role and path
                if 'dense' in str(full_path) + str(layer_name):
                    node_type = f"RepConv_Dense_{layer}"
                elif '1x1' in str(full_path) + str(layer_name):
                    node_type = f"RepConv_1x1_{layer}"
                elif 'add' in role.lower() or 'add' in str(full_path) + str(layer_name):
                    node_type = f"RepConv_Add_{layer}"
                else:
                    # Use a generic RepConv type as fallback
                    node_type = f"RepConv_Conv_{layer}"
        
        # Check for SPPCSPC sub-modules (layer 51)
        elif layer == SPPSCSPC_LR_LAYER:
            # Use sub_module if available
            if sub_module:
                if sub_module.startswith('cv'):
                    # Determine if this is a Conv or Activation based on role and path
                    if role == 'output' and ('act' in str(full_path) or 'act' in str(layer_name)):
                        node_type = f'SPPCSPC_Activation_{sub_module}'
                    else:
                        node_type = f'SPPCSPC_Conv_{sub_module}'
                elif sub_module.startswith('m.'):
                    node_type = f'SPPCSPC_MaxPool_{sub_module[2:]}'
            else:
                # Fall back to looking for patterns in full_path or layer_name
                cv_match = re.search(r'cv(\d+)', str(full_path) + str(layer_name))
                m_match = re.search(r'm\.(\d+)', str(full_path) + str(layer_name))
                
                if cv_match:
                    cv_num = cv_match.group(1)
                    # Determine if this is a Conv or Activation based on role and path
                    if role == 'output' and ('act' in str(full_path) or 'act' in str(layer_name)):
                        node_type = f'SPPCSPC_Activation_cv{cv_num}'
                    else:
                        node_type = f'SPPCSPC_Conv_cv{cv_num}'
                elif m_match:
                    pool_idx = m_match.group(1)
                    node_type = f'SPPCSPC_MaxPool_{pool_idx}'
        
        # Check for Implicit nodes in layer 105
        elif layer == DETECTION_LAYER:
            # Check if this is a detection convolution first
            m_match = re.search(r'm\.(\d+)', str(full_path) + str(layer_name))
            if m_match:
                detect_idx = int(m_match.group(1))
                node_type = f'DetectionConv_{detect_idx}'
            else:
                # Try to extract implicit component info (existing code)
                comp_match = re.search(r'i([am])\.(\d+)', str(full_path) + str(layer_name))
                if comp_match:
                    comp = 'ia' if comp_match.group(1) == 'a' else 'im'
                    idx = comp_match.group(2)
                    node_type = f'Implicit_{comp}_{idx}'
        
        # If node_type is still None, infer from role
        if node_type is None:
            if 'weight' in role.lower() or 'bias' in role.lower():
                node_type = 'Conv'
            elif 'act' in role.lower() or role.lower() == 'output' and ('act' in str(full_path) or 'leaky' in str(full_path).lower()):
                node_type = 'Activation_LeakyRelu'
            elif 'concat' in str(full_path).lower() or 'concat' in str(layer_name).lower():
                node_type = 'Concat'
            elif 'maxpool' in str(full_path).lower() or 'maxpool' in str(layer_name).lower():
                node_type = 'MaxPool'
            else:
                # Default case: use a basic type based on role
                node_type = {'input': 'Conv', 'output': 'Conv', 'weight': 'Conv', 'bias': 'Conv'}.get(role, 'Unknown')
        
        # Create node_type entry if it doesn't exist
        if node_type not in fl_lookup[layer]:
            fl_lookup[layer][node_type] = {}
        
        # Store the FL value by role for this layer and node type
        fl_lookup[layer][node_type][role] = fl
        logger.info(f"Mapped FL value {fl} for layer {layer}, node type {node_type}, role {role}")
    
    # Now map FL values to nodes using the enhanced lookup
    
    # 1. Map Conv nodes
    mapped_convs = []
    for conv in model_structure['conv_nodes']:
        layer_num = conv.get('layer_number')
        node_type = conv.get('node_type', 'Conv')
        
        if layer_num is not None and layer_num in fl_lookup:
            # Special handling for RepConv layers (102-104)
            if layer_num in REPCONV_LAYERS:
                # First try to match the exact node_type for RepConv components
                if node_type in fl_lookup[layer_num]:
                    fl_values = fl_lookup[layer_num][node_type]
                    
                    # Map input FL value if available
                    if 'input' in fl_values:
                        conv['input_fl'] = fl_values['input']
                    
                    # Map weight FL value if available
                    if 'weight' in fl_values:
                        conv['weight_fl'] = fl_values['weight']
                    
                    # Map bias FL value if available and node has bias
                    if 'bias' in fl_values and conv['has_bias']:
                        conv['bias_fl'] = fl_values['bias']
                    
                    # Map output FL value if available
                    if 'output' in fl_values:
                        conv['output_fl'] = fl_values['output']
                    
                    logger.info(f"Matched exact node_type: Mapped FL values for {node_type} in layer {layer_num}: "
                               f"input={conv.get('input_fl')}, weight={conv.get('weight_fl')}, "
                               f"bias={conv.get('bias_fl')}, output={conv.get('output_fl')}")
                
                # If not found, try using sub_module to find appropriate node type
                elif 'sub_module' in conv:
                    sub_module = conv['sub_module']  # 'rbr_dense' or 'rbr_1x1'
                    
                    # Look for matching node types with this sub_module
                    matched = False
                    for nt in fl_lookup[layer_num]:
                        if sub_module in nt.lower():
                            fl_values = fl_lookup[layer_num][nt]
                            
                            if 'input' in fl_values:
                                conv['input_fl'] = fl_values['input']
                            if 'weight' in fl_values:
                                conv['weight_fl'] = fl_values['weight']
                            if 'bias' in fl_values and conv['has_bias']:
                                conv['bias_fl'] = fl_values['bias']
                            if 'output' in fl_values:
                                conv['output_fl'] = fl_values['output']
                            
                            logger.info(f"Matched by sub_module: Mapped FL values for {sub_module} in layer {layer_num}: "
                                       f"input={conv.get('input_fl')}, weight={conv.get('weight_fl')}, "
                                       f"bias={conv.get('bias_fl')}, output={conv.get('output_fl')}")
                            matched = True
                            break
                    
                    # If still not matched, fall back to generic RepConv Conv
                    if not matched:
                        generic_types = [f"RepConv_Conv_{layer_num}", "Conv"]
                        for gen_type in generic_types:
                            if gen_type in fl_lookup[layer_num]:
                                fl_values = fl_lookup[layer_num][gen_type]
                                
                                if 'input' in fl_values and 'input_fl' not in conv:
                                    conv['input_fl'] = fl_values['input']
                                if 'weight' in fl_values and 'weight_fl' not in conv:
                                    conv['weight_fl'] = fl_values['weight']
                                if 'bias' in fl_values and conv['has_bias'] and 'bias_fl' not in conv:
                                    conv['bias_fl'] = fl_values['bias']
                                if 'output' in fl_values and 'output_fl' not in conv:
                                    conv['output_fl'] = fl_values['output']
                                
                                logger.info(f"Fallback: Mapped generic FL values for {gen_type} in layer {layer_num}: "
                                           f"input={conv.get('input_fl')}, weight={conv.get('weight_fl')}, "
                                           f"bias={conv.get('bias_fl')}, output={conv.get('output_fl')}")
                                break
                
                # If still no match, log warning
                if not any(k in conv for k in ['input_fl', 'weight_fl', 'bias_fl', 'output_fl']):
                    logger.warning(f"No FL values could be matched for {node_type} in layer {layer_num}")
            ## 0518 add final layer repeated conv
            elif layer_num == DETECTION_LAYER and 'detect_idx' in conv:
                # Use the specific detection conv node type for lookup
                node_type = f'DetectionConv_{conv["detect_index"]}'
                
                if node_type in fl_lookup[layer_num]:
                    fl_values = fl_lookup[layer_num][node_type]
                    
                    if 'input' in fl_values:
                        conv['input_fl'] = fl_values['input']
                    if 'weight' in fl_values:
                        conv['weight_fl'] = fl_values['weight']
                    if 'bias' in fl_values and conv['has_bias']:
                        conv['bias_fl'] = fl_values['bias']
                    if 'output' in fl_values:
                        conv['output_fl'] = fl_values['output']
        
                logger.info(f"Mapped FL values for Detection Conv {conv['detect_index']} in layer 105")

            # Regular Conv handling (non-RepConv layers)
            else:
                # First try to match the exact node_type
                if node_type in fl_lookup[layer_num]:
                    fl_values = fl_lookup[layer_num][node_type]
                    
                    # Map input FL value if available
                    if 'input' in fl_values:
                        conv['input_fl'] = fl_values['input']
                    
                    # Map weight FL value if available
                    if 'weight' in fl_values:
                        conv['weight_fl'] = fl_values['weight']
                    
                    # Map bias FL value if available and node has bias
                    if 'bias' in fl_values and conv['has_bias']:
                        conv['bias_fl'] = fl_values['bias']
                    
                    # Map output FL value if available
                    if 'output' in fl_values:
                        conv['output_fl'] = fl_values['output']
                    
                    logger.info(f"Mapped FL values for {node_type} in layer {layer_num}: "
                               f"input={conv.get('input_fl')}, weight={conv.get('weight_fl')}, "
                               f"bias={conv.get('bias_fl')}, output={conv.get('output_fl')}")
                else:
                    # Fallback to the generic 'Conv' type if available
                    if 'Conv' in fl_lookup[layer_num]:
                        fl_values = fl_lookup[layer_num]['Conv']
                        
                        if 'input' in fl_values:
                            conv['input_fl'] = fl_values['input']
                        if 'weight' in fl_values:
                            conv['weight_fl'] = fl_values['weight']
                        if 'bias' in fl_values and conv['has_bias']:
                            conv['bias_fl'] = fl_values['bias']
                        if 'output' in fl_values:
                            conv['output_fl'] = fl_values['output']
                        
                        logger.info(f"Fallback: Mapped FL values for Conv in layer {layer_num}: "
                                   f"input={conv.get('input_fl')}, weight={conv.get('weight_fl')}, "
                                   f"bias={conv.get('bias_fl')}, output={conv.get('output_fl')}")
                    else:
                        logger.warning(f"No FL values found for {node_type} in layer {layer_num}")
            
            # Add to mapped list if any FL values were mapped
            if any(k in conv for k in ['input_fl', 'weight_fl', 'bias_fl', 'output_fl']):
                mapped_convs.append(conv)
    
    # 2. Map Concat nodes
    mapped_concats = []
    for concat in model_structure['concat_nodes']:
        layer_num = concat.get('layer_number')
        node_type = concat.get('node_type', 'Concat')
        
        if layer_num is not None and layer_num in fl_lookup:
            # Try to match the exact node_type first
            if node_type in fl_lookup[layer_num] and 'output' in fl_lookup[layer_num][node_type]:
                concat['output_fl'] = fl_lookup[layer_num][node_type]['output']
                logger.info(f"Mapped output FL value {concat['output_fl']} for {node_type} in layer {layer_num}")
            elif 'Concat' in fl_lookup[layer_num] and 'output' in fl_lookup[layer_num]['Concat']:
                # Fallback to generic Concat
                concat['output_fl'] = fl_lookup[layer_num]['Concat']['output']
                logger.info(f"Fallback: Mapped output FL value {concat['output_fl']} for Concat in layer {layer_num}")
            else:
                logger.warning(f"No output FL value found for {node_type} in layer {layer_num}")
            
            # Add to mapped list if FL value was mapped
            if 'output_fl' in concat:
                mapped_concats.append(concat)
    
    # 3. Map MaxPool nodes
    mapped_maxpools = []
    for maxpool in model_structure['MaxPool_nodes']:
        layer_num = maxpool.get('layer_number')
        node_type = maxpool.get('node_type', 'MaxPool')
        
        if layer_num is not None and layer_num in fl_lookup:
            # Try to match the exact node_type first
            if node_type in fl_lookup[layer_num] and 'output' in fl_lookup[layer_num][node_type]:
                maxpool['output_fl'] = fl_lookup[layer_num][node_type]['output']
                logger.info(f"Mapped output FL value {maxpool['output_fl']} for {node_type} in layer {layer_num}")
            elif 'MaxPool' in fl_lookup[layer_num] and 'output' in fl_lookup[layer_num]['MaxPool']:
                # Fallback to generic MaxPool
                maxpool['output_fl'] = fl_lookup[layer_num]['MaxPool']['output']
                logger.info(f"Fallback: Mapped output FL value {maxpool['output_fl']} for MaxPool in layer {layer_num}")
            else:
                logger.warning(f"No output FL value found for {node_type} in layer {layer_num}")
            
            # Add to mapped list if FL value was mapped
            if 'output_fl' in maxpool:
                mapped_maxpools.append(maxpool)
    
    # 4. Map special nodes
    
    # 4.1 Map LeakyReLU/Activation nodes
    for leaky in model_structure['special_nodes']['leakyrelu']:
        layer_num = leaky.get('layer_number')
        node_type = leaky.get('node_type', 'Activation_LeakyRelu')
        
        if layer_num is not None and layer_num in fl_lookup:
            # Try to match the exact node_type first
            if node_type in fl_lookup[layer_num]:
                # Look for 'act_output' first, then fallback to 'output'
                if 'act_output' in fl_lookup[layer_num][node_type]:
                    leaky['output_fl'] = fl_lookup[layer_num][node_type]['act_output']
                    logger.info(f"Mapped act_output FL value {leaky['output_fl']} for {node_type} in layer {layer_num}")
                elif 'output' in fl_lookup[layer_num][node_type]:
                    leaky['output_fl'] = fl_lookup[layer_num][node_type]['output']
                    logger.info(f"Mapped output FL value {leaky['output_fl']} for {node_type} in layer {layer_num}")
            # Try generic activation types
            elif 'Activation_LeakyRelu' in fl_lookup[layer_num]:
                if 'act_output' in fl_lookup[layer_num]['Activation_LeakyRelu']:
                    leaky['output_fl'] = fl_lookup[layer_num]['Activation_LeakyRelu']['act_output']
                elif 'output' in fl_lookup[layer_num]['Activation_LeakyRelu']:
                    leaky['output_fl'] = fl_lookup[layer_num]['Activation_LeakyRelu']['output']
            # Last resort - try to find any activation FL value for this layer
            else:
                for nt, roles in fl_lookup[layer_num].items():
                    if 'Activation' in nt and ('act_output' in roles or 'output' in roles):
                        leaky['output_fl'] = roles.get('act_output', roles.get('output'))
                        logger.info(f"Fallback: Mapped from {nt} FL value {leaky['output_fl']} for {node_type} in layer {layer_num}")
                        break
    
    # 4.2 Map RepConv Add nodes
    for add_node in model_structure['special_nodes']['repconv_add']:
        layer_num = add_node.get('layer_number')
        node_type = add_node.get('node_type', f'RepConv_Add_{layer_num}')
        if layer_num is not None and layer_num in fl_lookup:
            # Try to match the exact node_type first
            if node_type in fl_lookup[layer_num] and 'add' in fl_lookup[layer_num][node_type]:
                add_node['output_fl'] = fl_lookup[layer_num][node_type]['add']
                logger.info(f"Mapped add FL value {add_node['output_fl']} for {node_type} in layer {layer_num}")
    # 4.3 Map specific RepConv dense and 1x1 nodes to ensure they're marked for processing
    # This ensures we handle both main Conv nodes and special-case RepConv nodes
    for rep_dense in model_structure['special_nodes']['repconv_dense']:
        layer_num = rep_dense.get('layer_number')
        node_type = rep_dense.get('node_type', f"RepConv_Dense_{layer_num}")
        
        # Check if this node has already been mapped as a regular Conv
        already_mapped = False
        for conv in mapped_convs:
            if conv['node_idx'] == rep_dense['node_idx']:
                already_mapped = True
                break
                
        if not already_mapped and layer_num is not None and layer_num in fl_lookup:
            if node_type in fl_lookup[layer_num]:
                fl_values = fl_lookup[layer_num][node_type]
                
                if 'input' in fl_values:
                    rep_dense['input_fl'] = fl_values['input']
                if 'weight' in fl_values:
                    rep_dense['weight_fl'] = fl_values['weight']
                if 'bias' in fl_values and rep_dense['has_bias']:
                    rep_dense['bias_fl'] = fl_values['bias']
                if 'output' in fl_values:
                    rep_dense['output_fl'] = fl_values['output']
                
                if any(k in rep_dense for k in ['input_fl', 'weight_fl', 'bias_fl', 'output_fl']):
                    mapped_convs.append(rep_dense)
    
    for rep_1x1 in model_structure['special_nodes']['repconv_1x1']:
        layer_num = rep_1x1.get('layer_number')
        node_type = rep_1x1.get('node_type', f"RepConv_1x1_{layer_num}")
        
        # Check if this node has already been mapped as a regular Conv
        already_mapped = False
        for conv in mapped_convs:
            if conv['node_idx'] == rep_1x1['node_idx']:
                already_mapped = True
                break
                
        if not already_mapped and layer_num is not None and layer_num in fl_lookup:
            if node_type in fl_lookup[layer_num]:
                fl_values = fl_lookup[layer_num][node_type]
                
                if 'input' in fl_values:
                    rep_1x1['input_fl'] = fl_values['input']
                if 'weight' in fl_values:
                    rep_1x1['weight_fl'] = fl_values['weight']
                if 'bias' in fl_values and rep_1x1['has_bias']:
                    rep_1x1['bias_fl'] = fl_values['bias']
                if 'output' in fl_values:
                    rep_1x1['output_fl'] = fl_values['output']
                
                if any(k in rep_1x1 for k in ['input_fl', 'weight_fl', 'bias_fl', 'output_fl']):
                    mapped_convs.append(rep_1x1)
    
    # 4.4 Map Implicit nodes in detection layer
    for implicit in model_structure['special_nodes']['implicit_nodes']:
        layer_num = implicit.get('layer_number')
        comp = implicit.get('component')  # 'ia' or 'im'
        idx = implicit.get('index')
        node_type = implicit.get('node_type', f'Implicit_{comp}_{idx}')
        
        if layer_num is not None and layer_num in fl_lookup:
            # Try to match the exact node_type first
            if node_type in fl_lookup[layer_num]:
                # For weight
                if 'weight' in fl_lookup[layer_num][node_type]:
                    implicit['weight_fl'] = fl_lookup[layer_num][node_type]['weight']
                    logger.info(f"Mapped weight FL value {implicit['weight_fl']} for {node_type} in layer {layer_num}")
                
                # For output
                if 'output' in fl_lookup[layer_num][node_type]:
                    implicit['output_fl'] = fl_lookup[layer_num][node_type]['output']
                    logger.info(f"Mapped output FL value {implicit['output_fl']} for {node_type} in layer {layer_num}")
            
            # Try fallback to generic Implicit types with component info
            elif f'Implicit_{comp}' in fl_lookup[layer_num]:
                if 'weight' in fl_lookup[layer_num][f'Implicit_{comp}']:
                    implicit['weight_fl'] = fl_lookup[layer_num][f'Implicit_{comp}']['weight']
                if 'output' in fl_lookup[layer_num][f'Implicit_{comp}']:
                    implicit['output_fl'] = fl_lookup[layer_num][f'Implicit_{comp}']['output']
                logger.info(f"Fallback: Mapped from Implicit_{comp} in layer {layer_num}")
            
            # Last resort - check for any Implicit FL values
            else:
                for nt, roles in fl_lookup[layer_num].items():
                    if 'Implicit' in nt:
                        if 'weight' in roles and 'weight_fl' not in implicit:
                            implicit['weight_fl'] = roles['weight']
                        if 'output' in roles and 'output_fl' not in implicit:
                            implicit['output_fl'] = roles['output']
                        logger.info(f"Fallback: Mapped from {nt} for {node_type} in layer {layer_num}")
    
    # Log summary of mapping results
    logger.info(f"Mapped FL values for {len(mapped_convs)} Conv nodes")
    logger.info(f"Mapped FL values for {len(mapped_concats)} Concat nodes")
    logger.info(f"Mapped FL values for {len(mapped_maxpools)} MaxPool nodes")
    logger.info(f"Mapped FL values for {len([n for n in model_structure['special_nodes']['leakyrelu'] if 'output_fl' in n])} LeakyReLU nodes")
    logger.info(f"Mapped FL values for {len([n for n in model_structure['special_nodes']['repconv_add'] if 'output_fl' in n])} RepConv Add nodes")
    logger.info(f"Mapped FL values for {len([n for n in model_structure['special_nodes']['repconv_dense'] if any(k in n for k in ['input_fl', 'weight_fl', 'bias_fl', 'output_fl'])])} RepConv Dense nodes")
    logger.info(f"Mapped FL values for {len([n for n in model_structure['special_nodes']['repconv_1x1'] if any(k in n for k in ['input_fl', 'weight_fl', 'bias_fl', 'output_fl'])])} RepConv 1x1 nodes")
    logger.info(f"Mapped FL values for {len([n for n in model_structure['special_nodes']['implicit_nodes'] if 'weight_fl' in n or 'output_fl' in n])} Implicit nodes")
    
    # Check for proper mapping validation
    total_fl_mappings = sum(len(fl_lookup.get(layer, {}).get(node_type, {})) 
                           for layer in fl_lookup 
                           for node_type in fl_lookup.get(layer, {}))
    logger.info(f"Total available FL values from Excel: {total_fl_mappings}")
    
    # Return the mapping results
    return {
        'mapped_convs': mapped_convs,
        'mapped_concats': mapped_concats,
        'mapped_maxpools': mapped_maxpools,
        'special_nodes': model_structure['special_nodes']
    }

from collections import defaultdict ###
def add_quantization_nodes(model, mapped_nodes):
    """Add quantization nodes to the ONNX model with improved node type differentiation"""
    graph = model.graph
    
    # 用於追踪要插入的節點：key 是原節點的 index
    inserted_before = defaultdict(list) ###
    inserted_after = defaultdict(list)  ###
    # Keep track of used names and added nodes to avoid duplicates and track operation
    used_names = set()
    # new_nodes = []
    quantized_outputs = set()  # Track which outputs have already been quantized
    
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
        if fl_value < 0:
            logger.warning(f"Negative FL value {fl_value} detected, using absolute value instead")
            # fl_value = abs(fl_value)
        
        scale_value = float(np.power(2.0, -fl_value))
        scale = helper.make_tensor(scale_name, TensorProto.FLOAT, [], [scale_value])
        zero = helper.make_tensor(zero_name, zero_type, [], [0])
        return scale, zero
    
    # Helper function to add output quantization for a node
    def add_output_quantization(node, node_info, fl_value, node_unique_id):
        """Add output quantization for a node and update consumers"""
        output_tensor = node.output[0]
        
        # Skip if this output has already been quantized
        if output_tensor in quantized_outputs:
            logger.warning(f"Output {output_tensor} has already been quantized, skipping duplicate quantization")
            return []
        
        # Create unique names
        output_scale_name = get_unique_name(f"out_quant_scale_{node_unique_id}")
        output_zero_name = get_unique_name(f"out_zero_point_{node_unique_id}")
        quant_node_name = get_unique_name(f"out_quant_{node_unique_id}")
        dequant_node_name = get_unique_name(f"out_dequant_{node_unique_id}")
        quant_output = get_unique_name(f"{output_tensor}_quantized")
        dequant_output = get_unique_name(f"{output_tensor}_dequantized")
        
        # Create scale and zero point tensors
        scale, zero = create_scale_zero_tensors(fl_value, output_scale_name, output_zero_name)
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
        
        # Mark this output as quantized
        quantized_outputs.add(output_tensor)
        
        # Return the new nodes
        return [quant_node, dequant_node]
    
    # 1. Process Conv nodes
    for conv in mapped_nodes['mapped_convs']:
        node = conv['node']
        idx = conv['node_idx']  ###
        layer_num = conv.get('layer_number', 'unknown')
        node_type = conv.get('node_type', 'Conv')
        
        
        # Create a unique identifier for this node
        node_unique_id = f"{node_type}_{idx}_{layer_num}"  ###
        # node_unique_id = f"{node_type}_{conv['node_idx']}_{layer_num}"
        
        logger.info(f"Processing quantization for {node_type} in layer {layer_num} (node: {conv['node_name']})")
        
        # 1.1 Add input quantization if FL value exists
        if conv.get('input_fl') is not None:
            input_tensor = node.input[0]
            
            # Create unique names for tensors and nodes
            input_scale_name = get_unique_name(f"in_quant_scale_{node_unique_id}")
            input_zero_name = get_unique_name(f"in_zero_point_{node_unique_id}")
            quant_node_name = get_unique_name(f"in_quant_{node_unique_id}")
            dequant_node_name = get_unique_name(f"in_dequant_{node_unique_id}")
            quant_output = get_unique_name(f"{input_tensor}_quantized")
            dequant_output = get_unique_name(f"{input_tensor}_dequantized")
            
            # Create scale and zero point tensors
            scale, zero = create_scale_zero_tensors(conv['input_fl'], input_scale_name, input_zero_name)
            graph.initializer.extend([scale, zero])
            
            # Create quantize and dequantize nodes
            quant_node = helper.make_node(
                "QuantizeLinear",
                inputs=[input_tensor, input_scale_name, input_zero_name],
                outputs=[quant_output],
                name=quant_node_name
            )
            
            dequant_node = helper.make_node(
                "DequantizeLinear",
                inputs=[quant_output, input_scale_name, input_zero_name],
                outputs=[dequant_output],
                name=dequant_node_name
            )
            
            # Update the Conv node to use the dequantized input
            node.input[0] = dequant_output
            
            # new_nodes.extend([quant_node, dequant_node])
            inserted_before[idx].extend([quant_node, dequant_node]) ### 標記插入位置
            # logger.info(f"Added input quantization for {node_type} in layer {layer_num} with FL={conv['input_fl']}")
            logger.info(f"Added input quantization for {node_type} (before node {idx}) with FL={conv['input_fl']}")
            
        # 1.2 Add weight quantization if FL value exists
        if conv.get('weight_fl') is not None and conv.get('weight_name') is not None:
            weight_tensor = conv['weight_name']
            
            # Create unique names
            weight_scale_name = get_unique_name(f"weight_quant_scale_{node_unique_id}")
            weight_zero_name = get_unique_name(f"weight_zero_point_{node_unique_id}")
            dequant_node_name = get_unique_name(f"weight_dequant_{node_unique_id}")
            dequant_output = get_unique_name(f"{weight_tensor}_dequantized")
            
            # Create scale and zero point tensors
            scale, zero = create_scale_zero_tensors(conv['weight_fl'], weight_scale_name, weight_zero_name)
            graph.initializer.extend([scale, zero])
            
            # Create dequantize node
            dequant_node = helper.make_node(
                "DequantizeLinear",
                inputs=[weight_tensor, weight_scale_name, weight_zero_name],
                outputs=[dequant_output],
                name=dequant_node_name
            )
            
            # Update the Conv node to use the dequantized weight
            node.input[1] = dequant_output
            inserted_before[idx].append(dequant_node) ### 標記插入位置
            # new_nodes.append(dequant_node)
            logger.info(f"Added weight quantization for {node_type} in layer {layer_num} with FL={conv['weight_fl']}")
        
        # 1.3 Add bias quantization if FL value exists and bias exists
        if conv.get('bias_fl') is not None and conv.get('has_bias') and conv.get('bias_name') is not None:
            bias_tensor = conv['bias_name']
            
            # Create unique names
            bias_scale_name = get_unique_name(f"bias_quant_scale_{node_unique_id}")
            bias_zero_name = get_unique_name(f"bias_zero_point_{node_unique_id}")
            dequant_node_name = get_unique_name(f"bias_dequant_{node_unique_id}")
            dequant_output = get_unique_name(f"{bias_tensor}_dequantized")
            
            # Create scale and zero point tensors
            scale, zero = create_scale_zero_tensors(conv['bias_fl'], bias_scale_name, bias_zero_name, TensorProto.INT8)
            graph.initializer.extend([scale, zero])
            
            # Create dequantize node
            dequant_node = helper.make_node(
                "DequantizeLinear",
                inputs=[bias_tensor, bias_scale_name, bias_zero_name],
                outputs=[dequant_output],
                name=dequant_node_name
            )
            
            # Update the Conv node to use the dequantized bias
            node.input[2] = dequant_output
            inserted_before[idx].append(dequant_node)  ### 標記插入位置
            # new_nodes.append(dequant_node)
            logger.info(f"Added bias quantization for {node_type} in layer {layer_num} with FL={conv['bias_fl']}")
        
        # 1.4 Add output quantization if FL value exists
        if conv.get('output_fl') is not None:
            output_quant_nodes = add_output_quantization(node, conv, conv['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes) ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={conv['output_fl']}")
    
    # 2. Process Concat nodes
    for concat in mapped_nodes['mapped_concats']:
        node = concat['node']
        layer_num = concat.get('layer_number', 'unknown')
        node_type = concat.get('node_type', 'Concat')
        idx = concat['node_idx']
        
        if concat.get('output_fl') is not None:
            # Create unique identifier
            node_unique_id = f"{node_type}_{concat['node_idx']}_{layer_num}"
            
            output_quant_nodes = add_output_quantization(node, concat, concat['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes)  ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={concat['output_fl']}")
    
    # 3. Process MaxPool nodes
    for maxpool in mapped_nodes['mapped_maxpools']:
        node = maxpool['node']
        layer_num = maxpool.get('layer_number', 'unknown')
        node_type = maxpool.get('node_type', 'MaxPool')
        idx = maxpool['node_idx']
        
        if maxpool.get('output_fl') is not None:
            # Create unique identifier
            node_unique_id = f"{node_type}_{maxpool['node_idx']}_{layer_num}"
            
            output_quant_nodes = add_output_quantization(node, maxpool, maxpool['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes) ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={maxpool['output_fl']}")
    
    # 4. Process special nodes
    
    # 4.1 LeakyReLU nodes
    for leaky in mapped_nodes['special_nodes']['leakyrelu']:
        if leaky.get('output_fl') is not None:
            node = leaky['node']
            layer_num = leaky.get('layer_number', 'unknown')
            node_type = leaky.get('node_type', 'Activation_LeakyRelu')
            idx = leaky['node_idx']
            # Create unique identifier
            node_unique_id = f"{node_type}_{leaky['node_idx']}_{layer_num}"
            
            output_quant_nodes = add_output_quantization(node, leaky, leaky['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes) ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={leaky['output_fl']}")
    
    # 4.2 SPPCSPC activation nodes
    for act in mapped_nodes['special_nodes']['sppcspc_act']:
        if act.get('output_fl') is not None:
            node = act['node']
            layer_num = act.get('layer_number', 'unknown')
            sub_module = act.get('sub_module', 'unknown')
            node_type = act.get('node_type', f'SPPCSPC_Activation_{sub_module}')
            idx = act['node_idx']
            # Create unique identifier
            node_unique_id = f"{node_type}_{act['node_idx']}_{layer_num}"
            
            output_quant_nodes = add_output_quantization(node, act, act['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes) ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={act['output_fl']}")
    
    # 4.3 SPPCSPC MaxPool nodes
    for maxpool in mapped_nodes['special_nodes']['sppcspc_maxpool']:
        if maxpool.get('output_fl') is not None:
            node = maxpool['node']
            layer_num = maxpool.get('layer_number', 'unknown')
            pool_index = maxpool.get('pool_index', 'unknown')
            node_type = maxpool.get('node_type', f'SPPCSPC_MaxPool_{pool_index}')
            idx = maxpool['node_idx']
            # Create unique identifier
            node_unique_id = f"{node_type}_{maxpool['node_idx']}_{layer_num}"
            
            output_quant_nodes = add_output_quantization(node, maxpool, maxpool['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes) ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={maxpool['output_fl']}")
    
    # 4.4 RepConv Add nodes
    for add_node in mapped_nodes['special_nodes']['repconv_add']:
        if add_node.get('output_fl') is not None:
            node = add_node['node']
            layer_num = add_node.get('layer_number', 'unknown')
            node_type = add_node.get('node_type', f'RepConv_Add_{layer_num}')
            idx = add_node['node_idx']
            # Create unique identifier
            node_unique_id = f"{node_type}_{add_node['node_idx']}_{layer_num}"
            
            # Get both inputs to the Add operation to verify they're properly quantized
            input_0 = node.input[0] if len(node.input) > 0 else None
            input_1 = node.input[1] if len(node.input) > 1 else None
            
            # Log input information for debugging
            logger.info(f"RepConv Add node {node_type} inputs: {input_0}, {input_1}")
            
            # Now proceed with output quantization as before
            output_quant_nodes = add_output_quantization(node, add_node, add_node['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes) ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={add_node['output_fl']}")
    
    # 4.5 Implicit nodes in detection layer
    for implicit in mapped_nodes['special_nodes']['implicit_nodes']:
        node = implicit['node']
        layer_num = implicit.get('layer_number', 'unknown')
        comp = implicit.get('component', '')  # 'ia' or 'im'
        idx = implicit.get('index', '')
        node_type = implicit.get('node_type', f'Implicit_{comp}_{idx}')
        
        # Create unique identifier
        node_unique_id = f"{node_type}_{implicit['node_idx']}_{layer_num}"
        
        # Add weight quantization if FL value exists
        if implicit.get('weight_fl') is not None and len(node.input) > 0:
            weight_tensor = node.input[0]
            
            # Create unique names
            weight_scale_name = get_unique_name(f"implicit_weight_scale_{node_unique_id}")
            weight_zero_name = get_unique_name(f"implicit_weight_zero_{node_unique_id}")
            dequant_name = get_unique_name(f"implicit_weight_dequant_{node_unique_id}")
            dequant_output = get_unique_name(f"{weight_tensor}_dequantized")
            
            # Create scale and zero point tensors
            scale, zero = create_scale_zero_tensors(implicit['weight_fl'], weight_scale_name, weight_zero_name)
            graph.initializer.extend([scale, zero])
            
            # Create dequantize node
            dequant_node = helper.make_node(
                "DequantizeLinear",
                inputs=[weight_tensor, weight_scale_name, weight_zero_name],
                outputs=[dequant_output],
                name=dequant_name
            )
            
            # Update the node input to use the dequantized weight
            node.input[0] = dequant_output
            inserted_before[idx].append(dequant_node) ### 標記插入位置
            # new_nodes.append(dequant_node)
            logger.info(f"Added weight quantization for {node_type} in layer {layer_num} with FL={implicit['weight_fl']}")
        
        # Add output quantization if FL value exists
        if implicit.get('output_fl') is not None:
            output_quant_nodes = add_output_quantization(node, implicit, implicit['output_fl'], node_unique_id)
            # new_nodes.extend(output_quant_nodes)
            if output_quant_nodes:
                inserted_after[idx].extend(output_quant_nodes) ### 標記插入位置
                logger.info(f"Added output quantization for {node_type} in layer {layer_num} with FL={implicit['output_fl']}")
    
    # # Add all new nodes to the graph
    # graph.node.extend(new_nodes)
    # logger.info(f"Added {len(new_nodes)} new quantization nodes to the model")
    # logger.info(f"Quantized {len(quantized_outputs)} unique output tensors")
    # --- 重組 Graph (Rebuild the graph node list) ---
    final_nodes = []
    # 遍歷原始圖中的每一個節點
    for i, node in enumerate(graph.node):
        # 1. 先放入需要在該節點"之前"插入的新節點 (Input/Weight Q/DQ)
        if i in inserted_before:
            final_nodes.extend(inserted_before[i])
        
        # 2. 放入原始節點
        final_nodes.append(node)
        
        # 3. 放入需要在該節點"之後"插入的新節點 (Output Q/DQ)
        if i in inserted_after:
            final_nodes.extend(inserted_after[i])
            
    # 清空並填入排序好的節點
    del graph.node[:]
    graph.node.extend(final_nodes)
    
    logger.info(f"Rebuilt graph with {len(final_nodes)} nodes (Topologically sorted)")
    return model

def main():
    args = parse_arguments()

    # Override module-level constants if CLI args provided
    global SPPSCSPC_LR_LAYER, DETECTION_LAYER, REPCONV_LAYERS
    if args.sppcspc_layer is not None:
        SPPSCSPC_LR_LAYER = args.sppcspc_layer
        logger.info(f"SPPCSPC layer overridden to {SPPSCSPC_LR_LAYER}")
    if args.detection_layer is not None:
        DETECTION_LAYER = args.detection_layer
        logger.info(f"Detection layer overridden to {DETECTION_LAYER}")
    if args.repconv_layers:
        REPCONV_LAYERS = [int(x.strip()) for x in args.repconv_layers.split(",") if x.strip()]
        logger.info(f"RepConv layers overridden to {REPCONV_LAYERS}")

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
    
    # Analyze model structure
    logger.info("Analyzing model structure...")
    model_structure = analyze_model_structure(model)
    
    # Map FL values to nodes
    logger.info("Mapping FL values to nodes...")
    mapped_nodes = map_fl_values_to_nodes(model_structure, fl_df)
    
    # Add quantization nodes
    logger.info("Adding quantization nodes...")
    quantized_model = add_quantization_nodes(model, mapped_nodes)
    
    # Save the quantized model
    logger.info(f"Saving quantized model to: {args.output_path}")
    onnx.save(quantized_model, args.output_path)
    logger.info("Quantization completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit(main())