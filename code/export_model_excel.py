#!/usr/bin/env python3
"""
Fixed FL Value Extractor

This script extracts FL (Fractional Length) values from a YOLO model dump file
and exports them to Excel format with layer name, quantizer type, and FL value.
Fixed to correctly extract layer numbers from module paths.

Usage:
    python fixed_fl_extractor.py <paste_txt_file> [output_excel_file]
"""

import re
import os
import sys
import pandas as pd
from collections import defaultdict
import logging



# Defaults (v7 p70) — overridden at runtime by CLI args from the agent
CONCAT_LAYER = [10, 16, 23, 29, 36, 42, 49, 55, 62, 67, 74, 80, 87, 93, 100]
MAXPOOL_LAYER = [12, 25, 38, 76, 89]
UPSAMPLE_LAYER = [53, 65]


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_fl_values(paste_file):
    """
    Extract FL values from the model dump file with enhanced node type classification
    Returns a DataFrame with layer name, quantizer type, node type, and FL value
    """
    logger.info(f"Extracting FL values from: {paste_file}")
    
    # Initialize data structure
    fl_data = []
    
    # Regular expressions for extracting information
    layer_pattern = r'\(([^)]+)\):\s*(\w+)'
    quantizer_pattern = r'(\w+Quantizer)\(bw\s*=\s*(\d+)\s*,\s*fl\s*=\s*(-?\d+)'
    component_pattern = r'\(([^:]+)\):\s*(\w+)'
    
    # Read the file
    with open(paste_file, 'r') as f:
        content = f.read()
    
    # Process line by line to maintain context
    lines = content.split('\n')
    
    # Context tracking variables
    current_layer_num = None
    current_layer_type = None
    current_component = None
    current_node_type = None
    component_stack = []
    indent_stack = []
    node_type_stack = []
    in_idetect = False
    
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        # Calculate indentation level
        indent = len(line) - len(line.lstrip())
        
        # Track IDetect layer
        if "IDetect" in line:
            in_idetect = True
            idetect_match = re.search(r'model\.(\d+)', line)
            if idetect_match:
                idetect_layer_num = int(idetect_match.group(1))
        
        # Check for layer declarations
        layer_match = re.search(layer_pattern, line)
        if layer_match:
            current_layer_id = layer_match.group(1)
            current_layer_type = layer_match.group(2)
            
            # Default node type is the layer type
            current_node_type = current_layer_type
            
            # Try to extract layer number from module path
            path_match = re.search(r'module\.model\.(\d+)', line)
            if path_match:
                current_layer_num = int(path_match.group(1))
            else:
                # Try direct number extraction
                layer_num_match = re.search(r'(\d+)', current_layer_id)
                if layer_num_match:
                    current_layer_num = int(layer_num_match.group(1))
                
            current_component = current_layer_id
            
            # Update node type stack
            while node_type_stack and indent_stack and indent <= indent_stack[-1]:
                node_type_stack.pop()
                
            node_type_stack.append(current_node_type)
        
        # Track components and maintain stack based on indentation
        component_match = re.search(component_pattern, line)
        if component_match:
            component_name = component_match.group(1)
            component_type = component_match.group(2)
            
            # Update the node type
            current_node_type = component_type
            
            # Update the component stack based on indentation
            while indent_stack and indent_stack[-1] >= indent:
                indent_stack.pop()
                if component_stack:
                    component_stack.pop()
                if node_type_stack:
                    node_type_stack.pop()
            
            indent_stack.append(indent)
            component_stack.append(component_name)
            node_type_stack.append(current_node_type)
            current_component = component_name
        
        # Extract quantizer information
        quantizer_matches = re.finditer(quantizer_pattern, line)
        for match in quantizer_matches:
            quantizer_type = match.group(1)
            bw = int(match.group(2))
            fl = int(match.group(3))
            
            # Determine quantizer role
            role = None
            if 'input_quantizer' in line:
                role = 'input'
            elif 'output_quantizer' in line:
                role = 'output'
            elif 'weight_quantizer' in line:
                role = 'weight'
            elif 'bias_quantizer' in line:
                role = 'bias'
            elif 'add_quantizer' in line:
                role = 'add'  # Special for RepConv add operation
            
            # Build the full path for the component
            if component_stack:
                full_path = ".".join(component_stack)
            else:
                full_path = current_component if current_component else ""
                
            # Extract layer number from full path if possible
            layer_num = current_layer_num
            if 'module.model.' in full_path:
                path_match = re.search(r'module\.model\.(\d+)', full_path)
                if path_match:
                    layer_num = int(path_match.group(1))
            
            # Determine node type more specifically
            node_type = "Unknown"
            node_role = "Unknown"
            print(f"full_path: {full_path}, current_layer_num: {current_layer_num}, current_component: {current_component}")
            # Check for specific node types in the path
            if "conv" in full_path.lower():
                node_type = "Conv"
                node_role = "Conv"
            elif "act" in full_path.lower() or "LeakyReLU" in full_path:
                node_type = "LeakyReLU"
                node_role = "Act"
            elif "bn" in full_path.lower() or "batchnorm" in full_path.lower():
                node_type = "BatchNorm"
                node_role = "BN"
            elif "51.m" in full_path:
                node_type = "MaxPool"
                node_role = "MaxPool"
                print("SPPCSPC MaxPool layer found")
            elif "105.m" in full_path:
                node_type = "Conv"
                node_role = "Detect"
                print("IDetect layer found")
            elif "maxpool" in full_path.lower() or "MP" in full_path:
                node_type = "MaxPool"
                node_role = "MaxPool"
            elif "concat" in full_path.lower():
                node_type = "Concat"
                node_role = "Concat"
            elif "upsample" in full_path.lower():
                node_type = "Upsample"
                node_role = "Upsample"
            elif "rbr_dense" in full_path.lower():
                node_type = "Conv"
                node_role = "rbr_dense"
            elif "rbr_1x1" in full_path.lower():
                node_type = "Conv"
                node_role = "rbr_1x1"
            elif "ImplicitA" in full_path:
                node_type = "ImplicitA"
                node_role = "ImplicitA"
            elif "ImplicitM" in full_path:
                node_type = "ImplicitM"
                node_role = "ImplicitM"
            elif "add" in full_path.lower() or "Add" in full_path:
                node_type = "Add"
                node_role = "Add"
            elif layer_num in CONCAT_LAYER:
                node_type = "Concat"
                node_role = "Concat"
            elif layer_num in MAXPOOL_LAYER:
                node_type = "MaxPool"
                node_role = "MaxPool"
            elif layer_num in UPSAMPLE_LAYER:
                node_type = "Upsample"
                node_role = "Upsample"

            
            # Handle special case for Implicit nodes in IDetect
            if in_idetect and (".ia" in full_path or ".im" in full_path):
                if ".ia" in full_path:
                    node_type = "ImplicitA"
                    node_role = "ImplicitA"
                    print("ImplicitA layer found")
                elif ".im" in full_path:
                    node_type = "ImplicitM"
                    node_role = "ImplicitM"
                    print("ImplicitM layer found")
            
            # Determine layer name with node type
            if layer_num is not None:
                layer_name = f"{layer_num}:{node_type}"
                if current_component and current_component != str(layer_num):
                    sub_component = current_component.split('.')[-1] if '.' in current_component else current_component
                    if not sub_component.isdigit():
                        layer_name += f".{sub_component}"
            else:
                layer_name = f"{node_type}"
                if current_component:
                    layer_name = f"{current_component}:{node_type}"
            
            # Add the data
            fl_data.append({
                'layer_number': layer_num,
                'layer_name': layer_name,
                'component': current_component,
                'node_type': node_type,
                'node_role': node_role,
                'full_path': full_path,
                'quantizer_type': quantizer_type,
                'quantizer_role': role,
                'bw': bw,
                'fl': fl
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(fl_data)
    
    # Clean up and organize the data
    if not df.empty:
        # Sort by layer number
        if 'layer_number' in df.columns:
            df = df.sort_values('layer_number')
    
    logger.info(f"Extracted {len(df)} FL values from {df['layer_name'].nunique()} layers")
    return df
def export_to_excel(df, output_file):
    """
    Export the extracted data to Excel format
    """
    print(f"Exporting to {output_file}...")
    
    # Create sheets for different views
    with pd.ExcelWriter(output_file) as writer:
        # Main sheet with all data
        df.to_excel(writer, sheet_name='FL_Values', index=False)
        
        # Create a pivot view for easier reference
        pivot_df = df.pivot_table(
            values='fl',
            index=['layer_number', 'layer_name'],
            columns=['quantizer_role'],
            aggfunc='first'
        ).reset_index()
        pivot_df.to_excel(writer, sheet_name='FL_Pivot', index=False)
        
        # Summary statistics
        summary = df.groupby(['quantizer_type', 'quantizer_role'])['fl'].agg(['min', 'max', 'mean', 'count']).reset_index()
        summary.to_excel(writer, sheet_name='FL_Summary', index=False)
    
    print(f"Export complete: {output_file}")

def _parse_int_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract FL values from model dump to Excel")
    parser.add_argument("paste_file", help="Path to model.txt")
    parser.add_argument("output_file", nargs="?", help="Output Excel path")
    parser.add_argument("--concat-layers",  default="", help="Comma-separated Concat layer indices")
    parser.add_argument("--maxpool-layers", default="", help="Comma-separated MaxPool layer indices")
    parser.add_argument("--upsample-layers", default="", help="Comma-separated Upsample layer indices")
    args = parser.parse_args()

    # Override module-level constants if args provided
    global CONCAT_LAYER, MAXPOOL_LAYER, UPSAMPLE_LAYER
    if args.concat_layers:
        CONCAT_LAYER = _parse_int_list(args.concat_layers)
    if args.maxpool_layers:
        MAXPOOL_LAYER = _parse_int_list(args.maxpool_layers)
    if args.upsample_layers:
        UPSAMPLE_LAYER = _parse_int_list(args.upsample_layers)

    paste_file = args.paste_file
    output_file = args.output_file or (os.path.splitext(paste_file)[0] + "_fl_values_fixed.xlsx")

    # Extract FL values
    df = extract_fl_values(paste_file)
    
    # Export to Excel
    if not df.empty:
        export_to_excel(df, output_file)
        
        # Print summary statistics
        print("\nFL Value Statistics:")
        print(f"Min FL: {df['fl'].min()}")
        print(f"Max FL: {df['fl'].max()}")
        print(f"Most common FL: {df['fl'].value_counts().index[0]} (occurs {df['fl'].value_counts().values[0]} times)")
        
        # Print some sample data
        print("\nSample of extracted data:")
        sample_cols = ['layer_number', 'layer_name', 'quantizer_role', 'quantizer_type', 'fl']
        print(df[sample_cols].head(10))
    else:
        print("No FL values extracted")

if __name__ == "__main__":
    main()