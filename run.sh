# cd ../
python export_model_excel.py model.txt
python export_quant_fused.py  --weights ./runs/train/20260112_yolov7_640_a8w8_p70/weights/best.pt --img 640 --ezquant_int8 
# cd ./runs/train/yolov7_640_a8w8_actDFP_weiDFP_fuse_frzfl10_p70_sds/weights
cp ./runs/train/20260112_yolov7_640_a8w8_p70/weights/best_sim_annotate.onnx ./wrap_all_model_onnx_export/wrap_all_best_temp.onnx
python export_quant_fused.py  --weights ./runs/train/20260112_yolov7_640_a8w8_p70/weights/best.pt --img 640 --onnx_infer
cd ./wrap_all_model_onnx_export
python model_fp32_int8.py --input ./wrap_all_best_temp.onnx --output ./int8_converted_model.onnx
python onnx_view_wrap_all_repconv_topo.py --model_path ./int8_converted_model.onnx --output_path ./wrap_all_temp.onnx --quant_info ../model_fl_values_fixed.xlsx
python implicit_topo.py --model_path ./wrap_all_temp.onnx --output_path ./output_topo.onnx --quant_info ../model_fl_values_fixed.xlsx --detect_layer 105
python modify_model_topo.py
# python onnx_compare.py

