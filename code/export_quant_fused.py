"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
from functools import reduce
import sys
import time
from typing import Union

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
# from utils.activations import Hardswish, SiLU
# from utils.general import set_logging, check_img_size
import onnx, onnxsim
from onnx import helper

def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    layer = module
    for attr in names:
        if hasattr(layer, attr):
            layer = getattr(layer, attr)
    return layer
    # return reduce(getattr, names, module)


def load_model(weights, device, opt_int8 = False, opt_dir = None):
    if len(weights) == 1 and weights[0].endswith(".ckpt"):
        from aimet_torch.quantsim import load_checkpoint
        sim = load_checkpoint(weights[0])
        print(sim)
        return sim.model

    ckpt = torch.load(weights[0] if isinstance(weights, list) else weights, map_location='cpu', weights_only=False)
    model = ckpt['model'].to(device).float().eval()

    if opt_dir:
        with open(opt_dir / "opt.txt", "wt") as f:
            f.write("python " + " ".join(sys.argv)+"\n")
            
    if model.__class__.__name__ == "QuantModel":

        print("Disable Act: ", model.act_disabled)
        print("Quantizing weights...")
        model.fuse_batchnorm()
        if(opt_int8):
            model.quantize_weight_export_int8()
        else:
            model.quantize_weight_before_forward()
        if opt_dir:
            with open(opt_dir / "opt.txt", "at") as f:
                f.write(model.wrapper_info() + "\n")
                f.write(str(model)+"\n")

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/qv6_Hand0113fmask_a8w4b16_fused_DFP_frzFL3_frzBN_lr4_clip10_QConcat_DFPout_EIoU_AabsWstd_/weights/best.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img_size', nargs='+', type=int, default=[192, 320], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--onnx2pb', action='store_true', default=False, help='export onnx to pb')
    parser.add_argument('--onnx_infer', action='store_true', default=True, help='onnx infer test')
    #=======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', default=False, help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', default=False, help='fp16 infer')
    #================================================================
    parser.add_argument('--ezquant_int8', action='store_true', default=False,help='ezquant INT8 export')
    
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    # set_logging()
    t = time.time()

    # Load PyTorch model
    # model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    model = load_model(opt.weights, torch.device('cpu'), opt_int8 = opt.ezquant_int8)
    model.module.model[-1].export = True
    
    model.unwrap()
    model = model.module
    # delattr(model.model[-1], 'anchor_grid')
    # model.model[-1].anchor_grid=[torch.zeros(1)] * 3 # nl=3 number of detection layers
    # model.model[-1].export_cat = True
    # model.eval()
    # labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    # opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.randn(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    # for k, m in model.named_modules():
    #     m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    #     if isinstance(m, models.common.Conv):  # assign export-friendly activations
    #         if isinstance(m.act, nn.Hardswish):
    #             m.act = Hardswish()
    #         elif isinstance(m.act, nn.SiLU):
    #             m.act = SiLU()
    #     # elif isinstance(m, models.yolo.Detect):
    #     #     m.forward = m.forward_export  # assign forward (optional)
    #     if isinstance(m, models.common.ShuffleV2Block):#shufflenet block nn.SiLU
    #         for i in range(len(m.branch1)):
    #             if isinstance(m.branch1[i], nn.SiLU):
    #                 m.branch1[i] = SiLU()
    #         for i in range(len(m.branch2)):
    #             if isinstance(m.branch2[i], nn.SiLU):
    #                 m.branch2[i] = SiLU()
    y = model(img)  # dry run
    
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)
        
    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    # model.fuse()  # only for ONNX
    img = img - 0.5
    input_names=['input']
    output_names=['detect1', 'detect2', 'detect3']
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('ONNX export success, saved as %s' % f)

    # ONNX simplify
    print('\nStarting ONNX simplify with onnxsim ...')

    onnx_model_opt, _ = onnxsim.simplify(onnx_model, skip_fuse_bn=True)

    q_model = load_model(opt.weights, torch.device('cpu'))
    q_model.module.model[-1].export = True
    

    for node in onnx_model_opt.graph.node:
        print(node.name)
        try:
            layer_name = ".".join(["module"] + node.name.split("/")[1:-1])
            layer = get_module_by_name(q_model, layer_name)
            node.doc_string = repr(layer)
            # for quantizer_name in ["input_quantizer", "output_quantizer", "weight_quantizer", "bias_quantizer"]:
            #     if hasattr(layer, quantizer_name):
            #         node.attribute.append(helper.make_attribute(quantizer_name, repr(getattr(layer, quantizer_name))))
        except AttributeError as e:
            # print(e)
            pass

    onnx.checker.check_model(onnx_model_opt)  # check onnx model

    sf = f.replace(".onnx", "_sim_annotate.onnx")

    onnx.save(onnx_model_opt, sf)

    print('Simplified ONNX export success, saved as %s' % sf)
    
    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


    # onnx infer
    if opt.onnx_infer:
        import onnxruntime
        import numpy as np
        providers =  ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(sf, providers=providers)
        im = img.cpu().numpy().astype(np.float32) # torch to numpy
        y_onnx = session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: im})
        for y_onnx_out, y_out in zip(y_onnx, y):
            y_out = y_out.detach().cpu()
            print("="*30)
            print("onnx pred's shape is ", y_onnx_out.shape)
            print("orig pred's shape is ", y_out.shape)
            print("max(|torch_pred - onnx_pred|) =",abs(y_out.cpu().numpy()-y_onnx_out).max())


    # TensorRT export
    if opt.onnx2trt:
        from torch2trt.trt_model import ONNX_to_TRT
        print('\nStarting TensorRT...')
        ONNX_to_TRT(onnx_model_path=f,trt_engine_path=f.replace('.onnx', '.trt'),fp16_mode=opt.fp16_trt)

    # PB export
    if opt.onnx2pb:
        print('download the newest onnx_tf by https://github.com/onnx/onnx-tensorflow/tree/master/onnx_tf')
        from onnx_tf.backend import prepare
        import tensorflow as tf

        outpb = f.replace('.onnx', '.pb')  # filename
        # strict=True maybe leads to KeyError: 'pyfunc_0', check: https://github.com/onnx/onnx-tensorflow/issues/167
        tf_rep = prepare(onnx_model, strict=False)  # prepare tf representation
        tf_rep.export_graph(outpb)  # export the model

        out_onnx = tf_rep.run(img) # onnx output

        # check pb
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(outpb, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                input_x = sess.graph.get_tensor_by_name(input_names[0]+':0')  # input
                outputs = []
                for i in output_names:
                    outputs.append(sess.graph.get_tensor_by_name(i+':0'))
                out_pb = sess.run(outputs, feed_dict={input_x: img})

        print(f'out_pytorch {y}')
        print(f'out_onnx {out_onnx}')
        print(f'out_pb {out_pb}')
