import copy
from functools import partial
import torch
from torch import nn, nan, Tensor
import torch.nn.functional as F


# from models.common import Concat
# from utils.layers import FeatureConcat, Concat, Shortcut
from models.common import Concat
INOUT_QUANTIZED_LAYERS = ()
OUTPUT_QUANTIZED_LAYERS = (Concat)
# OUTPUT_CONV_LAYERS = ["/model.4/conv/Conv","/model.5/conv/Conv","/model.7/conv/Conv","/model.9/conv/Conv", #concat 10
#                       "/model.13/conv/Conv","/model.15/conv/Conv", # concat 16
#                       "/model.17/conv/Conv","/model.18/conv/Conv","/model.20/conv/Conv","/model.22/conv/Conv", # concat 23
#                       "/model.26/conv/Conv","/model.28/conv/Conv", # concat 29
#                       "/model.30/conv/Conv","/model.31/conv/Conv","/model.33/conv/Conv","/model.35/conv/Conv", # concat 36
#                       "/model.39/conv/Conv","/model.41/conv/Conv", # concat 42
#                       "/model.43/conv/Conv","/model.44/conv/Conv","/model.46/conv/Conv","/model.48/conv/Conv", # concat 49
#                       "/model.0/conv/Conv"] 
ACTIVATION_LAYERS = (nn.LeakyReLU, nn.Mish, nn.ReLU, nn.ReLU6)
FL_THRESHOLD = 0.7
INPUT_BIAS = -0.5


QUANT_IMPLICIT= True
QUANT_REPCONV = True
# ----------------------
HEAD_STD_NUM = 1
MID_STD_NUM = 3
# ----------------------

print("77777 Using QuantModel v7 77777")
print("FL_THRESHOLD", FL_THRESHOLD)
print("INPUT_BIAS", INPUT_BIAS)



########### helper functions ###########

def replace_layer(parent_layer, name, new_layer):
    "Replace layer name in parent_layer with new_layer"
    for l in name.split(".")[:-1]:
        parent_layer = getattr(parent_layer, l)
    setattr(parent_layer, name.split(".")[-1], new_layer)

def BN_fuse(paras_dict, eps = 1e-3):
    def reshape_to_bias(input):
        return input.reshape(-1)
    def reshape_to_weight(input):
        return input.reshape(-1, 1, 1, 1)
    # BN融合
    scale = paras_dict["gamma"] / torch.sqrt(paras_dict["running_var"] + eps)
    if  paras_dict["bias"] is not None:
        bias = reshape_to_bias(paras_dict["beta"] + (paras_dict["bias"] - paras_dict["running_mean"]) * scale)
    else:
        bias = reshape_to_bias(
            paras_dict["beta"] - paras_dict["running_mean"] * scale)  # b融running
    weight = paras_dict["weight"] * reshape_to_weight(scale)  # w融running
    return weight, bias

def instantiate_helper(class_type, *paras, **key_paras):
    "Helper function for instantiate class object, return None if class_type is None"
    return class_type(*paras, **key_paras) if class_type else None

def MSE(x, y):
    return ((x-y)**2).mean()

def calculate_FL(bw, abs_max):
    return bw - torch.log2(abs_max) - 1

def calculate_abs_FL(bw, x):
    abs_max = torch.max(torch.abs(x))
    new_fl = calculate_FL(bw, abs_max)
    return abs_max, new_fl

def calculate_std_FL(bw, x):
    # Modified by Ding
    #=============================================================
    # abs_max = torch.sqrt(x.pow(2).mean())*3
    abs_max = torch.sqrt(((x - x.mean()).pow(2).mean())) * 3 + x.mean()
    #=============================================================
    new_fl = calculate_FL(bw, abs_max)
    return abs_max, new_fl

def max_represent_value_fl(bw, fl):
    return pow(2.0, (bw - 1.0 - fl))

def analyze_min_mse_fl(x, bw):
    min_err = 1e8
    start_fl = torch.floor(torch.log2(x.abs().max()))
    q_x = DFPQuantizeFunction.forward(None, x, bw, start_fl)
    mse = MSE(x, q_x)
    while mse < min_err + 1e-8:
        min_err = mse
        start_fl += 1
        q_x = DFPQuantizeFunction.forward(None, x, bw, start_fl)
        mse = MSE(x, q_x)
    return start_fl - 1

#############################################

class DFPQuantizeFunction():
    '''
    Quantize the input activations to 8-bit
    # '''
    def apply(input, bw, fl):

        s = pow(2, -fl.to(input.dtype))
        q_max = s * (pow(2, bw-1)-1)
        q_min = -s * (pow(2, bw-1))
        x_forward = s * (torch.floor(input/s + 0.5)) # round( input/s + 0.5 )
        x_forward = torch.clamp(x_forward, min=q_min, max=q_max)
        x_backward = torch.clamp(input, min=q_min, max=q_max)
        return x_forward.detach() + x_backward - x_backward.detach()
    # def apply(input, bw, fl):
    #     s = pow(2, -fl.to(input.dtype))
    #     q_max = s * (pow(2, bw-1)-1)
    #     q_min = -s * (pow(2, bw-1))
        
    #     # Scale the input value
    #     scaled = input / s
        
    #     # Standard rounding implementation
    #     # For values not exactly at x.5, this is standard rounding
    #     rounded = torch.floor(scaled + 0.5)
        
    #     # Special handling for values that are exactly x.5
    #     # Calculate the fractional part, handling both positive and negative values
    #     abs_scaled = torch.abs(scaled)
    #     frac_part = abs_scaled - torch.floor(abs_scaled)
        
    #     # Identify values that have exactly 0.5 as fractional part
    #     is_half = torch.isclose(frac_part, torch.tensor(0.5, device=input.device, dtype=input.dtype))
        
    #     if torch.any(is_half):
    #         # Get the integer part (floor of the absolute value, keeping the sign)
    #         int_part = torch.floor(abs_scaled) * torch.sign(scaled)
    #         is_even = (torch.floor(abs_scaled).long() % 2 == 0)
    #         adjusted = scaled.clone()
            
    #         # For positive x.5 where x is even, round down to x
    #         # For negative x.5 where x is even, round up to x (toward zero)
    #         mask_even_half = is_half & is_even
    #         adjusted[mask_even_half] = int_part[mask_even_half]
            
    #         # For positive x.5 where x is odd, round up to x+1
    #         # For negative x.5 where x is odd, round down to x-1 (away from zero)
    #         mask_odd_half = is_half & (~is_even)
    #         adjusted[mask_odd_half & (scaled > 0)] = int_part[mask_odd_half & (scaled > 0)] + 1
    #         adjusted[mask_odd_half & (scaled < 0)] = int_part[mask_odd_half & (scaled < 0)] - 1
            
    #         # Replace the standard rounding with banker's rounding for x.5 cases
    #         rounded = torch.where(is_half, adjusted, rounded)
        
    #     # Apply scaling and clamping
    #     x_forward = s * rounded
    #     x_forward = torch.clamp(x_forward, min=q_min, max=q_max)
    #     x_backward = torch.clamp(input, min=q_min, max=q_max)
        
    #     return x_forward.detach() + x_backward - x_backward.detach()

    # def apply_int(input, bw, fl):
    #     s = pow(2, -fl.to(input.dtype))
    #     scale = pow(2, -fl)
    #     # q_max = s * (pow(2, bw-1)-1)
    #     q_max_int8 = (pow(2, bw-1)-1)
    #     # q_min = -s * (pow(2, bw-1))
    #     q_min_int8 = -(pow(2, bw-1))
    #     # x_forward = s * (torch.floor(input/s + 0.5)) # round( input/s + 0.5 )
    #     x_forward_int8 = torch.clamp(torch.floor(input / s + 0.5), min=-128, max=127).to(torch.int8)
    #     # q_input = torch.quantize_per_tensor(input, scale=s, zero_point=s, dtype=torch.qint8)
    #     # print(q_input)
    #     # x_forward = q_input.dequantize()
        
    #     # ==========================================
    #     # print(x_forward_int8)
    #     # ==========================================
    #     # x_forward_int8 = torch.clamp(x_forward_int8, min=q_min_int8, max=q_max_int8)
    #     # x_forward = torch.clamp(x_forward, min=q_min_int8, max=q_max_int8)
    #     # x_backward = torch.clamp(input, min=q_min_int8, max=q_max_int8)

    #     return x_forward_int8.detach()
    #     # return x_forward.detach() + x_backward - x_backward.detach()

    def apply_int(input, bw, fl):
        s = pow(2, -fl.to(input.dtype))
        scale = pow(2, -fl)
        q_max_int8 = (pow(2, bw-1)-1)
        q_min_int8 = -(pow(2, bw-1))
        
        # 手動實現銀行家四捨五入
        scaled = input / s
        
        # 獲取整數和小數部分
        int_part = torch.floor(scaled)
        frac_part = scaled - int_part
        
        # 處理非 0.5 的情況（標準四捨五入）
        rounded = torch.floor(scaled + 0.5)
        
        # 特殊處理 0.5 的情況
        is_half = torch.isclose(frac_part, torch.tensor(0.5, device=input.device, dtype=input.dtype))
        if torch.any(is_half):
            # 對於小數部分為 0.5 的數，檢查整數部分
            is_odd = torch.fmod(int_part, 2) == 1
            
            # 對於整數部分為奇數且小數為 0.5 的情況，向上取整
            # 對於整數部分為偶數且小數為 0.5 的情況，向下取整
            corrected = torch.where(is_odd, int_part + 1, int_part)
            
            # 只修改小數部分為 0.5 的值
            rounded = torch.where(is_half, corrected, rounded)
        
        # 限制範圍並轉換為 int8
        x_forward_int8 = torch.clamp(rounded, min=-128, max=127).to(torch.int8)
        
        return x_forward_int8.detach()

class BaseQuantizer(nn.Module):
    def __init__(self, disable_act = False, layer_name = ""):
        super().__init__()
        self.disable_act = disable_act
        self.layer_name = layer_name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.info()})"

class FPQuantizer(BaseQuantizer):
    "FixedPoint quantizer"
    def __init__(self, bw = 16, fl = 9, disable_act=False, layer_name = ""):
        super().__init__(disable_act=disable_act, layer_name=layer_name)
        self.register_buffer('fl', torch.Tensor([fl]).int())
        self.register_buffer('bw', torch.Tensor([bw]).int())
    
    def quantize(self, x):
        # Quantize Activation
        out = x if self.disable_act else DFPQuantizeFunction.apply(x, self.bw, self.fl)
        return out

    def FPGA(self, x):
        assert not self.disable_act
        return self.quantize(x) * pow(2, self.fl)

    def forward(self, x):
        return self.quantize(x)

    def info(self):
        return f"bw = {self.bw.item()} , fl = {self.fl.item()} , disable_act = {self.disable_act}"

class DFPQuantizer(BaseQuantizer):
    "DFP quantizer"
    def __init__(self, bw = 8, momentum = 0.01, disable_act=False, layer_name = ""):
        super().__init__(disable_act=disable_act, layer_name=layer_name)
        self.momentum = momentum
        self.freeze_fl = False
        self.register_buffer('fl', torch.zeros(1).int())
        self.register_buffer('bw', torch.Tensor([bw]).int())
        self.register_buffer('abs_max', torch.zeros(1))

    def update_fl(self, x):
        # update moving average abs_max
        # calculate fl
        # x_round = torch.round(x.to(torch.float32) * 10000) / 10000.0
        # x_round.to(torch.float16)
        # assert not torch.any(x_round.isinf()) 
        # assert not torch.any(x_round.isnan())
    

        x_abs_max, new_fl = calculate_abs_FL(self.bw, x.detach())
        force_update = False
        if self.abs_max == 0:
            self.abs_max.add_(x_abs_max)
            force_update = True
        else:
            # self.abs_max.mul_(1 - self.momentum).add_(x_abs_max * self.momentum)
            self.abs_max.lerp_(x_abs_max.to(self.abs_max.dtype), self.momentum)
        
        # new_fl = self.bw - torch.log2(( 4 * self.abs_max )/3) - 1
        if abs(new_fl - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl: {self.fl.item()} => {new_fl.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl.copy_(torch.round(new_fl))
        # # adaptive bw
        # new_il = torch.round(torch.log2(( 4 * self.abs_max )/3))
        # self.bw.copy_(new_il)
        # self.fl.copy_(torch.zeros(1).int())
    
    def quantize(self, x):
        # Quantize Activation
        out = x if self.disable_act else DFPQuantizeFunction.apply(x, self.bw, self.fl)
        return out

    def quantize_int8(self, x):
        # Quantize Activation
        out = x if self.disable_act else DFPQuantizeFunction.apply_int(x, self.bw, self.fl)
        return out

    def FPGA(self, x):
        assert not self.disable_act
        return self.quantize(x) * pow(2, self.fl)

    def forward(self, x):
        
        if self.training and not self.freeze_fl:
            self.update_fl(x)
        return self.quantize(x)

    def info(self):
        return f"bw = {self.bw.item()} , fl = {self.fl.item()} , abs_max = {self.abs_max.item()} , disable_act = {self.disable_act} , freeze_fl = {self.freeze_fl}"


class DFPQuantizer_stdFL(DFPQuantizer):
    "DFP quantizer"
    def update_fl(self, x):
        # update moving average abs_max
        # calculate fl
        x_abs_max, new_fl = calculate_std_FL(self.bw, x.detach())
        force_update = False
        if self.abs_max == 0:
            self.abs_max.add_(x_abs_max)
            force_update = True
        else:
            self.abs_max.lerp_(x_abs_max.to(self.abs_max.dtype), self.momentum)
        
        if abs(new_fl - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl: {self.fl.item()} => {new_fl.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl.copy_(torch.round(new_fl))


class DFPQuantizer_negFL(BaseQuantizer):
    "Activation DFP Wrapper"
    def __init__(self, bw = 8, momentum = 0.01, disable_act=False, layer_name = ""):
        super().__init__(disable_act=disable_act, layer_name=layer_name)
        self.momentum = momentum
        self.freeze_fl = False
        self.register_buffer('fl', torch.zeros(1).int())
        self.register_buffer('bw', torch.Tensor([bw]).int())
        self.register_buffer('abs_max', torch.zeros(1))
        self.register_buffer('neg_fl', torch.zeros(1).int())
        self.register_buffer('neg_abs_max', torch.zeros(1))

    def update_fl(self, x):
        # update moving average abs_max
        x = x.detach().clone()
        x_abs = x.abs()
        mask = x>0
        if not mask.any(): mask = x < 0
        # x_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*6
        x_max = torch.max(x_abs.mul(mask))
        force_update = False
        if self.abs_max == 0:
            self.abs_max.add_(x_max)
            force_update = True
        else:
            # self.abs_max.mul_(1 - self.momentum).add_(x_max * self.momentum)
            self.abs_max.lerp_(x_max.to(self.abs_max.dtype), self.momentum)
        # calculate fl
        new_fl = calculate_FL(self.bw, self.abs_max)
        # new_fl = self.bw - torch.log2(( 4 * self.abs_max )/3) - 1
        # new_fl = find_best_fl(x[x>0], self.bw)
        if abs(new_fl - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl: {self.fl.item()} => {new_fl.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl.copy_(torch.round(new_fl))

        # update moving average abs_max
        # x_min = torch.abs(torch.min(x.detach().clone()))
        mask = x<0
        if not mask.any(): mask = x > 0
        # x_min = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*6
        x_min = torch.max(x_abs.mul(mask))
        x_min.nan_to_num_(x_max.item())
        force_update = False
        if self.neg_abs_max == 0:
            self.neg_abs_max.add_(x_min)
            force_update = True
        else:
            # self.neg_abs_max.mul_(1 - self.momentum).add_(x_min * self.momentum)
            self.neg_abs_max.lerp_(x_min.to(self.neg_abs_max.dtype), self.momentum)
        # calculate fl
        new_fl = calculate_FL(self.bw, self.neg_abs_max)
        # new_fl = self.bw - torch.log2(( 4 * self.abs_max )/3) - 1
        # new_fl = find_best_fl(x[x<0], self.bw)
        if abs(new_fl - self.neg_fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New neg_fl: {self.neg_fl.item()} => {new_fl.item():10f} neg_ABS_MAX: {self.neg_abs_max.item():10f}")
            self.neg_fl.copy_(torch.round(new_fl))
    
    def quantize(self, x):
        # Quantize Activation
        out = x if self.disable_act else torch.where(x >= 0 , DFPQuantizeFunction.apply(x, self.bw, self.fl), DFPQuantizeFunction.apply(x, self.bw, self.neg_fl))
        return out

    def FPGA(self, x):
        assert not self.disable_act
        raise NotImplementedError
        return self.quantize(x) * pow(2, self.fl)

    def forward(self, x):
        if self.training and not self.freeze_fl:
            self.update_fl(x)
        return self.quantize(x)

    def info(self):
        return f"bw = {self.bw.item()} , fl = {self.fl.item()} , abs_max = {self.abs_max.item()} , disable_act = {self.disable_act} , freeze_fl = {self.freeze_fl}" + \
            f" , neg_fl = {self.neg_fl.item()} , neg_abs_max = {self.neg_abs_max.item()}"
  
    
class DFPQuantizer_negFL_stdFL(DFPQuantizer_negFL):
    "Activation DFP Wrapper"
    def update_fl(self, x):
        # update moving average abs_max
        x = x.detach().clone()
        # x_max = torch.abs(torch.max(x.detach().clone()))
        x_pow2 = x.pow(2)
        mask = x>0
        x_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*3
        force_update = False
        if self.abs_max == 0:
            self.abs_max.add_(x_max)
            force_update = True
        else:
            # self.abs_max.mul_(1 - self.momentum).add_(x_max * self.momentum)
            self.abs_max.lerp_(x_max.to(self.abs_max.dtype), self.momentum)
        # calculate fl
        new_fl = calculate_FL(self.bw, self.abs_max)
        # new_fl = self.bw - torch.log2(( 4 * self.abs_max )/3) - 1
        # new_fl = find_best_fl(x[x>0], self.bw)
        if abs(new_fl - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl: {self.fl.item()} => {new_fl.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl.copy_(torch.round(new_fl))

        # update moving average abs_max
        # x_min = torch.abs(torch.min(x.detach().clone()))
        mask = x<0
        x_min = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*3
        x_min.nan_to_num_(x_max.item())
        force_update = False
        if self.neg_abs_max == 0:
            self.neg_abs_max.add_(x_min)
            force_update = True
        else:
            # self.neg_abs_max.mul_(1 - self.momentum).add_(x_min * self.momentum)
            self.neg_abs_max.lerp_(x_min.to(self.neg_abs_max.dtype), self.momentum)
        # calculate fl
        new_fl = calculate_FL(self.bw, self.neg_abs_max)
        # new_fl = self.bw - torch.log2(( 4 * self.abs_max )/3) - 1
        # new_fl = find_best_fl(x[x<0], self.bw)
        if abs(new_fl - self.neg_fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New neg_fl: {self.neg_fl.item()} => {new_fl.item():10f} neg_ABS_MAX: {self.neg_abs_max.item():10f}")
            self.neg_fl.copy_(torch.round(new_fl))


class DFPQuantizer_TwoScale_Symm(BaseQuantizer):
    "DFP TwoScale with negative FL quatizer"
    def __init__(self, bw = 8, momentum = 0.01, disable_act=False, layer_name = ""):
        super().__init__(disable_act=disable_act, layer_name=layer_name)
        self.momentum = momentum
        self.freeze_fl = False
        self.register_buffer('bw', torch.Tensor([bw]).int())

        self.register_buffer('fl', torch.zeros(1).int())
        self.register_buffer('std_max', torch.zeros(1))
        self.register_buffer('fl_tail', torch.zeros(1).int())
        self.register_buffer('abs_max', torch.zeros(1))

    def update_fl(self, x):
        x = x.detach().clone()
        x_pow2 = x.pow(2)
        # update moving average positive
        x_std_max = torch.sqrt(x_pow2.mean())*3
        x_abs_max = torch.max(torch.abs(x))
        
        # ----------------------------------------------------------------------------------
        if x_std_max > x_abs_max : x_std_max = x_abs_max # limit max range from 3 std
        # ----------------------------------------------------------------------------------
        
        force_update = False
        if self.std_max == 0:
            self.std_max.add_(x_std_max)
            self.abs_max.add_(x_abs_max)
            force_update = True
        else:
            self.std_max.lerp_(x_std_max.to(self.std_max.dtype), self.momentum)
            self.abs_max.lerp_(x_abs_max.to(self.abs_max.dtype), self.momentum)

        # calculate fl
        new_fl = calculate_FL(self.bw, self.std_max)
        new_fl_tail = calculate_FL(self.bw, self.abs_max)
        # update fl
        if abs(new_fl + 0.5 - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl: {self.fl.item()} => {new_fl.item():10f} STD_MAX: {self.std_max.item():10f}")
            self.fl.copy_(torch.ceil(new_fl))
        if abs(new_fl_tail + 0.5 - self.fl_tail) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_tail: {self.fl_tail.item()} => {new_fl_tail.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl_tail.copy_(torch.ceil(new_fl_tail))
    
    def quantize(self, x):
        # Quantize Activation
        if self.disable_act: return x
        q_max = max_represent_value_fl(self.bw, self.fl)
        out = torch.where(x.abs() < q_max , DFPQuantizeFunction.apply(x, self.bw, self.fl), DFPQuantizeFunction.apply(x, self.bw, self.fl_tail))
        return out

    def FPGA(self, x):
        assert not self.disable_act
        raise NotImplementedError
        return self.quantize(x) * pow(2, self.fl)

    def forward(self, x):
        if self.training and not self.freeze_fl:
            self.update_fl(x)
        return self.quantize(x)

    def info(self):
        return f"bw = {self.bw.item()} , disable_act = {self.disable_act} , freeze_fl = {self.freeze_fl}" + \
            f" , fl = {self.fl.item()} , std_max = {self.std_max.item()}" + \
            f" , fl_tail = {self.fl_tail.item()} , abs_max = {self.abs_max.item()}"


class DFPQuantizer_TwoScale_negFL(BaseQuantizer):
    "DFP TwoScale with negative FL quatizer"
    def __init__(self, bw = 8, momentum = 0.01, disable_act=False, layer_name = ""):
        super().__init__(disable_act=disable_act, layer_name=layer_name)
        self.momentum = momentum
        self.freeze_fl = False
        self.register_buffer('bw', torch.Tensor([bw]).int())

        self.register_buffer('fl', torch.zeros(1).int())
        self.register_buffer('std_max', torch.zeros(1))
        self.register_buffer('fl_tail', torch.zeros(1).int())
        self.register_buffer('abs_max', torch.zeros(1))

        self.register_buffer('neg_fl', torch.zeros(1).int())
        self.register_buffer('neg_std_max', torch.zeros(1))
        self.register_buffer('neg_fl_tail', torch.zeros(1).int())
        self.register_buffer('neg_abs_max', torch.zeros(1))

    def update_fl(self, x):
        x = x.detach().clone()
        x_pow2 = x.pow(2)
        # update moving average positive
        mask = x>0
        if not mask.any(): mask = x < 0
        x_std_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*3
        x_abs_max = torch.max(torch.abs(x.mul(mask)))
        
        # ----------------------------------------------------------------------------------
        if x_std_max > x_abs_max : x_std_max = x_abs_max # limit max range from 3 std
        # ----------------------------------------------------------------------------------
        
        force_update = False
        if self.std_max == 0:
            self.std_max.add_(x_std_max)
            self.abs_max.add_(x_abs_max)
            force_update = True
        else:
            self.std_max.lerp_(x_std_max.to(self.std_max.dtype), self.momentum)
            self.abs_max.lerp_(x_abs_max.to(self.abs_max.dtype), self.momentum)

        # calculate fl
        new_fl = calculate_FL(self.bw, self.std_max)
        new_fl_tail = calculate_FL(self.bw, self.abs_max)
        # update fl
        if abs(new_fl + 0.5 - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl: {self.fl.item()} => {new_fl.item():10f} STD_MAX: {self.std_max.item():10f}")
            self.fl.copy_(torch.ceil(new_fl))
        if abs(new_fl_tail + 0.5 - self.fl_tail) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_tail: {self.fl_tail.item()} => {new_fl_tail.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl_tail.copy_(torch.ceil(new_fl_tail))

        # update moving average negative
        # x_min = torch.abs(torch.min(x.detach().clone()))
        mask = x<0
        if not mask.any(): mask = x > 0
        x_std_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*3
        x_abs_max = torch.max(torch.abs(x.mul(mask)))
        
        # ----------------------------------------------------------------------------------
        if x_std_max > x_abs_max : x_std_max = x_abs_max # limit max range from 3 std
        # ----------------------------------------------------------------------------------
        
        force_update = False
        if self.neg_std_max == 0:
            self.neg_std_max.add_(x_std_max)
            self.neg_abs_max.add_(x_abs_max)
            force_update = True
        else:
            self.neg_std_max.lerp_(x_std_max.to(self.neg_std_max.dtype), self.momentum)
            self.neg_abs_max.lerp_(x_abs_max.to(self.neg_abs_max.dtype), self.momentum)

        # calculate fl
        new_fl = calculate_FL(self.bw, self.neg_std_max)
        new_fl_tail = calculate_FL(self.bw, self.neg_abs_max)
        if abs(new_fl + 0.5 - self.neg_fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New neg_fl: {self.neg_fl.item()} => {new_fl.item():10f} NEG_STD_MAX: {self.neg_std_max.item():10f}")
            self.neg_fl.copy_(torch.ceil(new_fl))
        if abs(new_fl_tail + 0.5 - self.neg_fl_tail) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New neg_fl_tail: {self.neg_fl_tail.item()} => {new_fl_tail.item():10f} NEG_ABS_MAX: {self.neg_abs_max.item():10f}")
            self.neg_fl_tail.copy_(torch.ceil(new_fl_tail))
    
    def quantize(self, x):
        # Quantize Activation
        if self.disable_act: return x
        q_max = max_represent_value_fl(self.bw, self.fl)
        q_max_neg = max_represent_value_fl(self.bw, self.neg_fl).neg_()
        out_pos = torch.where(x < q_max, DFPQuantizeFunction.apply(x, self.bw, self.fl), DFPQuantizeFunction.apply(x, self.bw, self.fl_tail))
        out_neg = torch.where(x < q_max_neg, DFPQuantizeFunction.apply(x, self.bw, self.neg_fl_tail), DFPQuantizeFunction.apply(x, self.bw, self.neg_fl))
        out = torch.where(x >= 0 , out_pos, out_neg)
        return out

    def FPGA(self, x):
        assert not self.disable_act
        raise NotImplementedError
        return self.quantize(x) * pow(2, self.fl)

    def forward(self, x):
        if self.training and not self.freeze_fl:
            self.update_fl(x)
        return self.quantize(x)

    def info(self):
        return f"bw = {self.bw.item()} , disable_act = {self.disable_act} , freeze_fl = {self.freeze_fl}" + \
            f" , fl = {self.fl.item()} , std_max = {self.std_max.item()}" + \
            f" , fl_tail = {self.fl_tail.item()} , abs_max = {self.abs_max.item()}" + \
            f" , neg_fl = {self.neg_fl.item()} , neg_std_max = {self.neg_std_max.item()}" + \
            f" , neg_fl_tail = {self.neg_fl_tail.item()} , neg_abs_max = {self.neg_abs_max.item()}"

# ---------------------------------------------------------------------------------------------------
class DFPQuantizer_ThreeScale_Symm(BaseQuantizer):
    "DFP ThreeScale with negative FL quantizer"
    def __init__(self, bw = 8, momentum = 0.01, disable_act=False, layer_name = ""):
        super().__init__(disable_act=disable_act, layer_name=layer_name)
        self.momentum = momentum
        self.freeze_fl = False
        self.register_buffer('bw', torch.Tensor([bw]).int())

        self.register_buffer('fl_head', torch.zeros(1).int())
        self.register_buffer('head_max', torch.zeros(1))
        
        self.register_buffer('fl' , torch.zeros(1).int())
        self.register_buffer('mid_max', torch.zeros(1))
        
        self.register_buffer('fl_tail', torch.zeros(1).int())
        self.register_buffer('abs_max', torch.zeros(1))

    def update_fl(self, x):
        x = x.detach().clone()
        x_pow2 = x.pow(2)
        
        # update moving average positive
        x_head_max = torch.sqrt(x_pow2.mean())*HEAD_STD_NUM
        x_mid_max = torch.sqrt(x_pow2.mean())*MID_STD_NUM
        x_abs_max = torch.max(torch.abs(x))
        
        # ----------------------------------------------------------------------------------
        # if x_mid_max > x_abs_max : x_mid_max = x_abs_max # limit max range from 3 std
        # ----------------------------------------------------------------------------------
        
        force_update = False
        if self.mid_max == 0:
            self.head_max.add_(x_head_max)
            self.mid_max.add_(x_mid_max)
            self.abs_max.add_(x_abs_max)
            force_update = True
        else:
            self.head_max.lerp_(x_head_max.to(self.head_max.dtype), self.momentum)
            self.mid_max.lerp_(x_mid_max.to(self.mid_max.dtype), self.momentum)
            self.abs_max.lerp_(x_abs_max.to(self.abs_max.dtype), self.momentum)

        # calculate fl
        new_fl_head = calculate_FL(self.bw, self.head_max)
        new_fl      = calculate_FL(self.bw, self.mid_max)
        new_fl_tail = calculate_FL(self.bw, self.abs_max)
        # update fl
        if abs(new_fl_head + 0.5 - self.fl_head) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_head: {self.fl_head.item()} => {new_fl_head.item():10f} HEAD_MAX: {self.head_max.item():10f}")
            self.fl_head.copy_(torch.ceil(new_fl_head))
            
        if abs(new_fl + 0.5 - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_mid: {self.fl.item()} => {new_fl.item():10f} MID_MAX: {self.mid_max.item():10f}")
            self.fl.copy_(torch.ceil(new_fl))
            
        if abs(new_fl_tail + 0.5 - self.fl_tail) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_tail: {self.fl_tail.item()} => {new_fl_tail.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl_tail.copy_(torch.ceil(new_fl_tail))
    
    def quantize(self, x):
        # Quantize Activation
        if self.disable_act: return x
        q_head_max = max_represent_value_fl(self.bw, self.fl_head)
        q_mid_max = max_represent_value_fl(self.bw, self.fl)
        
        region_head = x.abs() < q_head_max # region under one std
        region_mid = (x.abs() >= q_head_max) & (x.abs() < q_mid_max) # region between one std and three std

        out = torch.where(region_head, DFPQuantizeFunction.apply(x, self.bw, self.fl_head), 
              torch.where(region_mid , DFPQuantizeFunction.apply(x, self.bw, self.fl), DFPQuantizeFunction.apply(x, self.bw, self.fl_tail)))
        return out

    def FPGA(self, x):
        assert not self.disable_act
        raise NotImplementedError
        return self.quantize(x) * pow(2, self.fl)

    def forward(self, x):
        if self.training and not self.freeze_fl:
            self.update_fl(x)
        return self.quantize(x)

    def info(self):
        return f"bw = {self.bw.item()} , disable_act = {self.disable_act} , freeze_fl = {self.freeze_fl}" + \
            f" , fl_head = {self.fl_head.item()} , head_max = {self.head_max.item()}" + \
            f" , fl_mid  = {self.fl.item()} , mid_max = {self.mid_max.item()}" + \
            f" , fl_tail = {self.fl_tail.item()} , abs_max = {self.abs_max.item()}"
    

class DFPQuantizer_ThreeScale_negFL(BaseQuantizer):
    "DFP ThreeScale with negative FL quatizer"
    def __init__(self, bw = 8, momentum = 0.01, disable_act=False, layer_name = ""):
        super().__init__(disable_act=disable_act, layer_name=layer_name)
        self.momentum = momentum
        self.freeze_fl = False
        self.register_buffer('bw', torch.Tensor([bw]).int())
        
        self.register_buffer('fl_head', torch.zeros(1).int())
        self.register_buffer('head_max', torch.zeros(1))

        self.register_buffer('fl', torch.zeros(1).int())
        self.register_buffer('mid_max', torch.zeros(1))
        
        self.register_buffer('fl_tail', torch.zeros(1).int())
        self.register_buffer('abs_max', torch.zeros(1))

        self.register_buffer('neg_fl_head', torch.zeros(1).int())
        self.register_buffer('neg_head_max', torch.zeros(1))
        
        self.register_buffer('neg_fl', torch.zeros(1).int())
        self.register_buffer('neg_mid_max', torch.zeros(1))
        
        self.register_buffer('neg_fl_tail', torch.zeros(1).int())
        self.register_buffer('neg_abs_max', torch.zeros(1))

    def update_fl(self, x):
        x = x.detach().clone()
        x_pow2 = x.pow(2)
        
        # update moving average positive
        mask = x>0
        if not mask.any(): mask = x < 0
        
        x_head_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*HEAD_STD_NUM
        x_mid_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*MID_STD_NUM
        x_abs_max = torch.max(torch.abs(x.mul(mask)))
        
        # ----------------------------------------------------------------------------------
        # if x_mid_max > x_abs_max : x_mid_max = x_abs_max # limit max range from 3 std
        # ----------------------------------------------------------------------------------
        
        force_update = False
        if self.mid_max == 0:
            self.head_max.add_(x_head_max)
            self.mid_max.add_(x_mid_max)
            self.abs_max.add_(x_abs_max)
            force_update = True
        else:
            self.head_max.lerp_(x_head_max.to(self.head_max.dtype), self.momentum)
            self.mid_max.lerp_(x_mid_max.to(self.mid_max.dtype), self.momentum)
            self.abs_max.lerp_(x_abs_max.to(self.abs_max.dtype), self.momentum)

        # calculate fl
        new_fl_head = calculate_FL(self.bw, self.head_max)
        new_fl      = calculate_FL(self.bw, self.mid_max)
        new_fl_tail = calculate_FL(self.bw, self.abs_max)
        # update fl
        if abs(new_fl_head + 0.5 - self.fl_head) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_head: {self.fl_head.item()} => {new_fl_head.item():10f} HEAD_MAX: {self.head_max.item():10f}")
            self.fl_head.copy_(torch.ceil(new_fl_head))
            
        if abs(new_fl + 0.5 - self.fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_mid: {self.fl.item()} => {new_fl.item():10f} MID_MAX: {self.mid_max.item():10f}")
            self.fl.copy_(torch.ceil(new_fl))
            
        if abs(new_fl_tail + 0.5 - self.fl_tail) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New fl_tail: {self.fl_tail.item()} => {new_fl_tail.item():10f} ABS_MAX: {self.abs_max.item():10f}")
            self.fl_tail.copy_(torch.ceil(new_fl_tail))



        # update moving average negative
        # x_min = torch.abs(torch.min(x.detach().clone()))
        mask = x<0
        if not mask.any(): mask = x > 0
        
        x_head_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*HEAD_STD_NUM
        x_mid_max = torch.sqrt(x_pow2.mul(mask).sum().div(mask.sum()))*MID_STD_NUM
        x_abs_max = torch.max(torch.abs(x.mul(mask)))
        
        # ----------------------------------------------------------------------------------
        # if x_mid_max > x_abs_max : x_mid_max = x_abs_max # limit max range from 3 std
        # ----------------------------------------------------------------------------------
        
        force_update = False
        if self.neg_mid_max == 0:
            self.neg_head_max.add_(x_head_max)
            self.neg_mid_max.add_(x_mid_max)
            self.neg_abs_max.add_(x_abs_max)
            force_update = True
        else:
            self.neg_head_max.lerp_(x_head_max.to(self.neg_head_max.dtype), self.momentum)
            self.neg_mid_max.lerp_(x_mid_max.to(self.neg_mid_max.dtype), self.momentum)
            self.neg_abs_max.lerp_(x_abs_max.to(self.neg_abs_max.dtype), self.momentum)

        # calculate fl
        new_fl_head = calculate_FL(self.bw, self.neg_head_max)
        new_fl      = calculate_FL(self.bw, self.neg_mid_max)
        new_fl_tail = calculate_FL(self.bw, self.neg_abs_max)
        
        if abs(new_fl_head + 0.5 - self.neg_fl_head) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New neg_fl_head: {self.neg_fl_head.item()} => {new_fl_head.item():10f} NEG_HEAD_MAX: {self.neg_head_max.item():10f}")
            self.neg_fl_head.copy_(torch.ceil(new_fl_head))
        
        if abs(new_fl + 0.5 - self.neg_fl) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New neg_fl_mid: {self.neg_fl.item()} => {new_fl.item():10f} NEG_MID_MAX: {self.neg_mid_max.item():10f}")
            self.neg_fl.copy_(torch.ceil(new_fl))
            
        if abs(new_fl_tail + 0.5 - self.neg_fl_tail) > FL_THRESHOLD and not self.freeze_fl or force_update:
            print(f"{self.layer_name:40s} New neg_fl_tail: {self.neg_fl_tail.item()} => {new_fl_tail.item():10f} NEG_ABS_MAX: {self.neg_abs_max.item():10f}")
            self.neg_fl_tail.copy_(torch.ceil(new_fl_tail))
    
    def quantize(self, x):
        # Quantize Activation
        if self.disable_act: return x
        q_head_max = max_represent_value_fl(self.bw, self.fl_head)
        q_mid_max  = max_represent_value_fl(self.bw, self.fl)
        
        q_neg_head_max = max_represent_value_fl(self.bw, self.neg_fl_head).neg_()
        q_neg_mid_max = max_represent_value_fl(self.bw, self.neg_fl).neg_()
        
        region_head = x < q_head_max # region under one std
        region_mid = (x >= q_head_max) & (x < q_mid_max) # region between one std and three std
        
        region_neg_head = x > q_neg_head_max # region under one std
        region_neg_mid = (x <= q_neg_head_max) & (x > q_neg_mid_max) # region between one std and three std
        
        out_pos = torch.where(region_head, DFPQuantizeFunction.apply(x, self.bw, self.fl_head), 
                  torch.where(region_mid , DFPQuantizeFunction.apply(x, self.bw, self.fl), DFPQuantizeFunction.apply(x, self.bw, self.fl_tail)))
        
        out_neg = torch.where(region_neg_head, DFPQuantizeFunction.apply(x, self.bw, self.neg_fl_head), 
                  torch.where(region_neg_mid , DFPQuantizeFunction.apply(x, self.bw, self.neg_fl), DFPQuantizeFunction.apply(x, self.bw, self.neg_fl_tail)))
        
        out = torch.where(x >= 0 , out_pos, out_neg)
        return out

    def FPGA(self, x):
        assert not self.disable_act
        raise NotImplementedError
        return self.quantize(x) * pow(2, self.fl)

    def forward(self, x):
        if self.training and not self.freeze_fl:
            self.update_fl(x)
        return self.quantize(x)

    def info(self):
        return f"bw = {self.bw.item()} , disable_act = {self.disable_act} , freeze_fl = {self.freeze_fl}" + \
            f" , fl_head = {self.fl_head.item()} , head_max = {self.head_max.item()}" + \
            f" , fl_mid  = {self.fl.item()} , mid_max = {self.mid_max.item()}" + \
            f" , fl_tail = {self.fl_tail.item()} , abs_max = {self.abs_max.item()}" + \
            f" , neg_fl_head = {self.neg_fl_head.item()} , neg_head_max = {self.neg_head_max.item()}" + \
            f" , neg_fl_mid  = {self.neg_fl.item()} , neg_mid_max = {self.neg_mid_max.item()}" + \
            f" , neg_fl_tail = {self.neg_fl_tail.item()} , neg_abs_max = {self.neg_abs_max.item()}"
# ---------------------------------------------------------------------------------------------------



class BaseLayerWrapper(nn.Module):
    "Layer wrapper for QAT"
    def __init__(self, module_to_wrap, layer_name = "", input_quantizer = None, output_quantizer = None, weight_quantizer = None, bias_quantizer = None):
        super().__init__()
        self.input_quantizer = input_quantizer
        self.module_to_wrap = module_to_wrap
        self.output_quantizer = output_quantizer
        self.weight_quantizer = weight_quantizer
        self.bias_quantizer = bias_quantizer
        self.layer_name = layer_name
        self.disable_wei = False

    def forward(self, x):
        # x_round

        # if self.input_quantizer is None:
        #     q_x = x
        # else:
        #     x_round = torch.round(x * 10000) / 10000.0
        #     x_round.to(x.dtype)
        #     q_x = self.input_quantizer(x_round)

        q_x = x if self.input_quantizer is None else self.input_quantizer(x)
        # print("layer_name: ", self.layer_name)
        out = self.module_to_wrap(q_x)
        # if self.layer_name == "model.0.conv":
        #     print("conv out : ", out[0][0][0][0:29])
        q_out = out if self.output_quantizer is None else self.output_quantizer(out)
        # if self.layer_name == "model.0.conv":
        #     print("quantized : ", q_out[0][0][0][0:29])
        return q_out

    def quantize_weight(self):
        pass
    
    def quantize_weight_int8(self):
        pass

    def restore_weight_after_backward(self):
        pass

    def restore_weight_hook(self, module, gin, gout):
        self.restore_weight_after_backward()
        return gin

    def info(self):
        s = f"{self.__class__.__name__} : {self.wrapper_info()}\n"
        s += f"weight : {self.weight_quantizer.info() if self.weight_quantizer is not None else ''}\n"
        s += f"bias : {self.bias_quantizer.info() if self.bias_quantizer is not None else ''}\n"
        s += f"input : {self.input_quantizer.info() if self.input_quantizer is not None else ''}\n"
        s += f"output : {self.output_quantizer.info() if self.output_quantizer is not None else ''}"
        return s

    def extra_repr(self) -> str:
        return self.wrapper_info() + super().extra_repr()

    def wrapper_info(self):
        return ""

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as err:
            return getattr(self._modules["module_to_wrap"], name)

class ConvLinearLayerWrapper(BaseLayerWrapper):

    "Quantize weight using DFP"
    def __init__(self, module_to_wrap, layer_name = "", input_quantizer = None, output_quantizer = None, weight_quantizer = None, bias_quantizer = None):
        super().__init__(module_to_wrap, layer_name, input_quantizer, output_quantizer, weight_quantizer, bias_quantizer)
        assert isinstance(module_to_wrap, (nn.Conv2d, nn.Linear))
        self.register_buffer('weight_backup', self.module_to_wrap.weight.data.clone(), persistent=False)
        if self.module_to_wrap.bias is not None: self.register_buffer('bias_backup', self.module_to_wrap.bias.data.clone(), persistent=False)

    def quantize_weight(self):
        self.weight_backup.copy_(self.module_to_wrap.weight.data)
        if self.module_to_wrap.bias is not None: self.bias_backup.copy_(self.module_to_wrap.bias.data)
        if self.disable_wei: return
        if self.weight_quantizer is not None:
            m = self.weight_backup
            q_m = self.weight_quantizer(m)
            # print(q_m)
            self.module_to_wrap.weight.data.copy_(q_m)
        if self.module_to_wrap.bias is not None and self.bias_quantizer is not None:
            m = self.bias_backup
            q_m = self.bias_quantizer(m)
            self.module_to_wrap.bias.data.copy_(q_m)

    def quantize_weight_int8(self):
        self.weight_backup.copy_(self.module_to_wrap.weight.data)
        if self.module_to_wrap.bias is not None: self.bias_backup.copy_(self.module_to_wrap.bias.data)
        if self.disable_wei: return
        if self.weight_quantizer is not None:
            m = self.weight_backup
            # export q_m and fl_value by layer
            q_m = self.weight_quantizer.quantize_int8(m)
            fl_value = self.weight_quantizer.fl.item()
            
            # print(q_m)
            self.module_to_wrap.weight.data.copy_(q_m)
        if self.module_to_wrap.bias is not None and self.bias_quantizer is not None:
            m = self.bias_backup
            q_m = self.bias_quantizer.quantize_int8(m)
            self.module_to_wrap.bias.data.copy_(q_m)

    def restore_weight_after_backward(self):
        self.module_to_wrap.weight.data.copy_(self.weight_backup)
        if self.module_to_wrap.bias is not None: self.module_to_wrap.bias.data.copy_(self.bias_backup)

# 0416 add repconv wrapper and implicit function wrapper
# ========================================================================================================
class RepConvWrapper(BaseLayerWrapper):
    """Wrapper for RepConv modules with quantization of the addition operation"""
    def __init__(self, module_to_wrap, layer_name="", output_quantizer=None):
        super().__init__(module_to_wrap, layer_name, None, output_quantizer)
        self.add_quantizer = None  # Will be set during wrapping
        
    def forward(self, inputs):
        # Handle deployed mode
        if hasattr(self.module_to_wrap, "rbr_reparam"):
            out = self.module_to_wrap.rbr_reparam(inputs)
            if self.add_quantizer is not None:
                out = self.add_quantizer(out)
            out = self.module_to_wrap.act(out)
            return out if self.output_quantizer is None else self.output_quantizer(out)
        
        # Handle non-deployed mode with separate branches
        # Get output from each branch
        if self.module_to_wrap.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.module_to_wrap.rbr_identity(inputs)
        
        dense_out = self.module_to_wrap.rbr_dense(inputs)
        x1x1_out = self.module_to_wrap.rbr_1x1(inputs)
        
        # Sum the outputs
        out_sum = dense_out + x1x1_out + id_out
        
        # Apply quantization to the sum
        if self.add_quantizer is not None:
            out_sum = self.add_quantizer(out_sum)
        
        # Apply activation

        out = self.act(out_sum)
        
        # out = self.module_to_wrap.act(out_sum)
        # print(out[0][0][0])
        # print(self.module_to_wrap.act(out_sum))        
        # Apply output quantization( is already quantized)
        return out
        # return out if self.output_quantizer is None else self.output_quantizer(out)

class ImplicitWrapper(BaseLayerWrapper):
    """Wrapper for ImplicitA and ImplicitM modules"""
    def __init__(self, module_to_wrap, layer_name="", output_quantizer=None):
        super().__init__(module_to_wrap, layer_name, None, output_quantizer)
    
    def forward(self, x=None):
        out = self.module_to_wrap()
        return out if self.output_quantizer is None else self.output_quantizer(out)
    





class ImplicitAWrapper(BaseLayerWrapper):
    """Wrapper for ImplicitA modules in IDetect"""
    def __init__(self, module_to_wrap, layer_name="", output_quantizer=None, weight_quantizer=None):
        super().__init__(module_to_wrap, layer_name, None, output_quantizer, weight_quantizer, None)
        # Register backup buffer for the implicit parameter
        self.register_buffer('implicit_backup', self.module_to_wrap.implicit.data.clone(), persistent=False)
    
    def forward(self, x):
        # Apply implicit module with quantized parameter
        out = self.module_to_wrap(x)
        # Apply output quantization if available
        return out if self.output_quantizer is None else self.output_quantizer(out)
    
    def quantize_weight(self):
        if self.weight_quantizer is not None and not self.disable_wei:
            # Backup the original parameter
            self.implicit_backup.copy_(self.module_to_wrap.implicit.data)
            # Apply quantization
            q_implicit = self.weight_quantizer(self.implicit_backup)
            self.module_to_wrap.implicit.data.copy_(q_implicit)
    
    def quantize_weight_int8(self):
        if self.weight_quantizer is not None and not self.disable_wei:
            # Backup the original parameter
            self.implicit_backup.copy_(self.module_to_wrap.implicit.data)
            # Apply INT8 quantization
            q_implicit = self.weight_quantizer.quantize_int8(self.implicit_backup)
            self.module_to_wrap.implicit.data.copy_(q_implicit)
    
    def restore_weight_after_backward(self):
        if self.weight_quantizer is not None:
            self.module_to_wrap.implicit.data.copy_(self.implicit_backup)


class ImplicitMWrapper(BaseLayerWrapper):
    """Wrapper for ImplicitM modules in IDetect"""
    def __init__(self, module_to_wrap, layer_name="", output_quantizer=None, weight_quantizer=None):
        super().__init__(module_to_wrap, layer_name, None, output_quantizer, weight_quantizer, None)
        # Register backup buffer for the implicit parameter
        self.register_buffer('implicit_backup', self.module_to_wrap.implicit.data.clone(), persistent=False)
    
    def forward(self, x):
        # Apply implicit module with quantized parameter
        out = self.module_to_wrap(x)
        # Apply output quantization if available
        return out if self.output_quantizer is None else self.output_quantizer(out)
    
    def quantize_weight(self):
        if self.weight_quantizer is not None and not self.disable_wei:
            # Backup the original parameter
            self.implicit_backup.copy_(self.module_to_wrap.implicit.data)
            # Apply quantization
            q_implicit = self.weight_quantizer(self.implicit_backup)
            self.module_to_wrap.implicit.data.copy_(q_implicit)
    
    def quantize_weight_int8(self):
        if self.weight_quantizer is not None and not self.disable_wei:
            # Backup the original parameter
            self.implicit_backup.copy_(self.module_to_wrap.implicit.data)
            # Apply INT8 quantization
            q_implicit = self.weight_quantizer.quantize_int8(self.implicit_backup)
            self.module_to_wrap.implicit.data.copy_(q_implicit)
    
    def restore_weight_after_backward(self):
        if self.weight_quantizer is not None:
            self.module_to_wrap.implicit.data.copy_(self.implicit_backup)
# ========================================================================================================
class ConvLinearLayerWrapper_Fused(BaseLayerWrapper):
    "Quantize weight using DFP"
    def __init__(self, module_to_wrap, bn_layer, layer_name = "", input_quantizer = None, output_quantizer = None, weight_quantizer = None, bias_quantizer = None):
        super().__init__(module_to_wrap, layer_name, input_quantizer, output_quantizer, weight_quantizer, bias_quantizer)
        assert isinstance(module_to_wrap, (nn.Conv2d, nn.Linear))
        self.bn_layer = bn_layer
        if bn_layer is not None:
            bn_layer.momentum = 0.01
        self.freeze_fl = False
        self.freeze_bn_stat = False
        if isinstance(module_to_wrap, nn.Conv2d):
            self.function = partial(
                F.conv2d,
                stride   = self.module_to_wrap.stride,
                padding  = self.module_to_wrap.padding,
                dilation = self.module_to_wrap.dilation,
                groups   = self.module_to_wrap.groups
            )
        elif isinstance(module_to_wrap, nn.Linear):
            self.function = F.linear
        else:
            raise Exception("Unsupported function")

    def forward(self, x):
        q_x = x if self.input_quantizer is None else self.input_quantizer(x)
        if self.bn_layer is not None:
            # update bn
            if not self.freeze_bn_stat:
                direct_out = self.bn_layer(self.module_to_wrap(q_x))

            # fusing
            w, b = BN_fuse({
                "beta": self.bn_layer.bias,
                "gamma": self.bn_layer.weight,
                "running_mean": self.bn_layer.running_mean,
                "running_var": self.bn_layer.running_var,
                "weight": self.module_to_wrap.weight,
                "bias": self.module_to_wrap.bias,
            }, eps = self.bn_layer.eps)
        else:
            w = self.module_to_wrap.weight
            b = self.module_to_wrap.bias

        # quantize weight and bias
        q_w = w if self.weight_quantizer is None else self.weight_quantizer(w)
        q_b = b if self.bias_quantizer is None else self.bias_quantizer(b)
        out = self.function(
            input    = q_x,
            weight   = w if self.disable_wei else q_w,
            bias     = b if self.disable_wei else q_b,
        )
        if not self.freeze_bn_stat:
            out = out.detach() + direct_out - direct_out.detach()
        q_out = out if self.output_quantizer is None else self.output_quantizer(out)

        return q_out

    def fuse_for_deploy(self):
        "Fuse bn into module_to_wrap, do before unwrap"
        if self.bn_layer is None:
            print("No bn_layer found, not fusing")
            return
        print("Fusing for deploy...")
        # fusing
        w, b = BN_fuse({
            "beta": self.bn_layer.bias,
            "gamma": self.bn_layer.weight,
            "running_mean": self.bn_layer.running_mean,
            "running_var": self.bn_layer.running_var,
            "weight": self.module_to_wrap.weight,
            "bias": self.module_to_wrap.bias,
        }, eps = self.bn_layer.eps)

        self.module_to_wrap.weight.data.copy_(w)
        self.module_to_wrap.bias = torch.nn.Parameter(data=b, requires_grad=True)
        self.bn_layer = None

    def wrapper_info(self):
        return f"freeze_bn_stat = {self.freeze_bn_stat}"

class BinaryLayerWrapper(BaseLayerWrapper):
    "Quantize weight using DFP"
    def __init__(self, module_to_wrap, layer_name = "", input_quantizer = None, output_quantizer = None):
        super().__init__(module_to_wrap, layer_name, input_quantizer, output_quantizer)
        assert isinstance(module_to_wrap, (nn.Conv2d, nn.Linear))
        self.initialize_weight_bin()
        self.register_buffer('weight_backup', self.module_to_wrap.weight.data.clone(), persistent=False)
        self.n = self.module_to_wrap.weight.data[0].nelement()
        self.s = self.module_to_wrap.weight.data.size()
        if len(self.s) == 4: # conv2d
            self.binarizeParams = self._binarizeParams_Conv
            self._updateBinaryGradWeight = self._updateBinaryGradWeight_Conv
        elif len(self.s) == 2: # linear
            self.binarizeParams = self._binarizeParams_Linear
            self._updateBinaryGradWeight = self._updateBinaryGradWeight_Linear
        else:
            raise Exception("Unsupported binarize weight")
    
    def initialize_weight_bin(self):
        "Resize pretrained weight into [-1, +1] range"
        self.module_to_wrap.weight.data.copy_(self.module_to_wrap.weight.data.div(self.module_to_wrap.weight.data.abs().max()))
    
    def quantize_weight(self):
        # clamp parameters
        self.module_to_wrap.weight.data.clamp_(-1.0, 1.0)
        # save parameters
        self.weight_backup.copy_(self.module_to_wrap.weight.data)
        # binarize parameters
        self.binarizeParams()

    def updateBinaryGradWeight(self):
        weight = self.module_to_wrap.weight.data
        n = weight[0].nelement()
        s = weight.size()
        m_add = weight.sign().mul(self.module_to_wrap.weight.grad.data)
        if len(s) == 4:
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
        elif len(s) == 2:
            m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
        m_add = m_add.mul(weight.sign())
        m = self.module_to_wrap.weight.grad.data
        #m[weight.lt(-1.)] = 0 ##+1.960
        #m[weight.gt(+1.)] = 0
        index_tmp = weight.lt(-1.960) + weight.gt(+1.960)            
        m[index_tmp] = 0            
        self.module_to_wrap.weight.grad.data = m.add(m_add)
        
    def restore_weight_after_backward(self):
        self.module_to_wrap.weight.data.copy_(self.weight_backup)
        self.updateBinaryGradWeight()

    def _binarizeParams_Conv(self):
        m = self.module_to_wrap.weight.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(self.n)
        self.module_to_wrap.weight.data.sign_().mul_(m.expand(self.s))

    def _binarizeParams_Linear(self):
        m = self.module_to_wrap.weight.data.norm(1, 1, keepdim=True).div(self.n)
        self.module_to_wrap.weight.data.sign_().mul_(m.expand(self.s))

    def _updateBinaryGradWeight_Conv(self, m_add):
        m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(self.n).expand(self.s)
        return m_add
    
    def _updateBinaryGradWeight_Linear(self, m_add):
        m_add = m_add.sum(1, keepdim=True).div(self.n).expand(self.s)
        return m_add

class RPReLU(nn.PReLU):
    "ReActNet Parameterized ReLU"
    def __init__(self, out_channels):
        super().__init__(num_parameters=1)
        self.out_channels = out_channels
        self.bias1 = nn.Parameter(torch.zeros(1,out_channels,1,1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1,out_channels,1,1), requires_grad=True)

    def __repr__(self):
        s = f"{self.__class__.__name__}"
        s += f"(out_channels={self.out_channels}, num_parameters={self.num_parameters})\n"
        return s

    def forward(self, x):
        out1 = x + self.bias1.expand_as(x)
        out2 = super().forward(out1)
        out = out2 + self.bias2.expand_as(out2)
        return out

class RLeakyReLU(nn.LeakyReLU):
    "ReActNet Parameterized ReLU"
    def __init__(self, out_channels, negative_slope = 0.125, inplace = False):
        super().__init__(negative_slope, inplace)
        self.out_channels = out_channels
        self.bias1 = nn.Parameter(torch.zeros(1,out_channels,1,1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1,out_channels,1,1), requires_grad=True)

    def __repr__(self):
        s = f"{self.__class__.__name__}"
        s += f"(out_channels={self.out_channels}, negative_slope={self.negative_slope}, inplace={self.inplace})\n"
        return s

    def forward(self, x):
        out1 = x + self.bias1.expand_as(x)
        out2 = super().forward(out1)
        out = out2 + self.bias2.expand_as(out2)
        return out

#############################################################################################

class QuantModel(nn.Module):
    "Wrapped conv2d for activation quantization"
    class QuantConfig():
        def __init__(self):
            self.layer_index_not_to_quantize = []
            self.layer_index_keep_8_bits = []
            self.layer_name_not_to_wrap = []
            self.layer_index_quantize_output = []
            self.act_quantizer = DFPQuantizer
            self.wei_quantizer = DFPQuantizer
            # 0330 add bias quantizer
            
            # self.bias_quantizer = None
            self.bias_quantizer = DFPQuantizer
            # self.wei_quantizer_fused = DFPQuantizer
        def __repr__(self) -> str:
            return f"QuantConfig: {self.__dict__}"
    # 0406 add new init function     
    # def __init__(self, model, a_bw = 8, w_bw = 8, b_bw = None, act_disabled = False, wei_disabled = False, is_fused = False, quant_config: QuantConfig = QuantConfig()):
    #     super().__init__()
    #     self.a_bw = a_bw # activation bw
    #     self.w_bw = w_bw # weight bw
    #     self.b_bw = w_bw if b_bw is None else b_bw
    #     self.layers_to_wrapped = []
    #     # wrap detection layer
    #     self.module = copy.deepcopy(model)
    #     self.act_disabled = act_disabled
    #     self.wei_disabled = wei_disabled
    #     self.is_fused = is_fused
    #     self.quant_config = quant_config
    #     if w_bw == 1:
    #         self.wrap_binary()
    #     else:
    #         self.wrap_DFP_only()
    #     self.set_act_disable(act_disabled)
    #     self.set_weight_disable(wei_disabled)
    #     self.source_file = __file__
    #     # if self.act_disabled: self.disable_act()
    #     # print(self.wrapper_info())

    def __init__(self, model, a_bw = 8, w_bw = 8, b_bw = None, act_disabled = False, wei_disabled = False, is_fused = False, wrap_all = False, quant_config: QuantConfig = QuantConfig()):
        super().__init__()
        self.model = model.model
        self.a_bw = a_bw # activation bw
        self.w_bw = w_bw # weight bw
        self.b_bw = w_bw if b_bw is None else b_bw
        self.layers_to_wrapped = []
        # wrap detection layer
        self.module = copy.deepcopy(model)
        self.act_disabled = act_disabled
        self.wei_disabled = wei_disabled
        self.is_fused = is_fused
        self.wrap_all = wrap_all
        self.quant_config = quant_config
        
        if w_bw == 1:
            self.wrap_binary()
        elif wrap_all:
            self.wrap_all_layer()
        else:
            self.wrap_DFP_only()
            
        self.set_act_disable(act_disabled)
        self.set_weight_disable(wei_disabled)
        self.source_file = __file__


    def forward(self, x, *paras, **key_paras):
        # return self.module(x-0.5, *paras, **key_paras) # make first layer input symmetric
        # print(INPUT_BIAS)
        # assert False 
        return self.module(x + INPUT_BIAS, *paras, **key_paras) # keep original input
        # return self.model(x)

    def set_act_disable(self, act_disabled: bool):
        print("Disable activation quantize:", act_disabled)
        self.act_disabled = act_disabled
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                if m.input_quantizer is not None: m.input_quantizer.disable_act = act_disabled
                if m.output_quantizer is not None: m.output_quantizer.disable_act = act_disabled

    def set_weight_disable(self, wei_disabled: bool):
        print("Disable weight quantize:", wei_disabled)
        self.wei_disabled = wei_disabled
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                m.disable_wei = wei_disabled

    def set_freeze_act_fl(self, freeze: bool):
        print("Freeze activation FL:", freeze)
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                if m.input_quantizer is not None: m.input_quantizer.freeze_fl = freeze
                if m.output_quantizer is not None: m.output_quantizer.freeze_fl = freeze

    def set_freeze_wei_fl(self, freeze: bool):
        print("Freeze weight FL:", freeze)
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                if m.weight_quantizer is not None: m.weight_quantizer.freeze_fl = freeze
                if m.bias_quantizer is not None: m.bias_quantizer.freeze_fl = freeze

    def set_freeze_bn_statistic(self, freeze: bool):
        print("Freeze BN statistic in fusing model:", freeze)
        for n, m in self.module.named_modules():
            if isinstance(m, (ConvLinearLayerWrapper_Fused)):
                m.freeze_bn_stat = freeze

    def wrap_DFP_only(self):
        if self.is_fused:
            print("Fusing model...")
            ms = [(n,m) for n,m in self.module.named_modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
            conv_bn_dict = {}
            for i,(n,m) in enumerate(ms):
                if isinstance(m, nn.BatchNorm2d):
                    conv_linear, bn = ms[i-1], ms[i]
                    # sanity check
                    assert isinstance(conv_linear[1], (nn.Conv2d, nn.Linear)), f"Conv2d not found for BN {bn}"

                    out_channels = conv_linear[1].out_features if isinstance(conv_linear[1], torch.nn.Linear)\
                                else conv_linear[1].out_channels
                    assert out_channels == m.num_features, f"Channel num not match for {conv_linear} {bn}"
                    assert conv_linear[1].bias is None, f"Conv2d {conv_linear} has bias already for BN {bn}"
                    conv_bn_dict[conv_linear[0]] = bn
    
        
        # wrap model
        layer_to_wrap = self.module
        len_before = len(layer_to_wrap._modules)

        # do not use low-bit on first and detect (last) layer
        conv_linear_index_list = [i for i, (n,m) in enumerate(layer_to_wrap.named_modules()) if isinstance(m, (nn.Conv2d, nn.Linear))]
        conv_linear_8_bits = [conv_linear_index_list[i] for i in self.quant_config.layer_index_keep_8_bits]
        conv_linear_qout = [conv_linear_index_list[i] for i in self.quant_config.layer_index_quantize_output]
        conv_linear_no_quant = [conv_linear_index_list[i] for i in self.quant_config.layer_index_not_to_quantize]
        prev_out_channel = None

        act_quantizer  = self.quant_config.act_quantizer
        wei_quantizer  = self.quant_config.wei_quantizer
        bias_quantizer = self.quant_config.bias_quantizer
        # wei_quantizer_fused = self.quant_config.wei_quantizer_fused

        for i, (n, m) in enumerate(layer_to_wrap.named_modules()):
            print(n,m)
            if n in self.quant_config.layer_name_not_to_wrap:
                print("Not wrapping: ", n)
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                prev_out_channel = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
                if i in conv_linear_no_quant:
                    print("Skipping", type(m).__name__, n)
                    continue
                if i in conv_linear_8_bits:
                    print("Force 8 bits", type(m).__name__, n)
                self.layers_to_wrapped.append(n)
                assert self.is_fused, "Support fused only"
                # if self.is_fused and n in conv_bn_dict:
                #     replace_layer(layer_to_wrap, n, ConvLinearLayerWrapper_Fused(
                #         module_to_wrap   = m,
                #         bn_layer         = conv_bn_dict[n][1],
                #         layer_name       = n,
                #         input_quantizer  = act_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"input of {n}"),
                #         output_quantizer = None,
                #         weight_quantizer = wei_quantizer(bw = 8 if i in conv_linear_8_bits else self.w_bw, layer_name=f"weight of {n}"),
                #         bias_quantizer   = None
                #         )
                #     )
                #     replace_layer(self.module, conv_bn_dict[n][0], nn.Identity())
                
                # 0330 try to add bias and output quantizer for fused yolo model
                if self.is_fused and n in conv_bn_dict:
                    replace_layer(layer_to_wrap, n, ConvLinearLayerWrapper_Fused(
                        module_to_wrap   = m,
                        bn_layer         = conv_bn_dict[n][1],
                        layer_name       = n,
                        input_quantizer  = act_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"input of {n}"),
                        output_quantizer = DFPQuantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"output of {n}") if i in conv_linear_qout else None,
                        weight_quantizer = wei_quantizer(bw = 8 if i in conv_linear_8_bits else self.w_bw, layer_name=f"weight of {n}"),
                        bias_quantizer   = bias_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"bias of {n}")
                        )
                    )
                    replace_layer(self.module, conv_bn_dict[n][0], nn.Identity())
                
                
                else: # not fused or last conv layer
                    replace_layer(layer_to_wrap, n, ConvLinearLayerWrapper(
                        module_to_wrap   = m,
                        layer_name       = n,
                        input_quantizer  = act_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"input of {n}"),
                        output_quantizer = DFPQuantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"output of {n}") if i in conv_linear_qout else None,
                        weight_quantizer = wei_quantizer(bw = 8 if i in conv_linear_8_bits else self.w_bw, layer_name=f"weight of {n}"),
                        bias_quantizer   = None
                        )
                    )
            # elif isinstance(m, OUTPUT_QUANTIZED_LAYERS):
            #     self.layers_to_wrapped.append(n)
            #     replace_layer(layer_to_wrap, n, BaseLayerWrapper(
            #             module_to_wrap   = m,
            #             layer_name       = n,
            #             input_quantizer  = None,
            #             output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
            #             )
            #         )
            elif isinstance(m, INOUT_QUANTIZED_LAYERS):
                self.layers_to_wrapped.append(n)
                replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                        module_to_wrap   = m,
                        layer_name       = n,
                        input_quantizer  = act_quantizer(bw = self.a_bw, layer_name=f"input of {n}"),
                        output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                        )
                    )
            elif isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True
                m.momentum = 0.01
            # elif isinstance(m, ACTIVATION_LAYERS):
                # replace_layer(layer_to_wrap, n, RPReLU(out_channels=prev_out_channel))
                # replace_layer(layer_to_wrap, n, nn.LeakyReLU(0.125, inplace=True))
                # replace_layer(layer_to_wrap, n, RLeakyReLU(prev_out_channel, 0.125, inplace=True))
        # assert False
        assert len(layer_to_wrap._modules) == len_before, f"Some layer are not wrapped in {layer_to_wrap}"


    #  0406, add new finction to wrap all layers with output quantizers

    def wrap_all_layer(self):
        """
        Wrap all layers with output quantizers to ensure quantized inputs to the next layer.
        Input quantizer is only added to the first layer that processes the image input.
        Conv and Linear layers still have weight and bias quantizers as needed.
        """
        if self.is_fused:
            print("Fusing model...")
            ms = [(n,m) for n,m in self.module.named_modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
            conv_bn_dict = {}
            for i,(n,m) in enumerate(ms):
                if isinstance(m, nn.BatchNorm2d):
                    conv_linear, bn = ms[i-1], ms[i]
                    # sanity check
                    assert isinstance(conv_linear[1], (nn.Conv2d, nn.Linear)), f"Conv2d not found for BN {bn}"

                    out_channels = conv_linear[1].out_features if isinstance(conv_linear[1], torch.nn.Linear)\
                                else conv_linear[1].out_channels
                    assert out_channels == m.num_features, f"Channel num not match for {conv_linear} {bn}"
                    assert conv_linear[1].bias is None, f"Conv2d {conv_linear} has bias already for BN {bn}"
                    conv_bn_dict[conv_linear[0]] = bn

        # wrap model
        layer_to_wrap = self.module
        len_before = len(layer_to_wrap._modules)

        # Find the first layer that processes the image input
        first_layer_name = None
        first_layer_idx = None
        for i, (n, m) in enumerate(layer_to_wrap.named_modules()):
            if isinstance(m, (nn.Conv2d, nn.Linear)) and first_layer_name is None:
                first_layer_name = n
                first_layer_idx = i
                break

        # do not use low-bit on first and detect (last) layer
        conv_linear_index_list = [i for i, (n,m) in enumerate(layer_to_wrap.named_modules()) if isinstance(m, (nn.Conv2d, nn.Linear))]
        conv_linear_8_bits = [conv_linear_index_list[i] for i in self.quant_config.layer_index_keep_8_bits]
        conv_linear_no_quant = [conv_linear_index_list[i] for i in self.quant_config.layer_index_not_to_quantize]
        prev_out_channel = None

        act_quantizer = self.quant_config.act_quantizer
        wei_quantizer = self.quant_config.wei_quantizer
        bias_quantizer = self.quant_config.bias_quantizer

        for i, (n, m) in enumerate(layer_to_wrap.named_modules()):
            if n in self.quant_config.layer_name_not_to_wrap:
                print("Not wrapping: ", n)
                continue
            # print("====================================")
            # print("n : ", n)
            # print("m : ", m)
            # print("m_class_name : ", m.__class__.__name__)
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                prev_out_channel = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
                if i in conv_linear_no_quant:
                    print("Skipping", type(m).__name__, n)
                    continue
                if i in conv_linear_8_bits:
                    print("Force 8 bits", type(m).__name__, n)
                
                self.layers_to_wrapped.append(n)
                
                # Only add input quantizer to the first layer
                input_quantizer = act_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"input of {n}") if i == first_layer_idx else None
                
                # Prepare bias quantizer if bias is present
                b_quantizer = None
                if bias_quantizer is not None and (m.bias is not None or (self.is_fused and n in conv_bn_dict)):
                    # Use 8-bit for bias if in 8-bit list, otherwise use self.b_bw
                    b_quantizer = bias_quantizer(bw = 8 if i in conv_linear_8_bits else self.b_bw, layer_name=f"bias of {n}")
                is_detect_last_conv = (
                    isinstance(m, nn.Conv2d)
                    and m.out_channels == 18   # 3 * (nc + 5)
                )

                output_q = None if is_detect_last_conv else act_quantizer(
                    bw = 8 if i in conv_linear_8_bits else self.a_bw,
                    layer_name = f"output of {n}"
                )
                if self.is_fused and n in conv_bn_dict:
                    replace_layer(layer_to_wrap, n, ConvLinearLayerWrapper_Fused(
                        module_to_wrap   = m,
                        bn_layer         = conv_bn_dict[n][1],
                        layer_name       = n,
                        input_quantizer  = input_quantizer,
                        output_quantizer = act_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"output of {n}"),
                        # output_quantizer = output_q,
                        weight_quantizer = wei_quantizer(bw = 8 if i in conv_linear_8_bits else self.w_bw, layer_name=f"weight of {n}"),
                        bias_quantizer   = b_quantizer
                        )
                    )
                    replace_layer(self.module, conv_bn_dict[n][0], nn.Identity())
                else: # not fused or last conv layer
                    replace_layer(layer_to_wrap, n, ConvLinearLayerWrapper(
                        module_to_wrap   = m,
                        layer_name       = n,
                        input_quantizer  = input_quantizer,
                        output_quantizer = act_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"output of {n}"),
                        weight_quantizer = wei_quantizer(bw = 8 if i in conv_linear_8_bits else self.w_bw, layer_name=f"weight of {n}"),
                        bias_quantizer   = b_quantizer
                        )
                    )
            
            elif isinstance(m, OUTPUT_QUANTIZED_LAYERS):
                self.layers_to_wrapped.append(n)
                
                if "model.105" in n or "IDetect" in n:
                    output_q = None
                else:
                    output_q = act_quantizer(
                        bw=self.a_bw, layer_name=f"output of {n}"
                    )
                replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                    module_to_wrap   = m,
                    layer_name       = n,
                    input_quantizer  = None,  # No input quantizer
                    output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                    # output_quantizer = output_q
                    )
                )
            
            # Make sure activation layers are wrapped
            # Handle both standalone activation modules and activation functions within other modules
            elif isinstance(m, (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Mish, nn.ReLU6, ACTIVATION_LAYERS)):
                self.layers_to_wrapped.append(n)
                # print(n)
                replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                    module_to_wrap   = m,
                    layer_name       = n,
                    input_quantizer  = None,  # No input quantizer
                    output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                    )
                )
            
            
            # Handle RepConv modules
            elif "RepConv" in m.__class__.__name__ and QUANT_REPCONV:
                self.layers_to_wrapped.append(n)
                repconv_wrapper = RepConvWrapper(
                    module_to_wrap=m,
                    layer_name=n,
                    output_quantizer=None
                )
                
                # Add quantizer for the addition operation inside RepConv
                repconv_wrapper.add_quantizer = act_quantizer(bw=self.a_bw, layer_name=f"add of {n}")
                # print(m)
                replace_layer(layer_to_wrap, n, repconv_wrapper)
                
           # Handle IDetect layer with its ImplicitA and ImplicitM modules
            elif "IDetect" in m.__class__.__name__ and QUANT_IMPLICIT:
                self.layers_to_wrapped.append(n)
                
                # For ImplicitA modules
                for j in range(len(m.ia)):
                    implicit_name = f"{n}.ia.{j}"
                    implicit_wrapper = ImplicitAWrapper(
                        module_to_wrap=m.ia[j],
                        layer_name=implicit_name,
                        output_quantizer=act_quantizer(bw=self.a_bw, layer_name=f"output of {implicit_name}"),
                        weight_quantizer=wei_quantizer(bw=self.w_bw, layer_name=f"param of {implicit_name}")
                    )
                    m.ia[j] = implicit_wrapper

                # For ImplicitM modules
                for j in range(len(m.im)):
                    implicit_name = f"{n}.im.{j}"
                    implicit_wrapper = ImplicitMWrapper(
                        module_to_wrap=m.im[j],
                        layer_name=implicit_name,
                        output_quantizer=act_quantizer(bw=self.a_bw, layer_name=f"output of {implicit_name}"),
                        weight_quantizer=wei_quantizer(bw=self.w_bw, layer_name=f"param of {implicit_name}")
                    )
                    m.im[j] = implicit_wrapper
            # Make sure activation layers are wrapped
            # Handle both standalone activation modules and activation functions within other modules
            elif isinstance(m, (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Mish, nn.ReLU6, ACTIVATION_LAYERS)):
                self.layers_to_wrapped.append(n)
                replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                    module_to_wrap   = m,
                    layer_name       = n,
                    input_quantizer  = None,  # No input quantizer
                    output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                    )
                )
            
            elif isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True
                m.momentum = 0.01
                if not self.is_fused:  # Only wrap BN if not using fused mode
                    self.layers_to_wrapped.append(n)
                    replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                        module_to_wrap   = m,
                        layer_name       = n,
                        input_quantizer  = None,  # No input quantizer
                        output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                        )
                    )
            
            # Handle pooling layers
            elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
                self.layers_to_wrapped.append(n)
                replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                    module_to_wrap   = m,
                    layer_name       = n,
                    input_quantizer  = None,  # No input quantizer
                    output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                    )
                )
                
            # Handle Upsample layers
            elif isinstance(m, nn.Upsample):
                self.layers_to_wrapped.append(n)
                replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                    module_to_wrap   = m,
                    layer_name       = n,
                    input_quantizer  = None,  # No input quantizer
                    output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                    )
                )
                
        assert len(layer_to_wrap._modules) == len_before, f"Some layer are not wrapped in {layer_to_wrap}"
    def wrap_binary(self):
        layer_to_wrap = self.module
        # do not binarize first and detect (last) layer
        len_before = len(layer_to_wrap._modules)

        # do not use low-bit on first and detect (last) layer
        conv_linear_index_list = [i for i, (n,m) in enumerate(layer_to_wrap.named_modules()) if isinstance(m, (nn.Conv2d, nn.Linear))]
        conv_linear_8_bits = [conv_linear_index_list[i] for i in self.quant_config.layer_index_keep_8_bits]
        conv_linear_qout = [conv_linear_index_list[i] for i in self.quant_config.layer_index_quantize_output]
        conv_linear_no_quant = [conv_linear_index_list[i] for i in self.quant_config.layer_index_not_to_quantize]
        prev_out_channel = None

        assert not self.is_fused, "Binary quantization need batch norm !!"

        act_quantizer = self.quant_config.act_quantizer
        wei_quantizer = BinaryLayerWrapper

        for i, (n, m) in enumerate(layer_to_wrap.named_modules()):
            if n in self.quant_config.layer_name_not_to_wrap:
                print("Not wrapping: ", n)
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                prev_out_channel = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
                if i in conv_linear_no_quant:
                    print("Skipping", type(m).__name__, n)
                    continue
                if i in conv_linear_8_bits:
                    print("Force 8 bits", type(m).__name__, n)
                self.layers_to_wrapped.append(n)
                replace_layer(layer_to_wrap, n, wei_quantizer(
                    module_to_wrap = m,
                    layer_name = n,
                    input_quantizer = act_quantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"input of {n}"),
                    output_quantizer = DFPQuantizer(bw = 8 if i in conv_linear_8_bits else self.a_bw, layer_name=f"output of {n}") if i in conv_linear_qout else None,
                    )
                )
            elif isinstance(m, OUTPUT_QUANTIZED_LAYERS):
                self.layers_to_wrapped.append(n)
                replace_layer(layer_to_wrap, n, BaseLayerWrapper(
                        module_to_wrap = m,
                        layer_name = n,
                        input_quantizer = None,
                        output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
                        )
                    )
            # elif isinstance(m, INOUT_QUANTIZED_LAYERS):
            #     self.layers_to_wrapped.append(n)
            #     replace_layer(layer_to_wrap, n, BaseLayerWrapper(
            #             module_to_wrap = m,
            #             layer_name = n,
            #             input_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"input of {n}"),
            #             output_quantizer = act_quantizer(bw = self.a_bw, layer_name=f"output of {n}")
            #             )
            #         )
            elif isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.001
                m.reset_parameters()
            # elif isinstance(m, ACTIVATION_LAYERS):
                # replace_layer(layer_to_wrap, n, RPReLU(out_channels=prev_out_channel))
                # replace_layer(layer_to_wrap, n, nn.LeakyReLU(0.125, inplace=True))
                # replace_layer(layer_to_wrap, n, RLeakyReLU(prev_out_channel, 0.125, inplace=True))
            
        assert len(layer_to_wrap._modules) == len_before, f"Some layer are not wrapped in {layer_to_wrap}"

    def quantize_weight_before_forward(self):
        "Quantize weight before forward"
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                m.quantize_weight()
    def quantize_weight_export_int8(self):
        "Quantize weight with torch_int8"
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                m.quantize_weight_int8()


    def restore_weight_after_backward(self):
        "Restore weight after backward"
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                m.restore_weight_after_backward()

    def fuse_batchnorm(self):
        for n, m in self.module.named_modules():
            if isinstance(m, ConvLinearLayerWrapper_Fused):
                print(f"Fusing {n} : {m.__class__.__name__}")
                m.fuse_for_deploy()
                new_wrapper = ConvLinearLayerWrapper(
                    module_to_wrap = m.module_to_wrap,
                    layer_name = n,
                    input_quantizer = m.input_quantizer,
                    output_quantizer = m.output_quantizer,
                    weight_quantizer = m.weight_quantizer,
                    bias_quantizer = m.bias_quantizer
                ).to(m.weight_quantizer.fl.device)
                replace_layer(self.module, n, new_wrapper)
        if not self.module.training : self.module.eval()

    # def unwrap(self):
    #     # unwrap model
    #     layer_to_wrap = self.module
    #     len_before = len(layer_to_wrap._modules)
    #     for n, m in layer_to_wrap.named_modules():
    #         print(f"Unwrapping {n} : {m.__class__.__name__}")
    #         if isinstance(m, BaseLayerWrapper):
    #             replace_layer(layer_to_wrap, n, m.module_to_wrap)
    #     assert len(layer_to_wrap._modules) == len_before, f"Some layer are not unwrapped in {layer_to_wrap}"
    
    # 0416, update unwrap function to support RepConv and ImplicitWrapper
    def unwrap(self):
        """
        Unwrap the model by replacing wrapper modules with their wrapped modules.
        This implementation handles the hierarchical structure correctly by unwrapping
        modules from the bottom up.
        """
        # Create a dictionary to track modules that need unwrapping
        to_unwrap = {}
        
        # Find all wrapped modules
        for n, m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                # Add to dictionary with key = depth (number of dots in name)
                depth = n.count('.')
                if depth not in to_unwrap:
                    to_unwrap[depth] = []
                to_unwrap[depth].append((n, m))
    
        # Sort by depth in descending order (unwrap deepest modules first)
        for depth in sorted(to_unwrap.keys(), reverse=True):
            for n, m in to_unwrap[depth]:
                try:
                    print(f"Unwrapping {n} : {m.__class__.__name__}")
                    replace_layer(self.module, n, m.module_to_wrap)
                except AttributeError as e:
                    print(f"Warning: Could not unwrap {n}: {e}")
                    # Continue with other modules even if one fails


    def wrapper_info(self) -> str:
        s = f"QuantModel : {self.info()}\n"
        s += "="*30 + "\n"
        for n,m in self.module.named_modules():
            if isinstance(m, BaseLayerWrapper):
                s += f"( {n} ) : \n"
                s += f"{m.info()}\n"
                s += str(m) + "\n" + "="*30 + "\n"
        return s
    def info(self) -> str:
        return f"{self.source_file} , a_bw = {self.a_bw} , w_bw = {self.w_bw} , b_bw = {self.b_bw} , act_disabled = {self.act_disabled} , is_fused = {self.is_fused} , wrap_all = {self.wrap_all}\n{repr(self.quant_config)}"
    # def info(self) -> str:
    #     return f"{self.source_file} , a_bw = {self.a_bw} , w_bw = {self.w_bw} , b_bw = {self.b_bw} , act_disabled = {self.act_disabled} , is_fused = {self.is_fused}\n{repr(self.quant_config)}"

    def extra_repr(self) -> str:
        return self.info() + super().extra_repr()

    # def wrapper_info(self):
    #     s = f"QuantModel : a_bw = {self.a_bw} , w_bw = {self.w_bw} , act_disabled = {self.act_disabled} , is_fused = {self.is_fused}\n"
    #     s += f"Config: {self.quant_config.__dict__}\n" + "="*30 + "\n"
    #     for n in self.layers_to_wrapped:
    #         m = self.module
    #         for l in n.split("."):
    #             m = getattr(m, l)
    #         assert isinstance(m, BaseLayerWrapper)
    #         s += f"( {n} ) : \n"
    #         s += f"{m.info()}\n"
    #         s += str(m) + "\n" + "="*30 + "\n"
    #     return s
    def fuse(self):
        print("[QuantModel] fuse() skipped — quantized model has no BN fusion step.")
        return self

    def progressive_step(self):
        "Enable act one by one each step"
        disabled_act_list = [m for n,m in self.named_modules() if isinstance(m, DFPQuantizer) and m.disable_act is True]
        if len(disabled_act_list) == 0: return
        max_fl_i = 0
        for i in range(len(disabled_act_list)):
            if disabled_act_list[max_fl_i].abs_max > disabled_act_list[i].abs_max:
                max_fl_i = i
        disabled_act_list[max_fl_i].disable_act = False
        print(f"Progressive_step: {len(disabled_act_list)} left")
