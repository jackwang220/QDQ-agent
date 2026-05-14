# ezQuant-AMS Readme
## 實驗數據
- [ezQuant-AMS 模型訓練紀錄](<ezQuant-AMS 模型訓練紀錄.xlsx>)
- [AIMET 模型訓練紀錄](<AIMET 模型訓練紀錄.xlsx>)
## 交接影片
[Youtube 連結](https://youtu.be/OvGniMpHj14)
## Quant Model 架構
### QuantizeFunction
- `DFPQuantizeFunction` 量化函數，須注意backward寫法，需支援STEBC
### Quantizer
- `BaseQuantizer` Quantizer base class
- `DFPQuantizer` Dynamic Fixed Point Quantizer，用max計算FL
- `DFPQuantizer_stdFL` Dynamic Fixed Poin Quantizer，用std計算FL
- `FPQuantizer` Fixed Point Quantizer for DLA bias
- `DFPQuantizer_TwoScale_negFL` Asymmetric Multi-scale Dynamic Fixed Point Quantizer
- `DFPQuantizer_TwoScale_Symm` Symmetric Multi-scale Dynamic Fixed Point Quantizer
- `DFPQuantizer_negFL` Asymmetric Dynamic Fixed Point Quantizer，用max計算FL
- `DFPQuantizer_negFL_stdFL` Asymmetric Dynamic Fixed Point Quantizer，用std計算FL
### Wrapper
- `BaseLayerWrapper` Layer Wrapper base class
- `ConvLinearLayerWrapper` Layer Wrapper for Convolution and Linear
- `ConvLinearLayerWrapper_Fused` Layer Wrapper for Convolution and Linear with Batch Norm Fusing
- `BinaryLayerWrapper` Layer Wrapper for Convolution and Linear using binary quantization (BNN我很久沒維護了)
### QuantModel
- `QuantConfig` 量化設定容器class
- `QuantModel` 包裝模型的class本體
- `wrap_DFP_only()` 支援除了BNN的量化，如果要細緻調整量化時使用quantizer和bit數，請改這邊
- `wrap_binary()` 支援BNN，很久沒維護了
## 新模型支援SOP
1. 把 [quant_model_v6.py](quant_model_v6.py) 複製到`train.py`相同資料夾下
2. 根據模型架構修改以下變數，需自行import相關class
```python
OUTPUT_QUANTIZED_LAYERS = (FeatureConcat, Concat, WeightedFeatureFusion) # 除了conv/bn外，這些layer輸出會額外加上 output quantizer
ACTIVATION_LAYERS = (nn.LeakyReLU, nn.Mish, nn.ReLU, nn.ReLU6) # 這些layer會被替換為leakyReLU，並加上 output quantizer (目前已註解掉 QuantModel:wrap_DFP_only()中)
FL_THRESHOLD = 0.7 # 更新FL的threshold，不用改
# INPUT_BIAS = -0.5 # DLA或其他可設定輸入偏移的可用，使輸入變成-0.5~+0.5對稱，有利於DFP
INPUT_BIAS = 0.0 # 不加偏移，輸入一般為0~1之間
```
3. 修改`train.py`，以下參考YOLOR那包進行設定
- 引入`quant_model`
```python
import quant_model_v6.py as quant_model # import成quant_model
```
- 找到 model 建立好並吃完weight的地方
```python
# model建好吃完weight了
# Quant Model
quant_config = quant_model.QuantModel.QuantConfig() # 用這個設定量化參數，可以參考quant_model的code
quant_config.layer_index_not_to_quantize = [] # 手動指定不要量化的layer編號，以model.named_modules()順序為準
conv_list = [m for i, (n,m) in enumerate(model_ref.named_modules()) if isinstance(m, (nn.Conv2d, nn.Linear))] # 把所有的conv/linear抓出來列清單
quant_config.layer_index_keep_8_bits = [0] + [i for i,m in enumerate(conv_list) if m.bias is not None] # 指定第一層及輸出層維持8-bit
quant_config.layer_index_quantize_output = [i for i,m in enumerate(conv_list) if m.bias is not None] # 指定輸出層的輸出額外加上 output quantizer
quantizer = getattr(quant_model, opt.quantizer) # 從參數取得quantizer class
print("Using Quantizer: ", quantizer.__class__.__name__)
# 設定activation weight bias各自的quantizer class，None表示不用
quant_config.act_quantizer = None if opt.a_bit < 0 else quantizer # QBox或嵌入式平台不要量化activation，因為上板時對不起來
quant_config.wei_quantizer = None if opt.w_bit < 0 else quantizer
quant_config.bias_quantizer = None if opt.b_bit < 0 else quantizer
model = quant_model.QuantModel( # 包模型取代model
    model_ref, # 要包的模型，會被copy一份成QuantModel.module
    a_bw = opt.a_bit, w_bw = opt.w_bit, b_bw = opt.b_bit, # 指定各自需要的bit數
    act_disabled = opt.disable_act, # 這個不用下，直接透過act_quantizer = None就可以達成
    is_fused = opt.fuse, # 進行bn fusion
    quant_config = quant_config # 前面設定的QuantConfig
    ).to(device)
```
- `print(model)` # 印出來看看有沒有符合預期
```python
(Conv2d): ConvLinearLayerWrapper_Fused( # 這是包好的 layer wrapper
    freeze_bn_stat = False # 有沒有固定bn
    (module_to_wrap): Conv2d(3, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # 被包的layer
    (input_quantizer): # 沒有的話會沒有
    (output_quantizer): # 沒有的話會沒有
    (weight_quantizer): DFPQuantizer(bw = 8 , fl = 0 , abs_max = 0.0 , disable_act = False , freeze_fl = False) # quantizer類型和參數，disable_act不做事，freeze_fl表示fl凍結
    (bias_quantizer): DFPQuantizer(bw = 8 , fl = 0 , abs_max = 0.0 , disable_act = False , freeze_fl = False) # 剛建立好fl會是0，訓練時統計更新
    (bn_layer): BatchNorm2d(9, eps=0.0001, momentum=0.01, affine=True, track_running_stats=True) # batch norm 包在這裡
)
(BatchNorm2d): Identity() # batch norm 原本的位置替換成identity
(activation): LeakyReLU(negative_slope=0.125, inplace=True) # 我是activation function，有包量化會變成 BaseLayerWrapper
```
- 找到每個 epoch 開始的 for loop
```python
for epoch in range(start_epoch, epochs)
    model.train()
    freeze_fl = epoch >= opt.freeze_fl if opt.freeze_fl >=0 else False # 超過一定epoch凍結FL，預設opt.freeze_fl = 5
    freeze_bn = epoch >= opt.freeze_bn if opt.freeze_bn >=0 else False # 超過一定epoch凍結BN，預設opt.freeze_bn = 0
    model.set_freeze_act_fl(freeze_fl) # 設定 input/output quantizer freeze_fl 
    model.set_freeze_wei_fl(freeze_fl) # 設定 weight/bias quantizer freeze_fl
    model.set_freeze_bn_statistic(freeze_bn) # 設定BN凍結
```
- 在模型 **inference之前**加上`model.quantize_weight_before_forward()`
```python
    model.quantize_weight_before_forward() # 事先對weight/bias進行量化，feature map input/output會在runtime量化
    # Forward
    with amp.autocast(enabled=cuda and opt.mpt):
        pred = model(imgs)  # forward
```
- 在模型 **inference之後，step之前**加上`model.restore_weight_after_backward()`
```python
    # Backward
    scaler.scale(loss).backward() # 計算梯度
    model.restore_weight_after_backward() # 在套用梯度更新之前先將weight/bias恢復成未量化的數值
    if epoch > 0: # skipping first epoch to wash BN
        # Optimize
        if ni % accumulate == 0:
            # gradient clipping
            if opt.clip > 0: # opt.clip = 10 效果不錯，如果模型發散可以使用
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip, error_if_nonfinite=False)
            scaler.step(optimizer)  # optimizer.step # 實際套用梯度更新
            scaler.update()
            optimizer.zero_grad()
```
- **test部分**：同樣對模型inference前後加上量化與復原
```python
model.quantize_weight_before_forward() # 事先對weight/bias進行量化，feature map input/output會在runtime量化 
results, maps, times = test.test(opt.data,
                            batch_size=batch_size*2,
                            imgsz=imgsz_test,
                            # model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                            model=model,
                            single_cls=opt.single_cls,
                            dataloader=testloader,
                            save_dir=save_dir,
                            plots=plots and final_epoch,
                            log_imgs=opt.log_imgs if wandb else 0,
                            verbose=True)
model.restore_weight_after_backward() # 在套用梯度更新之前先將weight/bias恢復成未量化的數值
```
- 儲存時記得連同QuantModel一起存起來，這樣才會保留量化參數
- 參考 YOLOR 那包的 argparse 補上用到的參數即可
4. 修改 `test.py`，在讀入模型後對模型進行量化，建議可以用 `model.fuse_batchnorm()` 把BN融合後再測試
```python
# load model
ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
if ckpt['model'].__class__.__name__ == "QuantModel":
    model = ckpt['model'].to(device).eval()
    if opt.fuse: model.fuse_batchnorm() # BN融合
    model.quantize_weight_before_forward() # 量化
    with open(save_dir / "opt.txt", "wt") as f: # 我習慣把模型跟訓練下的參數都存在opt.txt中
        f.write("python " + " ".join(sys.argv)+"\n")
        f.write(str(model)+"\n") # 這樣就可以看到模型架構跟量化參數了
        f.write(model.wrapper_info() + "\n") # 比較漂亮的印法
    print("Disable Act: ", model.act_disabled)
```
5. 修改 `convert2onnx.py`，在讀入模型後對模型進行量化和BN融合和拆包
```python
def load_model(weights, device, opt_dir = None):
    ckpt = torch.load(weights[0] if isinstance(weights, list) else weights, map_location='cpu')
    model = ckpt['model'].to(device).float().eval()
    if opt_dir:
        with open(opt_dir / "opt.txt", "wt") as f:
            f.write("python " + " ".join(os.sys.argv)+"\n")
    if model.__class__.__name__ == "QuantModel":
        print("Disable Act: ", model.act_disabled)
        if opt_dir:
            with open(opt_dir / "opt.txt", "at") as f:
                f.write(model.wrapper_info() + "\n")
                f.write(str(model)+"\n")
    model.fuse_batchnorm() # BN 融合
    model.quantize_weight_before_forward() # 量化模型
    model.unwrap() # 拆掉 layer wrapper
    return model
```
