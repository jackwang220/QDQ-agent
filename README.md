---
title: 'Q_craft part# ezQuant-AMS Read'

---

具體轉檔流程可以參考run.sh


**Q_craft part**:

check for ezQuant-AMS Readme

**additional arguments** : 
    wrap_all : 我用來區分我的insert quantizer node跟學長的 如果下了這個的話就不會只抓conv跟act,會盡量把q craft中有提到的layer都抓
    
    
**export** : 
    model : export_quant_fused.py : model 如果unwrap()完之後，會把quantizer wrapper拔掉，因此要先把weights給還原進去，而這裡我會直接把int value給進去(args : ezquant_int8)
    excel : export_model_excel.py : 訓練好的model會包含每一個quantizer應該要有的fl value，我會將model的arch作為input txt來parse成一份excel file來讓後續做quantizer insertion跟方便debug
    
    
**datatype fp32->int8** : model_fp32_int8.py 在這邊我會把轉出來的onnx model進行datatype的轉換 確保node裡面含有的資訊都是int8

**insert quantizer** : onnx_view_wrap_all_repconv.py 在這份檔案裏面主要是針對model去把excel內容mapping到對應的onnx node.
                   implicit.py 之前implicit沒寫好 把它獨立成一個檔案去處理。

**post_process** : modify_model.py 主要是處理輸出格式，加一些transpose node讓輸出符合需求，如果前面有漏處理的node，這邊也提供可以直接用fl跟node name 去直接看圖insert的function，因此可以處理一些前面漏掉的部分，或是對onnx model增加輸出節點去做debug，模型的input bias也是在這裡加。

裡面大致上分幾個部分
1. analyze model structure : 會traverse整個onnx model graph並抓出那些node type並分類
2. map_fl_values_to_nodes : 根據node_type將excel中的fl value對應上去
3. add_quantization_nodes : 實際將quant/dequant node 加到onnx model上



**SOP** : 
1. 可以先跑一次原本的pytorch model包上q-craft,檢查一下有哪些節點是沒有被包上wrapper，並針對那些wrapper修改qcraft.py(quant_model_v7_revised.py)中的wrap_all_layer()，具體要包的位置以onnx為準，盡量確保每一個node後面都有quantizer，具體的話把pytorch model print出來確認就好，原則上至少要確保conv的所有input output都必須要經過quantizer。
2. 經過export_model_excel時注意一下有沒有role_type是unknown的出現，針對那些要補上對應的字串處理去抓
3. export_quant_fused就記得要下ezquant_int8去讓model在參數部分是維持整數的。具體其他repo的export基本上就是像這份去加入quantmodel的loading部分 對應著改
4. model_fp32_int8不用動，應該不太會有要更改的地方
5. onnx_view_wrap_all_repconv.py 基本上會對應到前面的export_model_excel，有針對新的wrapper做處理的話，就根據你設定的node_type去對我上面提到的三個函式做補充
6. modify_model就記得要看一下輸出點名字 改一下要處理的node
7. 正常這樣做完就可以去跑onnx inference&scoring，理論上因為rounding error不可能得到跟pytorch一樣的數值，但quantize過後不應該與pytorch有太大的差距。