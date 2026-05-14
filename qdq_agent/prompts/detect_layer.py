DETECT_LAYER_PROMPT = """\
You are analyzing a YOLOv7 quantized model architecture dump produced by `print(model)`.

The model is a QuantModel wrapping a YOLO network. Layers are numbered sequentially,
printed as:
    (0): Conv(...)
    (1): Conv(...)
    ...
    (N): IDetect(...)   ← this is what we are looking for

Your task: find the layer index of the **IDetect** (or Detect) layer — the final
detection head. It is recognizable by:
- Its class name contains "IDetect", "Detect", or "IKeypoint"
- It contains sub-modules named m.0, m.1, m.2 (the detection conv heads)
- It contains sub-modules named im.0, im.1, im.2 (implicit functions, ImplicitA/ImplicitM)
- It is typically the last numbered layer in the model

Model dump:
<model_txt>
{model_txt}
</model_txt>

Return ONLY a JSON object with no extra text:
{{"detect_layer": <integer>}}
"""
