FIX_UNKNOWN_PROMPT = """\
You are helping debug a script (`export_model_excel.py`) that parses a YOLOv7
QuantModel text dump and extracts quantizer FL values into an Excel file.

The script works by matching layer/component names against known string patterns.
When a component cannot be matched, it outputs `role_type = unknown`.

The following nodes produced `unknown` role_type:

<unknown_nodes>
{unknown_nodes}
</unknown_nodes>

Here is the relevant portion of the model dump for context:

<model_txt_excerpt>
{model_txt_excerpt}
</model_txt_excerpt>

For each unknown node, analyze what it is (conv weight, bias, activation output, etc.)
and suggest what string pattern should be added to the parsing logic.

Return a JSON list — one entry per unknown node:
[
  {{
    "layer": "<layer identifier from the unknown node>",
    "component": "<component name>",
    "role_type": "<what this quantizer actually is: input/output/weight/bias>",
    "suggested_pattern": "<the substring to match in the model dump line>",
    "reason": "<brief explanation>"
  }},
  ...
]

Return ONLY the JSON list with no extra text.
"""
