from collections import OrderedDict
from os import makedirs
from os.path import abspath, dirname
from pathlib import Path

import transformers
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers.onnx import OnnxConfig
from transformers.onnx import export

from multilingual_clip import pt_multilingual_clip


def quantization(input_file: Path, output_file: Path):
    quantize_dynamic(input_file, output_file, weight_type=QuantType.QUInt8)


class BertOnnxConfig(OnnxConfig):
    @property
    def inputs(self):
        return OrderedDict(
            [
                ("input_ids", {0: "batch_size", 1: "sequence"}),
                ("attention_mask", {0: "batch_size", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self):
        return OrderedDict(
            [
                ("output",  {0: "batch_size", 1: "sequence"})
            ]
        )


if __name__ == '__main__':
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'

    BASE_DIR = Path(dirname(abspath(__file__)))

    filename = 'mclip_textual.onnx'
    output_dir = BASE_DIR / 'onnx-out'
    quantified_dir = BASE_DIR / 'onnx-quant'

    output_quantified_onnx = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)

    onnx_config = BertOnnxConfig(model.config)
    dummy_input = onnx_config.generate_dummy_inputs(preprocessor=tokenizer)
    makedirs(output_dir)
    export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, output_dir / filename)

    if output_quantified_onnx:
        makedirs(quantified_dir)
        quantization(output_dir / filename, quantified_dir / filename)

