from os.path import dirname, abspath
from pathlib import Path

import numpy as np
import onnxruntime
import transformers


def tf_example(texts, model_name='M-CLIP/XLM-Roberta-Large-Vit-L-14'):
    from multilingual_clip import tf_multilingual_clip

    model = tf_multilingual_clip.MultiLingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    inData = tokenizer.batch_encode_plus(texts, return_tensors='tf', padding=True)
    embeddings = model(inData)
    print(embeddings.shape)


def pt_example(texts, model_name='M-CLIP/XLM-Roberta-Large-Vit-L-14'):
    from multilingual_clip import pt_multilingual_clip

    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    txt_tok = tokenizer(texts, padding=True, return_tensors='pt')

    embeddings = model.forward(**txt_tok)
    print(embeddings.shape)


def onnx_example(texts, onnx_file, model_name='M-CLIP/XLM-Roberta-Large-Vit-L-14'):
    providers = ['TensorrtExecutionProvider',
                 'CUDAExecutionProvider',
                 'CPUExecutionProvider']
    textual_session = onnxruntime.InferenceSession(str(onnx_file), providers=providers)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    txt_tok = tokenizer(texts, padding=True, return_tensors='np')

    txt_tok = {
        'input_ids': txt_tok['input_ids'].astype(np.int64),
        'attention_mask': txt_tok['attention_mask'].astype(np.int64),
    }

    textual_output = textual_session.run(output_names=['output'], input_feed=txt_tok)
    print(textual_output)


if __name__ == '__main__':
    exampleTexts = [
        'Three blind horses listening to Mozart.',
        'Älgen är skogens konung!',
        'Wie leben Eisbären in der Antarktis?',
        'Вы знали, что все белые медведи левши?'
    ]

    # tf_example(exampleTexts)
    # pt_example(exampleTexts)
    mclip_textual = Path(dirname(abspath(__file__))) / 'onnx-out' / 'mclip_textual.onnx'
    onnx_example(exampleTexts, mclip_textual)
