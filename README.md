# llama-pytorch
Implement Meta Llama in pytorch

``` python
import torch
from llama import LlamaModel
from llama import LlamaConfig
from transformers import BertTokenizerFast


def test_llama_model():
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizerFast('./vocab.txt')

    text = "hello 你好 こんにちは 안녕하세요"
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor(input_ids).long().unsqueeze(0).to(device)

    llama = LlamaModel(config=LlamaConfig(vocab_size=tokenizer.vocab_size))
    output = llama(input_ids)
    print(output.shape)


if __name__ == '__main__':
    test_llama_model()
```
