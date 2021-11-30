from typing import List, Union
from transformers import AutoConfig
from transformers.models.bert.configuration_bert import BertConfig


class BertLongConfig(BertConfig):
    model_type = "bert-long"

    def __init__(
        self,
        attention_window: Union[List[int], int] = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attention_window = attention_window
