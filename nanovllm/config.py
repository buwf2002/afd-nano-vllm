import os
from dataclasses import dataclass, field
from transformers import AutoConfig
from typing import Literal

@dataclass
class AFDConfig:
    afd_connector: str = "dummy"                         # Transport backend: 'dummy', 'stepmesh'
    afd_role: Literal["attention", "ffn"] = "attention"  # Server role
    afd_port: int = 1239                                 # Communication port
    afd_host: str = "127.0.0.1"                          # Host address
    num_afd_stages: int = 1                              # Number of pipeline stages
    num_attention_servers: int = 1                       # Number of attention servers
    num_ffn_servers: int = 1                             # Number of FFN servers
    afd_server_rank: int = 0                             # Server rank identifier

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 2
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    afd_config: AFDConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1


    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
