from dataclasses import dataclass
import torch
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from nanovllm.config import AFDConfig
import torch.distributed as dist

@dataclass
class AFDConnectorMetadata:
    shape: torch.Size | tuple[int, ...] = None  # Tensor shape
    layer_idx: int = 0                          # Layer index for computation
    stage_idx: int = 0                          # Pipeline stage index  
    dtype: torch.dtype | None = None            # Tensor data type
    device: torch.device | None = None          # Device
    shutdown: bool = False                     # Shutdown signal

class AFDConnectorBase(ABC):
    # Attention Worker Interface
    @abstractmethod
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata, *kargs) -> Any:
        pass

    @abstractmethod  
    def recv_ffn_output(self, timeout_ms: Optional[int] = None) -> torch.Tensor:
        pass
    
    # FFN Server Interface
    @abstractmethod
    def recv_attn_output(self, timeout_ms: Optional[int] = None) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        pass

    @abstractmethod
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: AFDConnectorMetadata) -> None:
        pass

    @abstractmethod
    def send_shutdown_signal(self):
        pass

class DummyAFDConnector(AFDConnectorBase):
    def __init__(self, rank: int, afd_config: AFDConfig):
        self.rank = rank
        self.afd_config = afd_config
        
        self.attn_serve_rank = 0
        self.ffn_serve_rank = self.attn_serve_rank + self.afd_config.num_attention_servers

        self.is_attn_server = (rank < self.afd_config.num_attention_servers)
        self.is_ffn_server = (rank >= self.afd_config.num_attention_servers)

        self.device = torch.device(f"cuda:{rank}")
        
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata, *kargs) -> Any:
        pass

    def recv_ffn_output(self, timeout_ms: Optional[int] = None) -> torch.Tensor:
        pass

    def recv_attn_output(self, timeout_ms: Optional[int] = None) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        pass

    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: AFDConnectorMetadata) -> None:
        pass

    def send_shutdown_signal(self):
        pass

class NaiveSyncAFDConnector(AFDConnectorBase):
    # TODO 超时等待卡死
    def __init__(self, rank: int, afd_config: AFDConfig):
        self.rank = rank
        self.afd_config = afd_config
        
        self.attn_serve_rank = 0
        self.ffn_serve_rank = self.attn_serve_rank + self.afd_config.num_attention_servers

        self.is_attn_server = (rank < self.afd_config.num_attention_servers)
        self.is_ffn_server = (rank >= self.afd_config.num_attention_servers)

        self.device = torch.device(f"cuda:{rank}")
        
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata, *kargs) -> Any:

        metadata.shape = hidden_states.shape
        metadata.dtype = hidden_states.dtype
        metadata.device = self.device
        
        dist.send_object_list([metadata], dst=self.ffn_serve_rank)
        
        dist.send(tensor=hidden_states, dst=self.ffn_serve_rank)
        

    def recv_ffn_output(self, timeout_ms: Optional[int] = None) -> torch.Tensor:

        object_list = [None] # 占位符
        dist.recv_object_list(object_list, src=self.ffn_serve_rank)
        metadata: AFDConnectorMetadata = object_list[0]

        output_shape = metadata.shape
        dtype = metadata.dtype
        
        recv_buffer = torch.empty(output_shape, dtype=dtype, device=self.device)
        
        dist.recv(tensor=recv_buffer, src=self.ffn_serve_rank)
        
        return recv_buffer, metadata

    def recv_attn_output(self, timeout_ms: Optional[int] = None) -> tuple[torch.Tensor, AFDConnectorMetadata]:

        object_list = [None] # 占位符
        dist.recv_object_list(object_list, src=self.attn_serve_rank)
        metadata: AFDConnectorMetadata = object_list[0]

        if metadata.shutdown:
            return None, metadata

        recv_buffer = torch.empty(metadata.shape, dtype=metadata.dtype, device=self.device)
        
        dist.recv(tensor=recv_buffer, src=self.attn_serve_rank)
        
        return recv_buffer, metadata

    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: AFDConnectorMetadata) -> None:

        metadata.shape = ffn_output.shape
        metadata.dtype = ffn_output.dtype
        metadata.device = self.device
        
        # 2. 发送 Metadata
        dist.send_object_list([metadata], dst=self.attn_serve_rank)
        
        # 3. 发送 Tensor
        dist.send(tensor=ffn_output, dst=self.attn_serve_rank)

    def send_shutdown_signal(self):
        metadata = AFDConnectorMetadata(shutdown=True)
        dist.send_object_list([metadata], dst=self.ffn_serve_rank)