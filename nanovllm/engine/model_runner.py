import atexit
from ctypes import Union
import pickle
from typing import List
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.misslayer import MissLayer
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.engine.afd_transfer.afd_factory import AFDConnectorFactory

class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.afd_config = config.afd_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化全局分布式环境
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        attn_ranks = list(range(self.afd_config.num_attention_servers))
        self.attn_ranks_group = dist.new_group(ranks=attn_ranks)

        # 初始化afd连接器
        self.afd_connector = None
        if self.afd_config is not None:
            self.afd_connector = AFDConnectorFactory.create_connector(
                self.rank, self.rank, self.afd_config
            )

        self.model = Qwen3ForCausalLM(hf_config, self.afd_connector)

        load_model(self.model, config.model)
        # 采样器，用于token生成时采样
        self.sampler = Sampler()

        self.warmup_model()
        # 分配kv缓存
        self.allocate_kv_cache()
        if not self.enforce_eager:
            # 放在warmup之后是为了避免捕获init等一次性开销
            self.capture_cudagraph()
        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.afd_config.num_attention_servers > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.afd_config.num_attention_servers > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        self.afd_connector.send_shutdown_signal()
        dist.destroy_process_group(self.attn_ranks_group)
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.afd_config.num_attention_servers > 1 and self.rank > 0
        self.event.wait() # 等待set
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        # 只使用GPU1 写入数据
        assert self.afd_config.num_attention_servers > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 通知其他进程写入完毕
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.afd_config.num_attention_servers > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        # 刷新显存状态
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 执行一次前向推理
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 已分配的显存总量
        # 每张卡负责的 kv 头数量
        # num_kv_heads = hf_config.num_key_value_heads // self.world_size
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算所有层所需的 kv cache 显存大小，单位字节
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # peak + current 是指
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 将kvcache分配到每一层的模块中
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        # 用-1填充block使得一个batch里面的seqs具有相同的block table长度. 为了合成GPU tensor
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # prefill 阶段需要处理没有缓存进入kvcache的部分
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 计算实际参与计算的序列长度
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen # key 全部都需要
            # 累计的序列长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) 
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # slot_mapping 构建
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                # 记录当前block的token对应的slot位置
                slot_mapping.extend(list(range(start, end)))
        # 有部分缓存已经进入kvcache，需要准备block tables. 目的: 需要准备 block_tables 来告诉 kernel 哪些 block 存着旧的 KV 数据。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 这里-1是因为从0开始
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)
        return logits
    
        '''
        TODO 使用cuda graph
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 如果batch size或者是prefill，直接进行模型计算
            hidden_states = self.model(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
            return logits
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 选择合适的graph bs
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 根据当前的token的实际值，更新图中的输入变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
        '''

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 进行kvcache的处理
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 为所有seqs准备对应的采样温度
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # 将概率分布采样为token ids
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 清理前一次运行的上下文信息
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # 捕获一次 kernel 调用序列 → replay 时直接在 GPU 执行 → 避免 Python 调度 → 大幅降低延迟
        """
        1. 初始化张量, 包括 input_ids, positions, slot_mapping, context_lens, block_tables, outputs
        2. 遍历不同的 batch size, 捕获对应的 CUDA Graph
        3. 将捕获的 CUDA Graph 存储在 self.graphs 中，以备后续重用
        4. 存储用于图重放的变量
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        # PagedAttention 的页表最大数量
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        # 在 KV Cache 中对应的槽位（slot）索引
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        # 已经处理的 上下文长度（context length）
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            # 第一次的时候为none，后续其他的bs可以复用同一个pool
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        # cuda graph replay使用相同的tensor地址，存储起来，可以修改 tensor 内容
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

class ModelFFNRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.afd_config = config.afd_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        
        self.shutdown_event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        ffn_ranks = list(range(self.afd_config.num_ffn_servers))
        self.ffn_ranks_group = dist.new_group(ranks=ffn_ranks)

        self.afd_connector = None
        if self.afd_config is not None:
            self.afd_connector = AFDConnectorFactory.create_connector(
                self.rank, self.rank, self.afd_config
            )

        self.model = Qwen3ForCausalLM(hf_config, self.afd_connector)
        load_model(self.model, config.model)
        
        # TODO: 恢复 cuda graph
        
        # 恢复 CPU 默认设置，防止后续无关操作误用 GPU
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        atexit.register(self.exit)

        try:
            self.ffn_loop()
        except Exception as e:
            print(f"[Rank {self.rank}] Fatal Error: {e}")
            raise e
        finally:
            self.exit()

    def ffn_loop(self):        
        while True:
            try:
                self.execute_model()
            except KeyboardInterrupt:
                break
            except StopIteration:
                print(f"[Rank {self.rank}] Received shutdown signal.")
                break
            except Exception as e:
                print(f"[Rank {self.rank}] Loop Error: {e}")
        
    def execute_model(self):
        hidden_states, metadata = self.afd_connector.recv_attn_output()

        if hidden_states is None and metadata.shutdown:
            raise StopIteration
        
        layer_idx = metadata.layer_idx
        ffn_output = self.model.model.layers[layer_idx].compute_mlp(hidden_states)
        self.afd_connector.send_ffn_output(ffn_output, metadata)

    def exit(self):
        dist.destroy_process_group(self.ffn_ranks_group)
        if dist.is_initialized():
            dist.destroy_process_group()
        
        
