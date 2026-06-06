import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Set, List

# Try to import real Mamba, fallback to Dummy if not compiled/installed
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class DummyMambaSSM(nn.Module):
    """
    Placeholder for the core Mamba state-space scan to allow testing without CUDA compilation.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class ConceptPerceptron(nn.Module):
    """Global context pooling mechanism mapping the input sequence into a condensed latent prefix."""
    def __init__(self, d_model: int, num_tokens: int = 16, chunk_size: int = 1024) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = chunk_size
        self.avg_pooling = nn.AdaptiveAvgPool1d(num_tokens)
        self.max_pooling = nn.AdaptiveMaxPool1d(num_tokens)
        self.proj = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        chunks: List[torch.Tensor] = []
        
        # Handle chunked inference natively for context windows > 1024 tokens
        for i in range(0, L, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size, :]
            chunk_t = chunk.transpose(1, 2)
            avg_pool = self.avg_pooling(chunk_t).transpose(1, 2)
            max_pool = self.max_pooling(chunk_t).transpose(1, 2)
            
            # Suction-Cup Granular Anchoring
            pooled_chunk = torch.cat([avg_pool, max_pool], dim=-1)
            chunks.append(pooled_chunk)
        
        aggregated = torch.stack(chunks, dim=0).mean(dim=0)
        return F.silu(self.proj(aggregated))

class LowRankBridge(nn.Module):
    """Bottleneck compression bridge routing into auxiliary reasoning engines."""
    def __init__(self, d_model: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))

class MambaLayer(nn.Module):
    """Mamba layer enforcing strict bfloat16 precision."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.ssm = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.ssm = DummyMambaSSM(d_model=d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        
        # Mandatory precision constraint: SSM strictly evaluated in bfloat16
        device_type = x.device.type if x.device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            x_ssm = self.ssm(x_norm)
            
        return x_ssm.to(x.dtype) + residual

class Mamba3Leviathan(nn.Module):
    """Mamba 3 Leviathan architecture with 8 parallel latent forcing arms and Gradient Checkpointing."""
    # LEVIATHAN DEFAULTS: 310M parameters designed for 12GB VRAM
    def __init__(self, vocab_size: int = 50304, d_model: int = 1024, n_layers: int = 32, mimo_paths: int = 8, use_gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mimo_paths = mimo_paths
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cp = ConceptPerceptron(d_model)
        self.thalamic_primer = MambaLayer(d_model)
        self.bridge = LowRankBridge(d_model)
        
        # Main Sequential Backbone (Deep)
        self.layers = nn.ModuleList([MambaLayer(d_model) for _ in range(n_layers)])
        
        # MIMO Engine: Parallel Latent Reasoning Chains (8 paths)
        self.mimo_reasoning_blocks = nn.ModuleList([MambaLayer(d_model) for _ in range(mimo_paths)])
        
        # Latent IPC (Cross-Talk)
        self.ipc_mixer = nn.Linear(d_model * mimo_paths, d_model * mimo_paths)
        
        # Synaptic Dam: Gated Tanh for global context injection
        self.cp_gate = nn.Parameter(torch.tensor(0.01))
        
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying
        
        # Phase 3j: Per-arm competitive vector router (Scaled to 8 arms)
        self.domain_router = nn.Linear(self.d_model, self.mimo_paths, bias=True)
        nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.domain_router.bias)
        
        self.last_telemetry = {
            'arm_collapse_metric': 0.0,
            'latent_energy': 0.0,
            'gate_score': 0.0,
            'primer_delta': 0.0
        }
        
        # Zero-Init Thalamic Primer for Identity Pass-through
        if hasattr(self.thalamic_primer.ssm, 'out_proj'):
            nn.init.zeros_(self.thalamic_primer.ssm.out_proj.weight)
        elif hasattr(self.thalamic_primer.ssm, 'proj'):
            nn.init.zeros_(self.thalamic_primer.ssm.proj.weight)
        
    def initialize_asymmetric_arms(self) -> None:
        """Asymmetric Initialization for all 8 MIMO arms."""
        for name, param in self.mimo_reasoning_blocks.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif param.dim() == 1:
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * 0.05)
        
    def forward(self, input_ids: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        orig_embs = self.embedding(input_ids)
        
        decay_factor = 0.7 ** loop_idx
        x = orig_embs * decay_factor
        
        cp_scratchpad = self.cp(x)
        
        # A. The Thalamic Primer
        primer_out = self.thalamic_primer(orig_embs)
        
        # Blend Primer into main stream (10% injection)
        x = orig_embs + primer_out * 0.1
        
        # Phase 3j: Temporal Vector Gating (8 Arms)
        with torch.no_grad():
            primer_delta = (primer_out - orig_embs).norm(dim=-1).mean()
            # Keep routing outside autograd to avoid interfering with arm gradients
            route_logits = self.domain_router(primer_out.detach())  # (B, L, 8)

        # Softmax forces arms to COMPETE for each token
        competitive_weights = F.softmax(route_logits, dim=-1)  # (B, L, 8)
        
        # 1. Hard Binary Mask
        top_indices = competitive_weights.argmax(dim=-1, keepdim=True)
        mask = torch.zeros_like(competitive_weights).scatter_(-1, top_indices, 1.0)
        
        # 2. Straight-Through Estimator
        hard_weights = mask - competitive_weights.detach() + competitive_weights

        # 3. Octopoda Trickle-Charge: clamp minimum per arm to prevent synaptic atrophy
        route_weights = torch.clamp(hard_weights, min=0.05)

        # 4. Renormalize so weights sum to 1.0
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)

        bridge_out = self.bridge(x)
        
        parallel_states = []
        autotomic_gates_list = []
        for i in range(self.mimo_paths):
            state = self.mimo_reasoning_blocks[i](bridge_out)
            
            # Autotomic pruning
            variance = state.var(dim=-1).mean()
            autotomic_gate = torch.clamp(torch.sigmoid((10.0 - variance) * 0.5), min=0.05)
            autotomic_gates_list.append(autotomic_gate.item() if isinstance(autotomic_gate, torch.Tensor) else autotomic_gate)
            
            arm_weight = route_weights[..., i:i+1]  # (B, L, 1)

            # Arm 0 gets full gradient
            if i == 0:
                parallel_states.append(state * autotomic_gate)
            else:
                parallel_states.append(state * arm_weight * autotomic_gate)
                
        mean_gate = route_weights[..., 1:].mean().item()
        
        # Telemetry
        if not self.training or (self.training and torch.rand(1).item() < 0.05):
            with torch.no_grad():
                self.last_telemetry.update({
                    'entropy': primer_delta.item() if isinstance(primer_delta, torch.Tensor) else primer_delta,
                    'gate_score': mean_gate,
                    'autotomic_gates': autotomic_gates_list,
                    'route_weights': route_weights.mean(dim=(0, 1)).tolist()
                })
        
        # Latent IPC Cross-Talk
        ipc_in = torch.cat(parallel_states, dim=-1)
        ipc_out = self.ipc_mixer(ipc_in)
        final_states = torch.split(ipc_out, self.d_model, dim=-1)
            
        x = x + (sum(final_states) / self.mimo_paths)
        
        # Backbone execution with gradient checkpointing
        for i, layer in enumerate(self.layers):
            if self.training and self.use_gradient_checkpointing:
                # Wrap layer execution in checkpoint to save activation memory
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
            # Deep Injection: global context residually injected every 6 layers
            if (i + 1) % 6 == 0:
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True)
                x = x + (self.cp_gate * torch.tanh(global_ctx))
                
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

