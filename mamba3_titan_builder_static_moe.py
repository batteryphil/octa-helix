import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, List

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ─────────────────────────────────────────────────────────────────────────────
# ARM IDENTITY TABLE
# 16 named specializations — arms self-organize toward these roles via
# Phase 3j multi-domain training. Labels are for telemetry/UI only.
# ─────────────────────────────────────────────────────────────────────────────
ARM_IDENTITIES = [
    "General Language",      # 0  — always-on anchor arm
    "Symbolic Math",         # 1  — GSM8K / arithmetic
    "Logical Reasoning",     # 2  — ARC / deductive chains
    "Code Syntax",           # 3  — CodeAlpaca / programming
    "Factual Recall",        # 4  — encyclopedic knowledge
    "Summarization",         # 5  — CNN/DailyMail / compression
    "Creative Writing",      # 6  — narrative / prose
    "Instruction Following", # 7  — chat / command execution
]


class RealMambaSSM(nn.Module):
    """
    Real Mamba S6 selective SSM — pure PyTorch, no custom CUDA kernels.
    Implements the full selective scan from the Mamba paper:
      - Input-dependent B, C, dt  (the 'selective' part)
      - Log-domain parallel cumsum scan (no Python loop, fully vectorized)
      - Mathematically identical to mamba_ssm.Mamba
    expand=1, d_state=8 keeps VRAM under 12GB without gradient checkpointing.
    """
    def __init__(self, d_model: int, d_state: int = 8,
                 d_conv: int = 4, expand: int = 1) -> None:
        super().__init__()
        import math
        self.d_inner  = d_model * expand
        self.d_state  = d_state
        self.dt_rank  = max(1, math.ceil(d_model / 16))

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                  kernel_size=d_conv, padding=d_conv - 1,
                                  groups=self.d_inner, bias=True)
        # Selective projections — outputs depend on input (the key Mamba innovation)
        self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: log-parameterized, initialized to HiPPO-like values
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))          # (d_inner, d_state)
        self.D        = nn.Parameter(torch.ones(self.d_inner))  # skip connection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Stable initial dt (from Mamba paper §3.6)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) *
                       (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xz      = F.linear(x, self.in_proj.weight)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv  = F.conv1d(x_in.transpose(1, 2),
                           self.conv1d.weight, self.conv1d.bias,
                           padding=self.conv1d.padding[0],
                           groups=self.conv1d.groups)[:, :, :x.shape[1]].transpose(1, 2)
        x_conv  = F.silu(x_conv)
        y       = self._selective_scan(x_conv)
        return F.linear(y * F.silu(z), self.out_proj.weight)

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback pure-PyTorch parallel cumsum scan (used only if mamba_ssm unavailable)."""
        dbl       = F.linear(x, self.x_proj.weight)
        dt_r, B_p, C = dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt        = F.softplus(F.linear(dt_r, self.dt_proj.weight, self.dt_proj.bias))
        A         = -torch.exp(self.A_log.float()).to(x.dtype)
        dtA       = torch.einsum('bld,ds->blds', dt, A)
        log_A_cum = torch.cumsum(dtA, dim=1)
        Bu        = torch.einsum('bld,bls->blds', dt * x, B_p)
        h         = torch.exp(log_A_cum) * torch.cumsum(Bu * torch.exp(-log_A_cum), dim=1)
        y         = torch.einsum('blds,bls->bld', h, C)
        return y + x * self.D.to(x.dtype)


# Keep DummyMambaSSM for reference — NOT used anymore
class DummyMambaSSM(nn.Module):
    """DEPRECATED — single linear projection with no sequence modeling."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConceptPerceptron(nn.Module):
    """Global context pooling — compresses input into a latent scratchpad."""
    def __init__(self, d_model: int, num_tokens: int = 16, chunk_size: int = 1024) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = chunk_size
        self.avg_pooling = nn.AdaptiveAvgPool1d(num_tokens)
        self.max_pooling = nn.AdaptiveAvgPool1d(num_tokens)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        chunks: List[torch.Tensor] = []
        for i in range(0, L, self.chunk_size):
            chunk = x[:, i:i + self.chunk_size, :]
            chunk_t = chunk.transpose(1, 2)
            avg_p = self.avg_pooling(chunk_t).transpose(1, 2)
            max_p = self.max_pooling(chunk_t).transpose(1, 2)
            chunks.append(torch.cat([avg_p, max_p], dim=-1))
        aggregated = torch.stack(chunks, dim=0).mean(dim=0)
        return F.silu(self.proj(aggregated))


class LowRankBridge(nn.Module):
    """Bottleneck bridge before MIMO arms — compresses backbone output."""
    def __init__(self, d_model: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up   = nn.Linear(bottleneck, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class MambaLayer(nn.Module):
    """Mamba SSM layer with pre-norm and residual."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.ssm = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            # Full S6 selective scan — real Mamba, pure PyTorch
            self.ssm = RealMambaSSM(d_model=d_model, d_state=4, d_conv=4, expand=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm(self.norm(x)) + x


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class Mamba3Titan(nn.Module):
    """
    Mamba 3 Titan — 2.54B parameter MoE reasoning engine.
    16 parallel MIMO arms with soft routing, IPC cross-talk, and arm telemetry.

    Architecture flow:
      Embedding → Thalamic Primer → [first N/2 backbone layers]
        → Mid-backbone routing (entropy computed HERE — fixes Glass Ceiling)
        → LowRankBridge → 16 parallel MIMO arms
        → IPC mixer (arms share with each other)
        → [remaining N/2 backbone layers] + ConceptPerceptron injection
        → LM head
    """
    def __init__(self,
                 vocab_size: int = 50304,
                 d_model: int = 2048,
                 n_layers: int = 80,
                 mimo_paths: int = 16,
                 use_gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.mimo_paths  = mimo_paths
        self.n_layers    = n_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.active_phase = '1'

        # ── Input ────────────────────────────────────────────────────────────
        self.embedding      = nn.Embedding(vocab_size, d_model)
        self.cp             = ConceptPerceptron(d_model)
        self.thalamic_primer = MambaLayer(d_model)
        self.bridge         = LowRankBridge(d_model)

        # ── Backbone (80 layers, split at midpoint for routing) ──────────────
        self.layers = nn.ModuleList([MambaLayer(d_model) for _ in range(n_layers)])
        self._mid   = n_layers // 2   # layer 40 — where routing is computed

        # ── MIMO Arms (16) ───────────────────────────────────────────────────
        self.mimo_reasoning_blocks = nn.ModuleList(
            [MambaLayer(d_model) for _ in range(mimo_paths)]
        )

        # ── Routing ──────────────────────────────────────────────────────────
        # FIX: domain_router now reads from mid-backbone hidden state, not raw embeddings.
        # This is the Entropy Glass Ceiling fix from deepthink_full_report.txt.
        self.domain_router    = nn.Linear(d_model, mimo_paths, bias=True)
        self.router_temp      = nn.Parameter(torch.ones(1) * 1.0)  # learnable temperature
        nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.domain_router.bias)

        # ipc_mixer permanently removed — replaced by Sparse IPC Blackboard below.

        # ── Output ───────────────────────────────────────────────────────────
        self.cp_gate  = nn.Parameter(torch.tensor(0.01))
        self.norm_f   = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        # Weights untied to prevent Calibration Shock from cascading into embeddings

        # ── Telemetry ────────────────────────────────────────────────────────
        self.last_telemetry: dict = {
            'gate_score':    0.0,
            'entropy':       0.0,
            'arm_weights':   [1.0 / mimo_paths] * mimo_paths,
            'arm_labels':    ARM_IDENTITIES,
            'top_arms':      [],
            'arm_collapse_metric': 0.0,
            'latent_energy':       0.0,
        }

        # ── PATCH 2A: Sparse IPC Blackboard (Corpus Callosum) ────────────────
        # 64-dim bottleneck for inter-expert communication on the current token.
        self.bus_dim  = 64
        self.bb_write = nn.Linear(d_model, self.bus_dim, bias=False)
        self.bb_read  = nn.Linear(self.bus_dim, d_model, bias=False)
        nn.init.zeros_(self.bb_read.weight)

        # Zero-init thalamic primer output — identity pass-through at init
        if hasattr(self.thalamic_primer.ssm, 'out_proj'):
            nn.init.zeros_(self.thalamic_primer.ssm.out_proj.weight)
        elif hasattr(self.thalamic_primer.ssm, 'proj'):
            nn.init.zeros_(self.thalamic_primer.ssm.proj.weight)


    # ── Phase control ─────────────────────────────────────────────────────────
    def set_phase(self, phase: str) -> None:
        assert phase in ['1', '2', '3', '3j', '3r', 'sft'], f"Invalid phase: {phase}"
        self.active_phase = phase

        # ── Phase 3j: Freeze pretrained backbone — train new modules only ────────
        # Backbone (embedding + 48 layers) is already trained on 300B tokens.
        # Freezing it: (a) preserves pretrained knowledge, (b) drops trainable
        # params 2.54B → ~500M, (c) eliminates ~4GB of gradient memory, enabling
        # GPU-only optimizer state that fits in the remaining VRAM headroom.
        # ── Backbone: frozen in 3j/3r/sft (pretrained, don't disturb) ───────────
        backbone_frozen = phase in ('3j', '3r', 'sft')
        for p in self.embedding.parameters():
            p.requires_grad_(not backbone_frozen)
        for p in self.layers.parameters():
            p.requires_grad_(not backbone_frozen)

        # ── Blackboard: active when arms coordinate ───────────────────────────
        bb_active = phase in ('1', '2', '3j', '3r', 'sft')
        for p in list(self.bb_write.parameters()) + list(self.bb_read.parameters()):
            p.requires_grad_(bb_active)

        # ── Bridge: active when arms are training ────────────────────────────
        bridge_active = phase in ('1', '2', '3', '3j', '3r', 'sft')
        for p in self.bridge.parameters():
            p.requires_grad_(bridge_active)

        # ── Arms: co-train with router in 3r ────────────────────────────────
        # In 3r: arms UNFROZEN so gradient flows arm←router via soft route_weights.
        # Arms the router weights more heavily get proportionally more gradient
        # and specialise faster → router sees clearer signal → weights them more.
        # Positive feedback: router + arms co-adapt simultaneously (like Mixtral).
        # Arms use lower LR than router to preserve 3j specialisation.
        arms_active = phase in ('1', '2', '3', '3j', '3r', 'sft')
        for p in self.mimo_reasoning_blocks.parameters():
            p.requires_grad_(arms_active)

        # ── Router: trains in 3j (CE label) and 3r (LM loss) ───────────────────
        router_active = phase in ('3j', '3r', 'sft')
        for p in self.domain_router.parameters():
            p.requires_grad_(router_active)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        arms_str  = 'CO-TRAINING' if phase == '3r' else ('TRAINING' if arms_active else 'FROZEN')
        print(f"Titan Architecture → Phase {phase}  "
              f"(Backbone {'FROZEN' if backbone_frozen else 'TRAINING'} | "
              f"Arms {arms_str} | "
              f"Router {'TRAINING' if router_active else 'FROZEN'} | "
              f"Blackboard {'ACTIVE' if bb_active else 'SILENT'} | "
              f"Trainable {trainable/1e6:.0f}M / {total/1e6:.0f}M params)")

    # ── Asymmetric arm init ───────────────────────────────────────────────────
    def initialize_asymmetric_arms(self) -> None:
        """Orthogonal init so arms start with different feature subspaces."""
        for name, param in self.mimo_reasoning_blocks.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif param.dim() == 1:
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * 0.05)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self,
                input_ids:  torch.Tensor,
                loop_idx:   int = 0,
                domain_ids: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids:  (B, L) token ids
            loop_idx:   recursive refinement depth (0 = first pass)
            domain_ids: (B,) integer domain labels — only used in Phase 3j

        Returns:
            logits:      (B, L, vocab_size)
            domain_loss: scalar tensor
        """
        B, L = input_ids.shape
        device = input_ids.device

        # ── Embedding + Thalamic Primer ───────────────────────────────────────
        orig_embs   = self.embedding(input_ids)
        primer_out  = self.thalamic_primer(orig_embs)

        if self.active_phase in ('3', '3j', 'sft'):
            x = orig_embs + primer_out * 0.1
        else:
            x = orig_embs   # Phases 1 & 2: primer effectively zero (zero-init)

        # ConceptPerceptron scratchpad (global context anchor)
        cp_scratchpad = self.cp(x)

        # ── First half of backbone (layers 0 … mid-1) ────────────────────────
        for i, layer in enumerate(self.layers[:self._mid]):
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            # ConceptPerceptron injection every 6 layers
            if (i + 1) % 6 == 0:
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True).clone()
                x = x + (self.cp_gate * torch.tanh(global_ctx))

        # ── MID-BACKBONE ROUTING (Entropy Glass Ceiling Fix) ─────────────────
        # Routing computed from layer-40 hidden state, NOT raw embeddings.
        # This gives the router real semantic signal instead of always-zero entropy.
        mid_hidden = x  # (B, L, D) — rich semantic representation

        if self.active_phase in ('1', '2'):
            # Trainable soft routing — router learns from step 1.
            # High temperature keeps weights near-uniform early on so all arms
            # receive gradient, but the router itself is LIVE and differentiable.
            route_logits  = self.domain_router(mid_hidden)          # [B, L, 16]
            if self.training:
                noise = torch.randn_like(route_logits) * 0.05       # mild exploration noise
                route_logits = route_logits + noise
            temp          = torch.clamp(self.router_temp, min=2.0, max=10.0)  # high temp = near-uniform
            route_weights = F.softmax(route_logits / temp, dim=-1)  # [B, L, 16]

            # Light load-balancing — discourages arm collapse without forcing uniformity
            if self.training:
                mu = route_weights.mean(dim=(0, 1))
                load_balance_loss = self.mimo_paths * (mu ** 2).sum() - 1.0
                domain_loss = 0.05 * load_balance_loss   # gentle — 1/3 of Phase 3 pressure
            else:
                domain_loss = torch.tensor(0.0, dtype=x.dtype, device=device)

        elif self.active_phase == '3':
            # Phase 3 is superseded by Phase 3j for all new runs.
            # Fall through to Phase 3j routing (domain-supervised).
            route_logits  = self.domain_router(mid_hidden)
            temp          = torch.clamp(self.router_temp, min=0.1, max=0.5)
            route_weights = F.softmax(route_logits / temp, dim=-1)
            domain_loss   = torch.tensor(0.0, dtype=x.dtype, device=device)

        elif self.active_phase == '3r':
            # ── Phase 3r: ROUTER TRAINING via LM loss ────────────────────────────
            # Arms are frozen specialists. Only the router and LM head train.
            # KEY: use FULL softmax during training — top-k argmax is non-differentiable,
            # meaning gradient only reaches logits for the selected arms, and the router
            # can't learn to select different arms. Full softmax = gradient to all 8 logits.
            # Top-k is inference-only (applied in the else branch below).
            route_logits = self.domain_router(mid_hidden)   # [B, L, 8]
            temp = torch.clamp(self.router_temp, min=0.3, max=1.5)

            if self.training:
                # Full soft routing — gradient flows to all 8 router logits.
                # Router learns "arm A gets lower LM loss for math inputs" because
                # grad of loss w.r.t. route_weights[math_arm] is negative (good).
                route_weights = F.softmax(route_logits / temp, dim=-1)  # [B, L, 8]
                # Load-balance: prevent router collapsing to one arm for everything
                mu = route_weights.mean(dim=(0, 1))                      # [8]
                load_balance = self.mimo_paths * (mu ** 2).sum() - 1.0  # >0 if collapsed
                domain_loss  = 0.02 * load_balance.clamp(min=0.0)
            else:
                # At inference: top-2 sparse gating (use the best arms, not all 8)
                soft_weights = F.softmax(route_logits / temp, dim=-1)
                topk_vals, topk_idx = soft_weights.topk(2, dim=-1)
                sparse = torch.zeros_like(soft_weights)
                sparse.scatter_(-1, topk_idx, topk_vals)
                route_weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-9)
                domain_loss = torch.tensor(0.0, dtype=x.dtype, device=device)

        else:  # Phase 3j and SFT
            # Hard 1-hot routing from domain labels during training.
            # Soft routing dilutes gradient across all arms — even with domain supervision
            # the router gives each arm partial gradient, slowing specialization.
            # Hard routing: each batch 100% targets its designated arm → maximum
            # domain-specific gradient → fastest possible arm divergence from clone baseline.
            route_logits = self.domain_router(mid_hidden)  # [B, L, 8]

            if self.training and domain_ids is not None:
                # Build 1-hot route_weights from domain_ids [B] → [B, L, 8]
                # domain_ids maps: 1=math, 2=logic, 3=code, 4=factual, 5=summary, 6=chat, 0/7=general
                one_hot = F.one_hot(domain_ids.clamp(0, self.mimo_paths - 1),
                                    num_classes=self.mimo_paths).float()  # [B, 8]
                # Broadcast to [B, L, 8] — every token in the batch goes to the same arm
                route_weights = one_hot.unsqueeze(1).expand(-1, x.shape[1], -1).detach()
                # Router still gets gradient via cross-entropy loss so it learns the mapping
                mean_logits = route_logits.mean(dim=1)   # [B, 8]
                domain_loss = F.cross_entropy(mean_logits, domain_ids.clamp(0, self.mimo_paths - 1))
            else:
                # ── TOP-K SPARSE GATING at inference ─────────────────────────
                TOP_K = 2
                temp = torch.clamp(self.router_temp, min=0.3, max=1.0)
                soft_weights = F.softmax(route_logits / temp, dim=-1)  # [B, L, 8]
                topk_vals, topk_idx = soft_weights.topk(TOP_K, dim=-1)
                sparse = torch.zeros_like(soft_weights)
                sparse.scatter_(-1, topk_idx, topk_vals)
                route_weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-9)
                domain_loss = torch.tensor(0.0, dtype=x.dtype, device=device)
                if domain_ids is not None:
                    mean_logits = route_logits.mean(dim=1)
                    domain_loss = F.cross_entropy(mean_logits, domain_ids.clamp(0, self.mimo_paths - 1))

        # ── LowRank Bridge → MIMO Arms (single pass) ──────────────────────────
        bridge_out = self.bridge(x)

        # Compute each arm exactly once
        raw_arm_outs  = []
        for i in range(self.mimo_paths):
            arm_out  = self.mimo_reasoning_blocks[i](bridge_out)
            raw_arm_outs.append(arm_out)

        # ── ARM DIVERSITY LOSS ─────────────────────────────────────────────────
        # Penalise arms for producing similar outputs on the same input.
        # Gradient flows through arm activations → arms are actively pushed apart.
        # FIX: removed torch.no_grad() + .detach() — previously this was a dead
        # scalar constant that never reached arm weights. Now it's live gradient.
        # Memory: sim_matrix is [8,8] — negligible. grad_checkpoint is OFF.
        diversity_loss = torch.tensor(0.0, dtype=x.dtype, device=device)
        if self.active_phase == '3j' and self.training and domain_ids is not None:
            DIVERSITY_LAMBDA = 0.15
            arm_means = torch.stack([o.mean(dim=(0, 1)) for o in raw_arm_outs])  # [8, d]
            arm_norms = F.normalize(arm_means.float(), p=2, dim=-1).to(x.dtype)
            sim_matrix = arm_norms @ arm_norms.T                                  # [8, 8]
            n = self.mimo_paths
            pair_sims = torch.stack([sim_matrix[ii, jj]
                                     for ii in range(n)
                                     for jj in range(ii + 1, n)])
            avg_sim = pair_sims.mean()  # live gradient — pushes arms apart
            diversity_loss = DIVERSITY_LAMBDA * avg_sim
            domain_loss    = domain_loss + diversity_loss


        # stacked_states: [B, L, d_model, 8]
        stacked_states = torch.stack(raw_arm_outs, dim=-1)

        # ── ARM SNAPSHOT for telemetry (pre-IPC, mean over B×L) ───────────────
        # [d_model, n_arms] — used by glass-brain collapse metric in telemetry
        with torch.no_grad():
            self._arm_snapshot = stacked_states.detach().float().mean(dim=(0, 1))  # [d_model, n_arms]

        # ipc_mixer removed — Blackboard handles inter-arm coordination below.

        # ── TEMPORAL MEMORY: Latent Scratchpad (Hippocampus) ──────────────────
        # ConceptPerceptron is injected every 6 backbone layers above — preserved.

        # ── SPATIAL MEMORY: Sparse IPC Blackboard (Corpus Callosum) ──────────
        # Phase 3: blackboard is COMPLETELY SILENT — no forward pass, no cross-talk.
        # Frozen weights alone are not enough: even a frozen blackboard broadcasts a
        # consensus signal into every arm each step, pulling them toward sameness and
        # actively fighting the divergence Phase 3 needs.
        # Arms are fully isolated here; cross-talk re-enables in Phase 3j for synthesis.
        if self.active_phase in ('1', '2', '3j', 'sft'):
            comm_mask       = (route_weights > 0.01).float().detach()   # [B, L, 16]
            speaking_weights = route_weights * comm_mask

            states_for_bus  = stacked_states.transpose(-1, -2)          # [B, L, 16, d_model]
            bb_writes       = self.bb_write(states_for_bus.to(self.bb_write.weight.dtype)).to(x.dtype)
            weighted_writes = bb_writes * speaking_weights.unsqueeze(-1)
            blackboard      = weighted_writes.sum(dim=-2)                # [B, L, bus_dim]
            shared_context  = self.bb_read(blackboard.to(self.bb_read.weight.dtype)).to(x.dtype)
            gated_broadcast = shared_context.unsqueeze(-1) * comm_mask.unsqueeze(-2)
            stacked_states  = stacked_states + gated_broadcast           # [B, L, d_model, 16]
        # Phase 3: stacked_states unchanged — each arm's output is purely its own.

        # ── FINAL COLLAPSE (strict einsum — no broadcast multiply) ────────────
        # Dormant arms receive 0.5% trickle-charge via route_weights.
        # einsum maintains memory contiguity across 16-arm batch dimension.
        collapsed_mimo = torch.einsum(
            'b l d m, b l m -> b l d',
            stacked_states,
            route_weights.to(stacked_states.dtype)
        )  # [B, L, d_model]
        x = x + collapsed_mimo

        # ── Second half of backbone (layers mid … end) ────────────────────────
        offset = self._mid
        for i, layer in enumerate(self.layers[self._mid:]):
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if ((offset + i + 1) % 6 == 0):
                global_ctx = cp_scratchpad.mean(dim=1, keepdim=True).clone()
                x = x + (self.cp_gate * torch.tanh(global_ctx))

        # ── Output ────────────────────────────────────────────────────────────
        x      = self.norm_f(x)
        logits = self.lm_head(x)

        # ── Telemetry (sampled to avoid overhead during training) ─────────────
        if not self.training or (self.training and torch.rand(1).item() < 0.20):
            with torch.no_grad():
                arm_w_mean = route_weights.mean(dim=(0, 1)).tolist()   # (16,)
                # Top-3 active arms by mean weight
                top_arms = sorted(
                    enumerate(arm_w_mean), key=lambda x: x[1], reverse=True
                )[:3]
                top_arm_labels = [
                    {"arm": idx, "label": ARM_IDENTITIES[idx], "weight": round(w, 4)}
                    for idx, w in top_arms
                ]
                # Entropy of routing distribution (0=collapsed, log(16)≈2.77=uniform)
                w_tensor = torch.tensor(arm_w_mean, dtype=torch.float32).clamp(min=1e-9)
                routing_entropy = float(-(w_tensor * w_tensor.log()).sum())

                self.last_telemetry.update({
                    'gate_score':     float(route_weights.mean()),
                    'entropy':        routing_entropy,
                    'arm_weights':    [round(w, 4) for w in arm_w_mean],
                    'arm_labels':     ARM_IDENTITIES,
                    'top_arms':       top_arm_labels,
                    'diversity_loss': float(diversity_loss.item()) if isinstance(diversity_loss, torch.Tensor) else 0.0,
                })

            # Glass Brain - Full 16x16 Pairwise Arm Divergence
            # OUTSIDE the no_grad block: inside grad-checkpoint recompute,
            # no_grad can conflict with checkpoint autograd hooks, silently
            # swallowing the computation. detach() already breaks the grad graph.
            # 1.0 = identical clones  |  0.0 = fully orthogonal
            try:
                ss = self._arm_snapshot                     # [d_model, n_arms] — pre-IPC mean
                arms_norm = F.normalize(ss.float(), dim=0)  # unit-length columns [d_model, n_arms]
                sim_matrix = arms_norm.T @ arms_norm        # [n_arms, n_arms] cosine similarities
                n = self.mimo_paths
                off_diag   = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
                # collapse score: 1=clone, -1=orthogonal (raw cosine, not shifted)
                per_arm    = (sim_matrix * off_diag.float()).sum(dim=1) / max(1.0, n - 1.0)
                self.last_telemetry.update({
                    'arm_collapse_metric': per_arm.mean().item(),
                    'arm_collapse_mean':   round(per_arm.mean().item(), 4),
                    'arm_collapse_max':    round(per_arm.max().item(),  4),
                    'arm_sims':            [round(v, 4) for v in per_arm.tolist()],
                    'latent_energy':       round(ss.norm(dim=0).mean().item(), 4),
                })
            except Exception:
                pass  # never crash training over telemetry



        return logits, domain_loss
