"""Float32 PyTorch policy forward matching the trained JAX architecture."""

import io
import math
from dataclasses import dataclass, fields
from importlib.resources import files as resources

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    hidden_dim: int
    mha_dim: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    enc_channels: int
    smolgen_hidden_channels: int
    smolgen_hidden_dim: int
    smolgen_gen_dim: int
    policy_dim: int
    encoder_qk_norm: str

    @classmethod
    def from_dict(cls, data):
        if (
            data.get("model_family") != "shogi_transformer_v2"
            or data.get("pe_input_layout") != "compact_2629"
        ):
            raise ValueError("unsupported architecture")
        spec = cls(**{f.name: data[f.name] for f in fields(cls)})
        if any(
            type(getattr(spec, f.name)) is not int or getattr(spec, f.name) < 1
            for f in fields(cls)
            if f.name != "encoder_qk_norm"
        ):
            raise ValueError("architecture dimensions must be positive integers")
        if spec.mha_dim % spec.num_attention_heads or spec.encoder_qk_norm not in ("none", "rms"):
            raise ValueError("invalid attention configuration")
        return spec


def parameter_shapes(c):
    result = {}

    def linear(name, width, inputs, bias=True):
        result[name + ".weight"] = (width, inputs)
        if bias:
            result[name + ".bias"] = (width,)

    def norm(name, width):
        result[name + ".weight"] = result[name + ".bias"] = (width,)

    linear("rating.context", 32, 2)
    linear("rating.clock_context", 32, 4)
    linear("ip_emb_preproc", 95 * c.enc_channels, 2629)
    linear("ip_emb", c.hidden_dim, 387 + c.enc_channels)
    norm("ip_emb_ln", c.hidden_dim)
    result["ip_mult_gate"] = result["ip_add_gate"] = (95, c.hidden_dim)
    linear("ip_emb_ffn.fc1", c.intermediate_size, c.hidden_dim)
    linear("ip_emb_ffn.fc2", c.hidden_dim, c.intermediate_size)
    norm("ip_emb_ffn.norm", c.hidden_dim)
    result["smolgen_w"] = (c.smolgen_gen_dim, 95 * 95)
    for i in range(c.num_layers):
        p = f"encoder.{i}."
        linear(f"rating.encoder.{i}", 4 * c.hidden_dim, 32)
        for n in ("q", "k", "v"):
            linear(p + n, c.mha_dim, c.hidden_dim)
        linear(p + "out", c.hidden_dim, c.mha_dim)
        norm(p + "norm1", c.hidden_dim)
        norm(p + "norm2", c.hidden_dim)
        linear(p + "ffn_fc1", c.intermediate_size, c.hidden_dim)
        linear(p + "ffn_fc2", c.hidden_dim, c.intermediate_size)
        s = p + "smolgen."
        linear(s + "compress", c.smolgen_hidden_channels, c.hidden_dim, False)
        linear(s + "dense1", c.smolgen_hidden_dim, 95 * c.smolgen_hidden_channels)
        norm(s + "ln1", c.smolgen_hidden_dim)
        linear(s + "dense2", c.num_attention_heads * c.smolgen_gen_dim, c.smolgen_hidden_dim)
        norm(s + "ln2", c.num_attention_heads * c.smolgen_gen_dim)
    linear("ip_pol", c.policy_dim, c.hidden_dim)
    for n in ("q", "k"):
        linear("heads.vanilla." + n, c.policy_dim, c.policy_dim)
    result["heads.vanilla.promo_from"] = result["heads.vanilla.promo_to"] = (c.policy_dim,)
    result["heads.vanilla.promo_bias"] = ()
    return result


class PolicyNetwork:
    def __init__(self, config, weights, *, device="cpu"):
        self.config, self.device = config, torch.device(device)
        shapes = parameter_shapes(config)
        if set(weights) != set(shapes) or any(tuple(weights[k].shape) != s for k, s in shapes.items()):
            raise ValueError("policy parameter names or shapes differ from architecture")
        if any(not torch.isfinite(x).all() for x in weights.values()):
            raise ValueError("nonfinite model weights")
        self.w = {k: v.to(device=self.device, dtype=torch.float32) for k, v in weights.items()}
        blob = resources("shogi_human_policy").joinpath("geometry.npz").read_bytes()
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            self.geometry = {
                k: torch.as_tensor(np.array(v), device=self.device).long() for k, v in data.items()
            }

    @torch.inference_mode()
    def __call__(self, batch):
        c = self.config
        b = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

        def linear(x, name):
            return F.linear(x, self.w[name + ".weight"], self.w.get(name + ".bias"))

        modulation = {}

        def norm(x, name):
            z = x - x.mean(-1, keepdim=True)
            y = (
                z * torch.rsqrt((z * z).mean(-1, keepdim=True) + 1e-3) * self.w[name + ".weight"]
                + self.w[name + ".bias"]
            )
            if name in modulation:
                scale, shift = modulation[name]
                y = y * (1 + scale[:, None]) + shift[:, None]
            return y

        def mish(x):
            return x * torch.tanh(F.softplus(x))

        known = b["condition_known"].float()
        r = torch.where(known[:, :1] > 0, (b["ratings"].float() - 1500) / 1500, 0)
        t = torch.where(known[:, 1:] > 0, torch.log1p(b["clock_seconds"].float()) / math.log1p(3600), 0)
        context = F.silu(
            linear(r, "rating.context") + linear(torch.cat((t, known), -1), "rating.clock_context")
        )
        board = b["board_piece_history"].long()
        hand = b["hand_count_history"].long() - 31
        index = torch.where(board >= 17, board - 3, board - 1)
        pieces = F.one_hot(index.clamp(0, 27), 28).float() * (board != 0)[..., None]
        counts = F.one_hot(hand.clamp(0, 18), 19).float()
        size = board.shape[0]
        repeat = b["repetition_history"].float()
        current_repeat = repeat[:, :1, None]
        current = torch.cat(
            (
                torch.cat((pieces[:, :, 0], current_repeat.expand(size, 81, 1)), -1).reshape(size, -1),
                torch.cat((counts[:, :, 0], current_repeat.expand(size, 14, 1)), -1).reshape(size, -1),
            ),
            -1,
        )
        pieces = F.pad(pieces, (0, 0, 0, 0, 0, 14))
        counts = F.pad(counts, (0, 0, 0, 0, 81, 0))
        frames = torch.cat((pieces, counts, repeat[:, None, :, None].expand(size, 95, 8, 1)), -1)
        ply = b["game_ply"].float()
        limit = b["draw_limit_max_plies"].float()
        extra = torch.stack(
            ((ply / limit).clamp(0, 1), ((limit - ply + 1) / limit).clamp(0, 1), torch.ones_like(ply)), -1
        )
        raw = torch.cat((frames.reshape(size, 95, 384), extra[:, None].expand(size, 95, 3)), -1)
        pe = linear(current, "ip_emb_preproc").reshape(size, 95, c.enc_channels)
        x = norm(mish(linear(torch.cat((raw, pe), -1), "ip_emb")), "ip_emb_ln")
        x = x * self.w["ip_mult_gate"] + self.w["ip_add_gate"]
        alpha = (2 * c.num_layers) ** (-0.25)
        x = norm(
            x + alpha * linear(F.relu(linear(x, "ip_emb_ffn.fc1")) ** 2, "ip_emb_ffn.fc2"), "ip_emb_ffn.norm"
        )
        heads = c.num_attention_heads
        dim = c.mha_dim // heads
        for i in range(c.num_layers):
            p = f"encoder.{i}."
            s1, t1, s2, t2 = linear(context, f"rating.encoder.{i}").chunk(4, -1)
            modulation[p + "norm1"] = (s1, t1)
            modulation[p + "norm2"] = (s2, t2)
            q, k, v = [
                linear(x, p + n).reshape(size, 95, heads, dim).permute(0, 2, 1, 3) for n in ("q", "k", "v")
            ]
            if c.encoder_qk_norm == "rms":
                q = q * torch.rsqrt((q * q).mean(-1, keepdim=True) + 1e-6)
                k = k * torch.rsqrt((k * k).mean(-1, keepdim=True) + 1e-6)
            scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
            s = p + "smolgen."
            h = linear(x, s + "compress").reshape(size, -1)
            h = norm(F.silu(linear(h, s + "dense1")), s + "ln1")
            h = norm(F.silu(linear(h, s + "dense2")), s + "ln2")
            bias = (h.reshape(size, heads, c.smolgen_gen_dim) @ self.w["smolgen_w"]).reshape(
                size, heads, 95, 95
            )
            attention = (
                (torch.softmax(scores + bias, -1) @ v).permute(0, 2, 1, 3).reshape(size, 95, c.mha_dim)
            )
            x = norm(x + alpha * linear(attention, p + "out"), p + "norm1")
            x = norm(x + alpha * linear(F.relu(linear(x, p + "ffn_fc1")) ** 2, p + "ffn_fc2"), p + "norm2")
        pol = mish(linear(x, "ip_pol"))
        q = linear(pol[:, :88], "heads.vanilla.q")
        k = linear(pol[:, :81], "heads.vanilla.k")
        sources = self.geometry["sources"]
        candidates = (board[:, :, 0] != 0)[:, sources.clamp(0, 80)] & (sources[None] >= 0)
        candidates = torch.where(
            self.geometry["direct"][None, :, None].bool(), sources[None] >= 0, candidates
        )
        valid = candidates.any(-1)
        selected = torch.gather(
            sources[None].expand(size, -1, -1), 2, candidates.long().argmax(-1)[..., None]
        )[..., 0].clamp(min=0)
        dest = self.geometry["destinations"]
        logits = torch.gather(
            (q @ k.transpose(-1, -2)).reshape(size, -1), 1, selected * 81 + dest
        ) / math.sqrt(c.policy_dim)
        pf = (q * self.w["heads.vanilla.promo_from"]).sum(-1)
        pt = (k * self.w["heads.vanilla.promo_to"]).sum(-1)
        promo = torch.gather(pf, 1, selected) + pt[:, dest] + self.w["heads.vanilla.promo_bias"]
        return torch.where(valid, logits + promo * self.geometry["promote"], 0)
