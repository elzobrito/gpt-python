# gpt_ptbr.py
# Toy-GPT (char-level pt-BR) em PyTorch, próximo ao seu C (RMSNorm + Attention + squared ReLU MLP)
# Requer: pip install torch

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 16
    mlp_mult: int = 4
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    steps: int = 2000
    temperature: float = 0.6
    seed: int = 42
    device: str = "cpu"


# ----------------------------
# Dataset + Tokenizer (pt-BR)
# ----------------------------
def load_lines(path: str, max_docs: int = 85000) -> List[str]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip("\r\n")
            if s:
                docs.append(s)
                if len(docs) >= max_docs:
                    break
    return docs


def build_char_vocab(docs: List[str]) -> Tuple[Dict[str, int], List[str], int]:
    # Codepoint-level: itera por caracteres unicode corretamente em Python
    charset = set()
    for d in docs:
        for ch in d:
            charset.add(ch)
    itos = sorted(list(charset))
    bos_id = len(itos)  # BOS/EOS token
    itos.append("<BOS>")
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos, bos_id


def encode(doc: str, stoi: Dict[str, int], bos_id: int) -> List[int]:
    # [BOS] + chars + [BOS]
    ids = [bos_id]
    for ch in doc:
        ids.append(stoi[ch])
    ids.append(bos_id)
    return ids


# ----------------------------
# Model pieces
# ----------------------------
class RMSNorm(nn.Module):
    # Igual ao seu C: só normaliza (sem gamma/beta)
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C) ou (B,C)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        # máscara causal fixa (T x T)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        B, T, C = x.shape

        q = self.wq(x)  # (B,T,C)
        k = self.wk(x)
        v = self.wv(x)

        # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)

        # aplica máscara causal (impede olhar o futuro)
        m = self.mask[:T, :T]
        att = att.masked_fill(~m, float("-inf"))

        w = F.softmax(att, dim=-1)
        y = w @ v  # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,C)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.n_embd
        self.fc1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        # squared ReLU: x^2 se x>0 senão 0
        h = F.relu(h).pow(2)
        return self.fc2(h)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.norm_in = RMSNorm(cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        # aproxima seu std=0.02, e zero-init em algumas projeções não é essencial aqui
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B,T)
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1,T)

        x = self.wte(idx) + self.wpe(pos)
        x = self.norm_in(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(x)  # (B,T,V)
        return logits


# ----------------------------
# Train + Sample
# ----------------------------
def cosine_lr(base_lr: float, step: int, total: int) -> float:
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total))


@torch.no_grad()
def sample(model: TinyGPT, bos_id: int, itos: List[str], cfg: Config, n_samples: int = 10) -> None:
    model.eval()
    V = model.lm_head.out_features

    for si in range(n_samples):
        idx = torch.full((1, 1), bos_id, dtype=torch.long, device=cfg.device)
        out_chars = []

        for t in range(cfg.block_size):
            # mantém só o último bloco
            idx_cond = idx[:, -cfg.block_size :]
            logits = model(idx_cond)  # (1,T,V)
            logits = logits[:, -1, :]  # (1,V)

            logits = logits / max(cfg.temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)  # (1,V)
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)

            nid = int(next_id.item())
            if nid == bos_id:
                break

            # converte id -> char
            ch = itos[nid]
            out_chars.append(ch)
            idx = torch.cat([idx, next_id], dim=1)

        print(f"sample {si+1:02d}: {''.join(out_chars)}")


def main():
    cfg = Config()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    docs = load_lines("input_ptbr.txt")
    random.shuffle(docs)

    stoi, itos, bos_id = build_char_vocab(docs)
    vocab_size = len(itos)

    sequences = [encode(d, stoi, bos_id) for d in docs]

    device = torch.device(cfg.device)
    model = TinyGPT(cfg, vocab_size).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

    print(f"docs: {len(docs)} | vocab: {vocab_size} | BOS id: {bos_id}")
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    model.train()
    for step in range(cfg.steps):
        # pega uma sequência (cíclico) e faz um recorte
        seq = sequences[step % len(sequences)]
        # garante pelo menos 2 tokens (input/target)
        if len(seq) < 2:
            continue

        # recorte aleatório melhora treino
        max_start = max(0, len(seq) - (cfg.block_size + 1))
        start = random.randint(0, max_start) if max_start > 0 else 0
        chunk = seq[start : start + cfg.block_size + 1]  # tamanho <= block_size+1

        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)   # (1,T)

        logits = model(x)  # (1,T,V)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()

        lr_t = cosine_lr(cfg.lr, step, cfg.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr_t
        opt.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1:4d}/{cfg.steps} | loss {loss.item():.4f} | lr {lr_t:.6f}")

    print("\n--- inference ---")
    sample(model, bos_id, itos, cfg, n_samples=20)


if __name__ == "__main__":
    main()
