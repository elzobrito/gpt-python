# gpt_ptbr_verbose.py
# GPT char-level mínimo (pt-BR), CPU, sem NumPy/Torch.
# Foco: clareza e estrutura (não performance).
#
# Diferenças intencionais em relação ao C:
# - Separação BOS/EOS (dois tokens especiais)
# - Organização por classes (Param, Config) e funções por responsabilidade
# - Warmup + cosine decay no LR (mais estável em toy-models)
# - ReLU (mais padrão) no MLP (em vez de squared ReLU)
# - Amostragem com top-k opcional (melhora legibilidade das samples)

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import List, Tuple, Dict


# ----------------------------
# Configuração do modelo
# ----------------------------
@dataclass
class Config:
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 1          # Mantemos 1 para manter o backward mais simples e didático
    block_size: int = 8
    mlp_mult: int = 4

    steps: int = 2000
    lr: float = 1e-3
    lr_min: float = 1e-5
    warmup_steps: int = 100

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    grad_clip: float = 1.0    # clipping por norma global
    print_every: int = 100

    temperature: float = 0.5
    top_k: int | None = 20    # None desliga


# ----------------------------
# Utilidades de dataset/tokenizer
# ----------------------------
def load_dataset(path: str, max_docs: int = 85000) -> List[str]:
    docs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip("\r\n")
            if s:
                docs.append(s)
                if len(docs) >= max_docs:
                    break
    return docs


def build_char_vocab(docs: List[str]) -> Tuple[Dict[str, int], List[str], int, int]:
    """
    Vocab char-level Unicode:
    - Coleta todos os caracteres presentes (inclui acentos)
    - Ordena para ficar determinístico
    - Adiciona tokens especiais BOS/EOS no final
    """
    charset = set()
    for d in docs:
        for ch in d:
            charset.add(ch)

    itos = sorted(list(charset))
    bos_id = len(itos)
    eos_id = len(itos) + 1
    itos.append("<BOS>")
    itos.append("<EOS>")
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos, bos_id, eos_id


def encode(doc: str, stoi: Dict[str, int], bos_id: int, eos_id: int) -> List[int]:
    ids = [bos_id]
    for ch in doc:
        ids.append(stoi[ch])
    ids.append(eos_id)
    return ids


# ----------------------------
# PRNG e amostragem
# ----------------------------
def softmax(logits: List[float]) -> List[float]:
    mx = max(logits)
    exps = [math.exp(z - mx) for z in logits]
    s = sum(exps)
    inv = 1.0 / s
    return [e * inv for e in exps]


def sample_from_logits(
    logits: List[float],
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None
) -> int:
    """
    - temperature < 1 deixa mais “conservador”
    - top_k limita para os k logits mais altos (melhora legibilidade em char-level)
    """
    if temperature <= 0:
        raise ValueError("temperature deve ser > 0")

    scaled = [z / temperature for z in logits]

    if top_k is not None and top_k > 0 and top_k < len(scaled):
        # pega os top-k índices por logit
        idx = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
        sub_logits = [scaled[i] for i in idx]
        sub_probs = softmax(sub_logits)

        r = rng.random()
        cum = 0.0
        for j, p in enumerate(sub_probs):
            cum += p
            if r < cum:
                return idx[j]
        return idx[-1]

    probs = softmax(scaled)
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1


# ----------------------------
# Núcleo matemático (vetores em listas Python)
# ----------------------------
def zeros(n: int) -> List[float]:
    return [0.0] * n


def linear_fwd(x: List[float], w: List[float], nout: int, nin: int) -> List[float]:
    """
    w está em row-major: w[r*nin + c]
    y = W @ x
    """
    out = [0.0] * nout
    for r in range(nout):
        base = r * nin
        s = 0.0
        for c in range(nin):
            s += w[base + c] * x[c]
        out[r] = s
    return out


def linear_bwd_x(w: List[float], dout: List[float], nout: int, nin: int, dx: List[float]) -> None:
    """
    dx += W^T @ dout
    """
    for c in range(nin):
        s = 0.0
        for r in range(nout):
            s += dout[r] * w[r * nin + c]
        dx[c] += s


def linear_bwd_w(x: List[float], dout: List[float], nout: int, nin: int, dw: List[float]) -> None:
    """
    dW += dout @ x^T
    """
    for r in range(nout):
        base = r * nin
        dr = dout[r]
        for c in range(nin):
            dw[base + c] += dr * x[c]


def rmsnorm_fwd(x: List[float], eps: float = 1e-5) -> Tuple[List[float], float]:
    ms = 0.0
    for v in x:
        ms += v * v
    ms /= len(x)
    scale = 1.0 / math.sqrt(ms + eps)
    out = [v * scale for v in x]
    return out, scale


def rmsnorm_bwd(x: List[float], scale: float, dout: List[float], dx: List[float]) -> None:
    """
    RMSNorm sem gamma/beta (igual toy-model).
    """
    dot = 0.0
    n = len(x)
    for i in range(n):
        dot += dout[i] * x[i]
    coeff = (scale ** 3) / n
    for i in range(n):
        dx[i] += scale * dout[i] - coeff * x[i] * dot


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


# ----------------------------
# Parâmetro treinável com Adam embutido
# ----------------------------
class Param:
    def __init__(self, size: int, init_std: float, rng: random.Random):
        self.w = [rng.gauss(0.0, init_std) for _ in range(size)]
        self.g = [0.0] * size
        self.m = [0.0] * size
        self.v = [0.0] * size

    def zero_grad(self) -> None:
        for i in range(len(self.g)):
            self.g[i] = 0.0

    def adam_step(self, lr: float, b1: float, b2: float, eps: float, step: int) -> None:
        b1c = 1.0 - (b1 ** (step + 1))
        b2c = 1.0 - (b2 ** (step + 1))
        for i in range(len(self.w)):
            gi = self.g[i]
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (gi * gi)

            mhat = self.m[i] / b1c
            vhat = self.v[i] / b2c
            self.w[i] -= lr * mhat / (math.sqrt(vhat) + eps)

            self.g[i] = 0.0


# ----------------------------
# Ativações salvas (para backward)
# ----------------------------
@dataclass
class Acts:
    # embedding + rmsnorm inicial
    x_embed: List[float]
    rms0: float

    # atenção
    x_in: List[float]
    xn_attn: List[float]
    rms_attn: float
    q: List[float]
    aw: List[List[float]]      # [head][t]
    attn_out: List[float]

    # mlp
    x_mid: List[float]
    xn_mlp: List[float]
    rms_mlp: float
    mlp_pre: List[float]
    mlp_post: List[float]

    # saída
    x_out: List[float]


# ----------------------------
# GPT mínimo (1 layer, pre-norm)
# ----------------------------
class TinyGPT:
    def __init__(self, cfg: Config, vocab_size: int, rng: random.Random):
        self.cfg = cfg
        self.vocab_size = vocab_size

        n_embd = cfg.n_embd
        mlp_dim = cfg.mlp_mult * n_embd
        bs = cfg.block_size

        # Embeddings
        self.wte = Param(vocab_size * n_embd, 0.02, rng)
        self.wpe = Param(bs * n_embd, 0.02, rng)

        # Atenção (matrizes NxN)
        self.wq = Param(n_embd * n_embd, 0.02, rng)
        self.wk = Param(n_embd * n_embd, 0.02, rng)
        self.wv = Param(n_embd * n_embd, 0.02, rng)
        self.wo = Param(n_embd * n_embd, 0.0, rng)  # zero init

        # MLP
        self.fc1 = Param(mlp_dim * n_embd, 0.02, rng)   # (mlp_dim x n_embd)
        self.fc2 = Param(n_embd * mlp_dim, 0.0, rng)    # (n_embd x mlp_dim)

        # Head final (vocab x n_embd)
        self.lm = Param(vocab_size * n_embd, 0.02, rng)

        # KV cache (1 layer)
        self.kv_k = [[0.0] * n_embd for _ in range(bs)]
        self.kv_v = [[0.0] * n_embd for _ in range(bs)]

    def clear_kv(self) -> None:
        n_embd = self.cfg.n_embd
        for t in range(self.cfg.block_size):
            rowk = self.kv_k[t]
            rowv = self.kv_v[t]
            for i in range(n_embd):
                rowk[i] = 0.0
                rowv[i] = 0.0

    def num_params(self) -> int:
        return sum(len(p.w) for p in [self.wte, self.wpe, self.wq, self.wk, self.wv, self.wo, self.fc1, self.fc2, self.lm])

    def forward_one(self, token_id: int, pos: int) -> Tuple[List[float], Acts]:
        cfg = self.cfg
        n_embd = cfg.n_embd
        mlp_dim = cfg.mlp_mult * n_embd
        head_dim = n_embd // cfg.n_head

        # 1) x = token_emb + pos_emb
        x = [0.0] * n_embd
        for i in range(n_embd):
            x[i] = self.wte.w[token_id * n_embd + i] + self.wpe.w[pos * n_embd + i]
        x_embed = x[:]

        # 2) RMSNorm inicial
        x, rms0 = rmsnorm_fwd(x)

        # ---- Atenção (pre-norm) ----
        x_in = x[:]
        xn_attn, rms_attn = rmsnorm_fwd(x)

        q = linear_fwd(xn_attn, self.wq.w, n_embd, n_embd)
        k = linear_fwd(xn_attn, self.wk.w, n_embd, n_embd)
        v = linear_fwd(xn_attn, self.wv.w, n_embd, n_embd)

        self.kv_k[pos] = k[:]
        self.kv_v[pos] = v[:]

        seq_len = pos + 1
        scale = 1.0 / math.sqrt(head_dim)

        aw = [[0.0] * cfg.block_size for _ in range(cfg.n_head)]
        attn_out = [0.0] * n_embd

        for h in range(cfg.n_head):
            hs = h * head_dim
            logits = [0.0] * seq_len

            for t in range(seq_len):
                dot = 0.0
                kt = self.kv_k[t]
                for j in range(head_dim):
                    dot += q[hs + j] * kt[hs + j]
                logits[t] = dot * scale

            probs = softmax(logits)
            for t in range(seq_len):
                aw[h][t] = probs[t]

            for j in range(head_dim):
                s = 0.0
                for t in range(seq_len):
                    s += probs[t] * self.kv_v[t][hs + j]
                attn_out[hs + j] = s

        # proj + residual
        tmp = linear_fwd(attn_out, self.wo.w, n_embd, n_embd)
        x = [tmp[i] + x_in[i] for i in range(n_embd)]
        x_mid = x[:]

        # ---- MLP (pre-norm) ----
        xn_mlp, rms_mlp = rmsnorm_fwd(x)
        h1 = linear_fwd(xn_mlp, self.fc1.w, mlp_dim, n_embd)
        h2 = [relu(z) for z in h1]
        tmp2 = linear_fwd(h2, self.fc2.w, n_embd, mlp_dim)

        x = [tmp2[i] + x_mid[i] for i in range(n_embd)]
        x_out = x[:]

        # head final
        logits = linear_fwd(x, self.lm.w, self.vocab_size, n_embd)

        acts = Acts(
            x_embed=x_embed, rms0=rms0,
            x_in=x_in, xn_attn=xn_attn, rms_attn=rms_attn, q=q, aw=aw, attn_out=attn_out,
            x_mid=x_mid, xn_mlp=xn_mlp, rms_mlp=rms_mlp, mlp_pre=h1, mlp_post=h2,
            x_out=x_out
        )
        return logits, acts

    def backward_sequence(
        self,
        tokens: List[int],
        targets: List[int],
        probs_saved: List[List[float]],
        acts_saved: List[Acts],
        n: int
    ) -> None:
        """
        Backprop manual na sequência inteira (n posições).
        Usa acumuladores dk/dv para lidar com o acoplamento temporal da atenção.
        """
        cfg = self.cfg
        n_embd = cfg.n_embd
        mlp_dim = cfg.mlp_mult * n_embd
        head_dim = n_embd // cfg.n_head

        dk_acc = [[0.0] * n_embd for _ in range(cfg.block_size)]
        dv_acc = [[0.0] * n_embd for _ in range(cfg.block_size)]
        inv_n = 1.0 / n

        for pos in range(n - 1, -1, -1):
            act = acts_saved[pos]
            seq_len = pos + 1

            # dL/dlogits = (p - one_hot) / n
            dl = [0.0] * self.vocab_size
            for i in range(self.vocab_size):
                dl[i] = (probs_saved[pos][i] - (1.0 if i == targets[pos] else 0.0)) * inv_n

            # backprop no head final: logits = lm @ x_out
            dx = [0.0] * n_embd
            linear_bwd_x(self.lm.w, dl, self.vocab_size, n_embd, dx)
            linear_bwd_w(act.x_out, dl, self.vocab_size, n_embd, self.lm.g)

            # ---- MLP backward ----
            d_h2 = [0.0] * mlp_dim
            linear_bwd_x(self.fc2.w, dx, n_embd, mlp_dim, d_h2)
            linear_bwd_w(act.mlp_post, dx, n_embd, mlp_dim, self.fc2.g)

            # ReLU backward
            d_h1 = [0.0] * mlp_dim
            for i in range(mlp_dim):
                d_h1[i] = d_h2[i] if act.mlp_pre[i] > 0.0 else 0.0

            d_xn_mlp = [0.0] * n_embd
            linear_bwd_x(self.fc1.w, d_h1, mlp_dim, n_embd, d_xn_mlp)
            linear_bwd_w(act.xn_mlp, d_h1, mlp_dim, n_embd, self.fc1.g)

            d_x_mid = [0.0] * n_embd
            rmsnorm_bwd(act.x_mid, act.rms_mlp, d_xn_mlp, d_x_mid)
            for i in range(n_embd):
                dx[i] += d_x_mid[i]   # residual do MLP

            # ---- Attention backward ----
            d_attn_out = [0.0] * n_embd
            linear_bwd_x(self.wo.w, dx, n_embd, n_embd, d_attn_out)
            linear_bwd_w(act.attn_out, dx, n_embd, n_embd, self.wo.g)

            d_q = [0.0] * n_embd
            scale = 1.0 / math.sqrt(head_dim)

            # atenção: cada head contribui
            for h in range(cfg.n_head):
                hs = h * head_dim

                # grad wrt attention weights
                d_aw = [0.0] * seq_len
                for j in range(head_dim):
                    for t in range(seq_len):
                        d_aw[t] += d_attn_out[hs + j] * self.kv_v[t][hs + j]
                        dv_acc[t][hs + j] += act.aw[h][t] * d_attn_out[hs + j]

                # softmax backward: d_logit = p * (d_p - sum(d_p*p))
                dot = 0.0
                for t in range(seq_len):
                    dot += d_aw[t] * act.aw[h][t]

                d_al = [0.0] * seq_len
                for t in range(seq_len):
                    d_al[t] = act.aw[h][t] * (d_aw[t] - dot)

                # logits[t] = (q · k_t) * scale
                for t in range(seq_len):
                    kt = self.kv_k[t]
                    for j in range(head_dim):
                        d_q[hs + j] += d_al[t] * kt[hs + j] * scale
                        dk_acc[t][hs + j] += d_al[t] * act.q[hs + j] * scale

            # volta pelos projetores Q/K/V
            d_xn = [0.0] * n_embd

            linear_bwd_x(self.wq.w, d_q, n_embd, n_embd, d_xn)
            linear_bwd_w(act.xn_attn, d_q, n_embd, n_embd, self.wq.g)

            linear_bwd_x(self.wk.w, dk_acc[pos], n_embd, n_embd, d_xn)
            linear_bwd_w(act.xn_attn, dk_acc[pos], n_embd, n_embd, self.wk.g)

            linear_bwd_x(self.wv.w, dv_acc[pos], n_embd, n_embd, d_xn)
            linear_bwd_w(act.xn_attn, dv_acc[pos], n_embd, n_embd, self.wv.g)

            # RMSNorm (atenção) + residual
            d_x_in = [0.0] * n_embd
            rmsnorm_bwd(act.x_in, act.rms_attn, d_xn, d_x_in)
            for i in range(n_embd):
                dx[i] += d_x_in[i]

            # RMSNorm inicial -> embeddings
            d_embed = [0.0] * n_embd
            rmsnorm_bwd(act.x_embed, act.rms0, dx, d_embed)

            tok = tokens[pos]
            for i in range(n_embd):
                self.wte.g[tok * n_embd + i] += d_embed[i]
                self.wpe.g[pos * n_embd + i] += d_embed[i]

    def parameters(self) -> List[Param]:
        return [self.wte, self.wpe, self.wq, self.wk, self.wv, self.wo, self.fc1, self.fc2, self.lm]


# ----------------------------
# Treinador: LR schedule + grad clip + steps
# ----------------------------
def lr_schedule(cfg: Config, step: int) -> float:
    """
    Warmup linear + cosine decay até lr_min.
    Isso deixa toy-models menos instáveis no começo.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps

    # progresso depois do warmup
    t = (step - cfg.warmup_steps) / max(1, (cfg.steps - cfg.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine


def clip_grad_global(params: List[Param], max_norm: float) -> None:
    """
    Clipping por norma global (L2).
    """
    if max_norm <= 0:
        return
    ss = 0.0
    for p in params:
        for g in p.g:
            ss += g * g
    norm = math.sqrt(ss)
    if norm <= max_norm or norm == 0.0:
        return
    scale = max_norm / norm
    for p in params:
        for i in range(len(p.g)):
            p.g[i] *= scale


def main():
    cfg = Config()
    rng = random.Random(42)

    docs = load_dataset("input.txt")
    rng.shuffle(docs)

    stoi, itos, BOS, EOS = build_char_vocab(docs)
    vocab_size = len(itos)

    model = TinyGPT(cfg, vocab_size, rng)

    print(f"docs: {len(docs)} | vocab: {vocab_size} | BOS: {BOS} | EOS: {EOS}")
    print(f"params: {model.num_params()}")

    probs_saved = [[0.0] * vocab_size for _ in range(cfg.block_size)]
    acts_saved: List[Acts] = []

    for step in range(cfg.steps):
        doc = docs[step % len(docs)]
        tok_seq = encode(doc, stoi, BOS, EOS)

        n = min(cfg.block_size, len(tok_seq) - 1)
        targets = [tok_seq[i + 1] for i in range(n)]

        acts_saved = []
        loss_sum = 0.0

        # forward (pos a pos)
        model.clear_kv()
        for pos in range(n):
            logits, acts = model.forward_one(tok_seq[pos], pos)
            probs = softmax(logits)
            probs_saved[pos] = probs
            acts_saved.append(acts)

            p = probs[targets[pos]]
            loss_sum += -math.log(p + 1e-30)

        loss = loss_sum / n

        # backward
        model.backward_sequence(tok_seq, targets, probs_saved, acts_saved, n)

        # grad clip
        clip_grad_global(model.parameters(), cfg.grad_clip)

        # update
        lr = lr_schedule(cfg, step)
        for p in model.parameters():
            p.adam_step(lr, cfg.beta1, cfg.beta2, cfg.eps, step)

        if step == 0 or (step + 1) % cfg.print_every == 0:
            print(f"step {step+1:4d}/{cfg.steps} | loss {loss:.4f} | lr {lr:.6f}")

    # inferência (samples)
    print("\n--- inference ---")
    for si in range(20):
        model.clear_kv()
        token = BOS
        out_chars: List[str] = []

        for pos in range(cfg.block_size):
            logits, _ = model.forward_one(token, pos)
            token = sample_from_logits(
                logits,
                rng=rng,
                temperature=cfg.temperature,
                top_k=cfg.top_k
            )
            if token == EOS:
                break
            # evita imprimir tokens especiais
            ch = itos[token]
            if ch not in ("<BOS>", "<EOS>"):
                out_chars.append(ch)

        print(f"sample {si+1:02d}: {''.join(out_chars)}")


if __name__ == "__main__":
    main()
