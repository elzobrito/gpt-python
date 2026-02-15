# make_dataset_ptbr.py
from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

def normalize_ptbr(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFC", s)
    s = s.replace("—", "-").replace("“", '"').replace("”", '"').replace("…", "...")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ ]+([,.!?;:])", r"\1", s)
    s = re.sub(r"\.{4,}", "...", s)
    s = re.sub(r"([!?]){3,}", r"\1\1", s)
    return s

def build_lines(seed: int = 42, n: int = 85000) -> list[str]:
    rng = random.Random(seed)

    sujeitos = ["eu", "você", "a gente", "o modelo", "o sistema", "o dado", "a rotina", "o treino"]
    verbos = ["testei", "medi", "reduzi", "aumentei", "quebrei", "corrigi", "refiz", "simplifiquei", "debuguei"]
    objetos = ["o dataset", "o prompt", "o código", "o loss", "o tokenizer", "a atenção", "o gradiente", "o cache"]

    efeitos = ["e ficou melhor.", "e piorou.", "e estabilizou.", "e explodiu.", "e fez sentido.", "e virou ruído."]

    conectivos_meio = ["porque", "mas", "só que", "então"]
    conectivos_inicio = ["se", "quando"]
    conectivo_contraste = "mesmo assim"

    negacoes = ["não", "nunca", "quase nunca", "nem sempre"]

    def clause_a() -> str:
        return f"{rng.choice(sujeitos)} {rng.choice(verbos)} {rng.choice(objetos)}"

    def clause_b() -> str:
        return rng.choice([
            "o resultado muda.",
            "o texto melhora.",
            "o loss cai.",
            "o modelo trava.",
            "a amostra fica torta.",
            "a atenção cola no começo.",
            "a frase não fecha.",
            "o dataset fica previsível.",
            "o ruído domina o padrão.",
        ])

    def reason_clause() -> str:
        return rng.choice([
            "o dataset está desigual.",
            "tem caractere raro demais.",
            "o começo se repete muito.",
            "o contexto é curto.",
            "o treino é pequeno.",
            "a amostragem está fria.",
        ])

    def qa_pair() -> str:
        q = rng.choice([
            "Por que o modelo erra tanto?",
            "Como melhorar o dataset?",
            "O que o GPT aprende primeiro?",
            "Por que o loss oscila?",
        ])
        a = rng.choice([
            "Repetição com variação controlada.",
            "Menos ruído, mais padrão.",
            "Contexto curto exige frases curtas.",
            "Unicode limpo faz diferença.",
        ])
        return f"Pergunta: {q} Resposta: {a}"

    templates = (
        ["{A} {E}"] * 2 +
        ["{S} {NEG} é mágica: é estatística."] +
        ["{S} aprende padrão, não intenção."] +
        ["Hoje: {A}. Amanhã: {B}"] +
        ["{QA}"] +
        ["{A}, {CMEIO} {B}"] * 6 +
        ["{A} {CMEIO} {B}"] * 2 +
        ["{A} {B} {MASSIM}, {B2}"] * 4 +
        ["{A}. Mesmo assim, {B2}"] * 4 +
        ["{CINI} {A}, então {B}"] * 5
    )

    def fill(tpl: str) -> str:
        A = clause_a()
        B = clause_b()
        B2 = clause_b()

        cmeio = rng.choice(conectivos_meio)
        if cmeio == "porque":
            B = reason_clause()
        elif cmeio == "então":
            B = rng.choice(["o loss cai.", "o texto melhora.", "a frase fecha melhor.", "o padrão aparece."])

        CINI = "Se" if rng.choice(conectivos_inicio) == "se" else "Quando"

        s = tpl.format(
            S=rng.choice(sujeitos),
            NEG=rng.choice(negacoes),
            A=A,
            B=B,
            B2=B2,
            CMEIO=cmeio,
            CINI=CINI,
            MASSIM=conectivo_contraste,
            E=rng.choice(efeitos),
            QA=qa_pair(),
        )
        return normalize_ptbr(s)

    lines: list[str] = []
    seen: set[str] = set()
    while len(lines) < n:
        s = fill(rng.choice(templates))

        if not (20 <= len(s) <= 110):
            continue
        if any(ch.isdigit() for ch in s):
            continue
        if s in seen:
            continue
        seen.add(s)
        lines.append(s)

    return lines

def write_dataset(out_path: Path, lines: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for s in lines:
            f.write(s + "\n")

def main() -> None:
    seed = 42
    n = 85000

    # Salva SEMPRE na mesma pasta do script
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "input.txt"

    lines = build_lines(seed=seed, n=n)
    write_dataset(out_path, lines)

    # Confirmação com caminho absoluto
    print(f"Gerado {out_path} com {len(lines)} linhas (UTF-8).")

if __name__ == "__main__":
    main()
