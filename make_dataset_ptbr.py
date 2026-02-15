import random

random.seed(42)

N = 85000

assuntos = [
    "IA na educação", "ensino médio", "matemática", "física", "química",
    "história", "geografia", "português", "redação", "programação",
    "ética", "cidadania", "estatística", "aprendizagem", "avaliação"
]

verbos = [
    "melhora", "piora", "acelera", "dificulta", "facilita", "transforma",
    "explica", "resume", "organiza", "automatiza", "aproxima", "amplia"
]

objetos = [
    "o planejamento de aula", "o estudo diário", "a revisão", "a compreensão",
    "a participação da turma", "a leitura", "a escrita", "a prática",
    "o raciocínio lógico", "a autonomia", "o foco", "a motivação"
]

perguntas = [
    "Como você explicaria isso em 3 passos?",
    "Qual é a ideia principal?",
    "Dê um exemplo simples.",
    "Por que isso acontece?",
    "Qual é a diferença entre causa e consequência?",
    "Resuma em uma frase.",
    "Explique como se fosse para um aluno do 1º ano.",
    "Quais são os erros comuns?",
    "Qual é a hipótese mais provável?",
    "O que muda se eu trocar a variável?"
]

frases_curta = [
    "Olá, tudo bem?",
    "Hoje eu estudei um pouco.",
    "Amanhã eu reviso de novo.",
    "Eu não entendi essa parte.",
    "Faz sentido agora.",
    "Isso é importante.",
    "Vamos por partes.",
    "Sem pressa.",
    "Tenta de novo.",
    "Boa pergunta."
]

acentos = [
    "ação", "coração", "educação", "atenção", "informação",
    "ciência", "técnica", "pública", "país", "família",
    "você", "não", "já", "é", "às vezes", "difícil", "fácil"
]

def linha():
    r = random.random()
    if r < 0.30:
        return random.choice(frases_curta)
    if r < 0.55:
        a = random.choice(assuntos)
        v = random.choice(verbos)
        o = random.choice(objetos)
        extra = random.choice(acentos)
        return f"{a} {v} {o}, mas {extra} depende do contexto."
    if r < 0.80:
        a = random.choice(assuntos)
        q = random.choice(perguntas)
        extra = random.choice(acentos)
        return f"{a}: {q} ({extra})"
    else:
        # mini-diálogo
        extra1 = random.choice(acentos)
        extra2 = random.choice(acentos)
        return f"Aluno: eu {extra1} entendi. Professor: ótimo, agora {extra2} explique com suas palavras."

with open("input.txt", "w", encoding="utf-8", newline="\n") as f:
    for _ in range(N):
        f.write(linha() + "\n")

print("Gerado input.txt com", N, "linhas (UTF-8).")
