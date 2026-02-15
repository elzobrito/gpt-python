# GPT-Python (pt-BR)

Este projeto é uma implementação de um modelo GPT em PyTorch. Ele opera a nível de caractere e foi projetado para fins educacionais, demonstrando os componentes chave de um transformador de uma forma compacta e acessível.

O modelo é treinado em um conjunto de dados em português do Brasil (`input_ptbr.txt`).

## Estrutura do Projeto

-   `gpt_ptbr.py`: O script principal que contém a definição do modelo, o loop de treinamento e a lógica de geração de texto.
-   `make_dataset_ptbr.py`: Um script para gerar um conjunto de dados de exemplo (`input.txt`) com frases em português.
-   `input_ptbr.txt`: O arquivo de texto usado para treinar o modelo.

## Requisitos

-   Python 3.x
-   PyTorch

## Instalação

Para instalar a única dependência, o PyTorch, execute o seguinte comando:

```bash
pip install torch
```

## Como Usar

### 1. Preparar o Conjunto de Dados

O repositório já inclui um arquivo `input_ptbr.txt` com 85.000 linhas, então você pode pular esta etapa se desejar.

Para gerar seu próprio conjunto de dados, execute o script `make_dataset_ptbr.py`:

```bash
python make_dataset_ptbr.py
```

Isso criará um arquivo chamado `input.txt`. Para usá-lo no script principal, você deve renomeá-lo para `input_ptbr.txt` ou alterar o nome do arquivo diretamente no `gpt_ptbr.py`.

### 2. Treinar o Modelo e Gerar Amostras

Para treinar o modelo e ver algumas amostras de texto geradas por ele, execute o script `gpt_ptbr.py`:

```bash
python gpt_ptbr.py
```

O script irá treinar o modelo por um número definido de passos (padrão: 2000) e, ao final, imprimirá 20 amostras de texto gerado.

## Configuração

Você pode ajustar os hiperparâmetros do modelo, como o tamanho do embedding, número de camadas e taxa de aprendizado, modificando os valores na classe `Config` dentro do arquivo `gpt_ptbr.py`.
