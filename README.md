# Scripts de simplificação textual

Esse repositório contém os scripts de simplificação textual seguindo padrões de linguagem simples com LLMs disponíveis no Ollama e os modelos Gemini 2.5 Flash e Pro.

O prompt utilizado pelo modelo foi baseado no disponível [neste link](https://github.com/bryankhelven/coherence-findings/blob/main/Prompt1_Local_Coherence_Analysis_Prompt.txt).

O pacote [gloe](https://gloe.ideos.com.br/index.html) também foi utilizado, ele serve como um utilitário para organizar funções em um DAG, facilitando manutenção e composição, os decoradores `@transformer`, `@partial_transformer` e outros não tem relação com transformers do Hugging Face.

## Configuração

- Criar um ambiente virtual `python -m venv venv` ou outro comando dependendo do gerenciador de pacotes utilizado
- Instalar as dependências com `pip install -r requirements.txt`, caso utilize o _uv_ o comando é apenas `uv sync`
- Configurar a chave de API em `LLM_API_KEY` e a url do serviço em `OLLAMA_HOST` no seu arquivo .env, a implementação suporta serviços compatíveis com interface da OpenAI (p. ex. Open WebUI, OpenRouter, etc.)

## Métricas Morfossintáticas

Para avaliar a complexidade textual com métricas morfossintáticas, os textos analisados passaram por um pré-processamento:
a sentenciação foi feita com o projeto [portSentencer](https://github.com/LuceleneL/portSentencer)
e a tokenização para o formato Conllu foi feita com o [portTokenizer](https://github.com/LuceleneL/portTokenizer) . 

O script utilizado para executar o PortParser pelo Google Colab está disponível 
[neste link](https://colab.research.google.com/drive/1HoyrhzemKY2rfKWEYISUgme_tIPlIgJ2?usp=sharing)



