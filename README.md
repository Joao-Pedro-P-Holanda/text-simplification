# Scripts de simplificação textual

Esse repositório contém os scripts de simplificação textual seguindo padrões de linguagem simples com LLMs disponíveis no Ollama e os modelos Gemini 2.5 Flash e Pro.

O prompt utilizado pelo modelo foi baseado no disponível [neste link](https://github.com/bryankhelven/coherence-findings/blob/main/Prompt1_Local_Coherence_Analysis_Prompt.txt).

O pacote [gloe](https://gloe.ideos.com.br/index.html) também foi utilizado, ele serve como um utilitário para organizar funções em um DAG, facilitando manutenção e composição, os decoradores `@transformer`, `@partial_transformer` e outros não tem relação com transformers do Hugging Face.

## Execução

- Criar um ambiente virtual `python -m venv venv` ou outro comando dependendo do gerenciador de pacotes utilizado
- Instalar as dependências com `pip install -r requirements.txt`, caso utilize o _uv_ o comando é apenas `uv sync`
- Adicionar a chave de API em `LLM_API_KEY` no seu arquivo .env
- Executar o script: `python main.py`


