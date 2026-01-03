import os
import sys
from glob import glob
from itertools import groupby

from langchain_ollama import ChatOllama
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, HallucinationMetric
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from file_processing_steps import read_markdown_file
from llama_model import OllamaDeepeval
from settings import config
from test.conftest import base_original_path, simplified_path


dir = os.path.dirname(__file__)
parent = os.path.dirname(dir)


sys.path.append(parent)


original_files = glob(f"{base_original_path()}/complete/2025*_stripped.md") + glob(
    f"{base_original_path()}/simplified/2025*_stripped.md"
)
simplified_files = glob(
    f"{simplified_path('gemini-2.5-flash-preview-04-17')}/2025*_stripped.md"
)

documents = [read_markdown_file(file) for file in original_files + simplified_files]

paired_documents = []
for k, g in groupby(
    sorted(documents, key=lambda x: (x.name, x.path)), key=lambda x: x.name
):
    paired_documents.append(tuple(g))

for pair in paired_documents:
    print([d.path for d in pair])


def test_simplified_document_dont_contain_hallucinations():
    original_document = read_markdown_file("")
    simplified_document = read_markdown_file("")

    test_case = LLMTestCase(
        input=original_document.text, actual_output=simplified_document.text
    )
    headers = {
        "Authorization": f"Bearer {config['llm_api_key'].get_secret_value()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    metric = HallucinationMetric(
        threshold=0.5,
        model=OllamaDeepeval(
            ChatOllama(
                model="gpt-oss:20b",
                temperature=0.8,
                base_url=config["llm_url"],
                client_kwargs={"headers": headers, "timeout": 360},
            )
        ),
        strict_mode=True,
    )

    assert_test(test_case=test_case, metrics=[metric])


def get_criteria_plain_language():
    return [
        GEval(
            name="Clareza e Simplicidade",
            criteria=(
                """
                Verifique se o texto simplificado é claro e direto.
                Considere essas características do texto:
                - Frases tem um tamanho adequado, com no máximo 20 palavras
                - Frases são escritas na ordem sujeito-verbo-objeto
                - Uso da voz ativa em vez da passiva
                - Não utiliza termos estrangeiros que não são comuns em Português
                - O vocabulário é acessível, com palavras de uso geral e evitando jargões
                - não utilizar substantivos vindos de verbo
                - Siglas são escritas por extenso na primeira ocorrência 
                """
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,  # Edital original
                LLMTestCaseParams.ACTUAL_OUTPUT,  # Texto em Linguagem Simples
                LLMTestCaseParams.CONTEXT,
            ],
            verbose_mode=True,
            rubric=[
                Rubric(
                    score_range=(0, 2),
                    expected_outcome="O texto mantém termos técnicos ou complexos, frases longas e estrutura difícil de entender.",
                ),
                Rubric(
                    score_range=(3, 5),
                    expected_outcome="Há alguma tentativa de simplificação, mas ainda contém vocabulário ou construções que dificultam a compreensão.",
                ),
                Rubric(
                    score_range=(6, 8),
                    expected_outcome="O texto é claro e simples na maior parte, com apenas poucos trechos que poderiam ser mais acessíveis.",
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome="O texto é totalmente claro e de fácil leitura, sem jargões ou estruturas complexas.",
                ),
            ],
        ),
        GEval(
            name="Fidelidade ao Conteúdo Original",
            criteria=(
                "Avalie se todas as informações essenciais do edital original foram preservadas "
                "e transmitidas corretamente no texto simplificado, sem distorções ou omissões críticas."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            verbose_mode=True,
            rubric=[
                Rubric(
                    score_range=(0, 2),
                    expected_outcome="Informações importantes estão ausentes, incorretas ou distorcidas.",
                ),
                Rubric(
                    score_range=(3, 5),
                    expected_outcome="Alguns detalhes importantes foram omitidos ou mal representados, comprometendo a fidelidade.",
                ),
                Rubric(
                    score_range=(6, 8),
                    expected_outcome="A maioria das informações essenciais está correta, com pequenas omissões ou simplificações aceitáveis.",
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome="Todas as informações essenciais estão corretas e completas, mesmo em linguagem simplificada.",
                ),
            ],
        ),
        GEval(
            name="Organização e Estrutura",
            criteria=(
                "Avalie se o texto simplificado apresenta uma estrutura lógica e bem organizada, "
                "com parágrafos ou seções que facilitam a leitura e o entendimento, "
                "seguindo uma ordem coerente em relação ao edital original."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            verbose_mode=True,
            rubric=[
                Rubric(
                    score_range=(0, 2),
                    expected_outcome="O texto é confuso, com ideias desordenadas e sem sequência lógica.",
                ),
                Rubric(
                    score_range=(3, 5),
                    expected_outcome="Há alguma tentativa de organização, mas a ordem das informações ou a separação em parágrafos é inadequada.",
                ),
                Rubric(
                    score_range=(6, 8),
                    expected_outcome="O texto é majoritariamente bem estruturado, com pequenas oportunidades de melhoria na fluidez.",
                ),
                Rubric(
                    score_range=(9, 10),
                    expected_outcome="A organização é excelente, facilitando a leitura e mantendo coerência com o edital original.",
                ),
            ],
        ),
    ]
