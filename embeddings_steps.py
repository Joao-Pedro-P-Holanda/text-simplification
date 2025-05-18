from gloe import partial_transformer
from sentence_transformers import SentenceTransformer

from schema import Document, EmbeddingCosineSimilarity, EmbeddingModelOptions


@partial_transformer
def compare_embedded_sentences_similarity(
    sentences: tuple[Document, Document], model_name: EmbeddingModelOptions
) -> EmbeddingCosineSimilarity:
    model = SentenceTransformer(model_name, trust_remote_code=True)

    embeddings = model.encode(
        [sentences[0].text, sentences[1].text], prompt_name="passage"
    )

    return EmbeddingCosineSimilarity(
        id=sentences[0].id,
        name=sentences[0].name,
        model=model_name,
        original_simplified_similarity=model.similarity(
            embeddings[0], embeddings[1]
        ).item(),
    )
