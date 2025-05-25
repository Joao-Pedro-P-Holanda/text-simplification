from gloe import partial_transformer
from sentence_transformers import SentenceTransformer

from schema import Document, EmbeddingCosineSimilarity, EmbeddingModelOptions
import logging

logger = logging.getLogger(__name__)


@partial_transformer
def compare_embedded_sentences_similarity(
    sentences: tuple[Document, Document], model_name: EmbeddingModelOptions
) -> EmbeddingCosineSimilarity:
    model = SentenceTransformer(model_name, trust_remote_code=True)

    logger.info(
        f"Comparing embedding of files {sentences[0].name} and {sentences[1].name}"
    )
    embeddings = model.encode(
        [sentences[0].text, sentences[1].text], prompt_name="passage"
    )

    return EmbeddingCosineSimilarity(
        id=sentences[0].id,
        name=sentences[0].name,
        year=sentences[0].year,
        model=model_name,
        generated_with=sentences[0].generated_with,
        original_simplified_similarity=model.similarity(
            embeddings[0], embeddings[1]
        ).item(),
    )
