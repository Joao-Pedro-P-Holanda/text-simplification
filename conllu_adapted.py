from collections import Counter
from dataclasses import dataclass
from statistics import mean
from itertools import chain
from typing import Any
from pyphen import Pyphen
import string

TokenId = int | tuple[int, int] | float


@dataclass
class Conllu:
    sentences: list["Sentence"]
    generator: str | None = None
    udpipe_model: str | None = None
    udpipe_model_licence: str | None = None

    @property
    def text(self) -> str:
        return "".join([sentence.text for sentence in self.sentences])

    @property
    def average_words_per_sentence(self) -> float:
        return mean([len(sentence.tokens) for sentence in self.sentences])

    @property
    def max_sentence_length(self) -> int:
        return max(
            [
                len(
                    list(
                        filter(
                            lambda tok: tok.form not in string.punctuation,
                            sentence.tokens,
                        )
                    )
                )
                for sentence in self.sentences
            ]
        )

    @property
    def token_count(self) -> int:
        tokens = list(
            chain.from_iterable([sentence.tokens for sentence in self.sentences])
        )
        return len(list(filter(lambda tok: tok.form not in string.punctuation, tokens)))

    @property
    def character_count(self) -> int:
        tokens = list(
            chain.from_iterable([sentence.tokens for sentence in self.sentences])
        )
        return sum([len(token.form) for token in tokens])

    @property
    def types(self) -> set[str]:
        tokens = chain.from_iterable([sentence.tokens for sentence in self.sentences])
        texts = [token.form for token in tokens if token.form not in string.punctuation]
        return set(texts)

    @property
    def syllables_count_pyphen(self) -> int:
        """
        Counts the number of syllables using pyphen dictionaries,
        based on the implementation in

        https://github.com/sloev/spacy-syllables/blob/master/spacy_syllables/__init__.py
        """

        count = 0
        dictionary = Pyphen(lang="pt_BR")
        tokens = chain.from_iterable([sentence.tokens for sentence in self.sentences])
        for token in tokens:
            trimmed_hyphen = token.form.replace("-", "")
            if trimmed_hyphen.isalpha():
                for subword in token.form.split("-"):
                    syllables = dictionary.inserted(subword.lower()).split("-")
                    count += len(syllables)

        return count

    @property
    def hapax_legomena(self) -> list[str]:
        tokens = chain.from_iterable([sentence.tokens for sentence in self.sentences])
        texts = [token.form for token in tokens]
        frequencies = Counter(texts)
        return [text for text, freq in frequencies.items() if freq == 1]

    @staticmethod
    def from_file(path: str, decontract: bool = False) -> "Conllu":
        text = ""
        with open(path) as file:
            blocks: list[list[str]] = []

            current_block: list[str] = []
            for line in file:
                if line != "\n":
                    current_block.append(line)
                else:
                    blocks.append(current_block)
                    current_block = []

            if current_block:
                blocks.append(current_block)

        if blocks:
            metadata = Conllu._resolve_metadata(blocks[0])

            return Conllu(
                sentences=Conllu._convert_blocks_to_sentences(blocks, decontract),
                **metadata,
            )

        else:
            raise ValueError("File don' match conllu format")

    @staticmethod
    def _resolve_metadata(block: list[str]) -> dict[str, Any]:
        """
        Extracts the metadata fields on the first sentence block

        This operation removes elements from the list
        """
        metadata = {}

        lines_to_remove = []

        fields = ["generator", "udpipe_model", "udpipe_model_licence"]

        for line in block:
            if any([line.startswith(f"# {field} = ") for field in fields]):
                metadata.update({line.split()[1]: line[line.index("=") + 2 :]})
                lines_to_remove.append(line)
            elif line.startswith("# newdoc") or line.startswith("# newpar"):
                lines_to_remove.append(line)
            elif line.startswith("# sent_id"):
                break
            else:
                raise ValueError(f"Couldn't identify metadata on comment line {line}")

        for metadata_line in lines_to_remove:
            block.remove(metadata_line)

        return metadata

    @staticmethod
    def _convert_blocks_to_sentences(
        blocks: list[list[str]], decontract: bool = False
    ) -> list["Sentence"]:
        sentences: list[Sentence] = []

        for block in blocks:
            sent_id = None
            tokens = []
            text = None
            for line in block:
                if line.startswith("# newpar"):
                    continue
                if line.startswith("# sent_id = "):
                    id_text = line.split()[-1]
                    if id_text.isnumeric():
                        sent_id = int(id_text)
                    else:
                        sent_id = id_text
                elif line.startswith("# text = "):
                    text = line[line.index("=") + 2 :]

                else:
                    try:
                        tokens.append(Conllu._token_for_conllu_line(line))
                    except:
                        raise

            if not decontract:
                # remove tokens that are present in a previous contracted token
                ids_to_remove = []
                for token in tokens:
                    if isinstance(token.id, tuple):
                        ids_to_remove.extend(list(range(token.id[0], token.id[1] + 1)))

                tokens = list(
                    filter(lambda token: token.id not in ids_to_remove, tokens)
                )

            if not tokens:
                continue

            if not sent_id or not text:
                raise ValueError("Invalid sentence")

            sentences.append(Sentence(id=sent_id, tokens=tokens, text=text))

        return sentences

    @staticmethod
    def _token_for_conllu_line(line: str) -> "Token":
        line = line.removesuffix("\n")
        fields = line.split("\t")

        if len(fields) != 10:
            raise ValueError("Number of fields should be exactly ten")

        field_for_index = {
            0: "id",
            1: "form",
            2: "lemma",
            3: "upos",
            4: "xpos",
            5: "feats",
            6: "head",
            7: "deprel",
            8: "deps",
            9: "misc",
        }

        def _conversion_for_field(
            field: str, text: str
        ) -> TokenId | str | dict[str, str] | list[tuple[TokenId, str]] | None:
            if text == "_":
                return None
            match field:
                case "id" | "head":
                    if "-" in text:
                        id_range = text.split("-")
                        return int(id_range[0]), int(id_range[1])
                    elif "." in text:
                        return float(text)
                    else:
                        return int(text)

                case "feats":
                    feats = text.split("|")

                    feats_dict: dict[str, str] = {}
                    for feat in feats:
                        pair = feat.split("=")
                        feats_dict.update({pair[0]: pair[1]})
                    return feats_dict

                case "deps":
                    # TODO: implement dependency graph
                    return None

                case _:
                    return text

        values = {}

        for idx, field_text in enumerate(fields):
            field_name = field_for_index[idx]
            values.update(
                {field_for_index[idx]: _conversion_for_field(field_name, field_text)}
            )

        return Token(**values)


@dataclass
class Sentence:
    id: int | str
    tokens: list["Token"]
    text: str


@dataclass
class Token:
    id: TokenId
    form: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: dict[str, str] | None
    head: TokenId | None
    deprel: str | None
    deps: list[tuple[TokenId, str]] | None
    misc: str | None
