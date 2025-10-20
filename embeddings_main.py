from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads

import torch

from file_processing_steps import read_markdown_file

if __name__ == "__main__":

    model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
    tokenizer = AutoTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased", do_lower_case=False
    )

    # TODO: structure in a pipeline and get the similarity

    # encode the whole text
    input_ids = tokenizer.encode(
        read_markdown_file(
        r""
        ).text,
        return_tensors="pt"
        ).squeeze(0)


    # count the number of tokens
    token_count = len(input_ids)

    # split in decoded chunks
    chunks = []
    # (max-1) - overlap
    for i in range(0,len(input_ids),511-50):
        chunks.append(input_ids[i:i+511])




    # encode back each chunk
    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            re_encoded = chunk.unsqueeze(0)
            outs = model(re_encoded)
            encoded = outs[0][0,1:-1]
            embeddings.append(encoded)


    # take mean of all chunks
    result = torch.cat(embeddings,dim=0).mean(dim=0)

    print(result)
