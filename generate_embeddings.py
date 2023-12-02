import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model
from Bio import SeqIO
import pandas as pd
import csv
from tqdm import tqdm
def read_fasta(file_path):
    ids = []
    sequences = []
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq).upper())
    return ids, sequences

# Get pretrained model
sequences_url = "../LLMArabidopsis/data/promoter_region/arabidopsis_promoters.fa"
ids, sequences = read_fasta(sequences_url)
embeddings_array = np.zeros((len(sequences), 1024))
gene_names = []

parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="500M_multi_species_v2",
            embeddings_layers_to_save=(24,),
                max_positions=1000,
                )

forward_fn = hk.transform(forward_fn)
for i in tqdm(range(4, len(sequences), 4)):
    #print("iter:", i)
    #gene_names.append(gene_id.split('.')[0])
    sequence = sequences[i-4:i]
    [gene_names.append(x.split(".")[0]) for x in ids[i-4:i]] 
    sequence = [s[:5994] if len(s) > 5994 else s for s in sequence]
    #print(mean_embeddings.shape)
    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequence)]
    tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequence)]

    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

    # Initialize random key
    random_key = jax.random.PRNGKey(0)

    # Infer
    outs = forward_fn.apply(parameters, random_key, tokens)

    # Get embeddings at layer 20
    embeddings = outs["embeddings_24"][:, 1:, :]
    padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)
    masked_embeddings = embeddings * padding_mask  # multiply by 0 pad tokens embeddings
    sequences_lengths = jnp.sum(padding_mask, axis=1)
    mean_embeddings = np.array((jnp.sum(masked_embeddings, axis=1)\
            / sequences_lengths))
    embeddings_array[i-4:i, :] = mean_embeddings
with open("../LLMArabidopsis/data/embeddings.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(embeddings_array)
with open("../LLMArabidopsis/data/gene_names.csv", "w", newline = '') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows([item] for item in gene_names)
    
