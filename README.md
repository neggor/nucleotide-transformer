# Nucleotide Transformer

[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3.8/library/index.html)
[![Jax Version](https://img.shields.io/badge/jax-0.3.25-informational)](https://jax.readthedocs.io/en/latest/)
[![license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)

Welcome to the InstaDeep Github repository of the [Nucleotide Transformer project](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3).

We are thrilled to open-source this work and provide the community with access to the code and pre-trained weights for eight genomics language models. This project was a collaboration with Nvidia and TUM, and the models were trained on DGX A100 nodes on Cambridge-1.

## Description 🧬

We present a comprehensive examination of foundational language models that were pre-trained on DNA sequences from whole-genomes.
Compared to other approaches, our models do not only integrate information from single reference genomes,
but leverage DNA sequences from over 3,200 diverse human genomes, as well as 850 genomes from a wide range of species,
including model and non-model organisms. Through robust and extensive evaluation,
we show that these large models provide extremely accurate molecular phenotype prediction compared to existing methods.

<img src="imgs/nt_results_rebuttal_2.png" alt= "Performance on downstream tasks" width="800" height="800">

*Fig. 1: The Nucleotide Transformer model accurately predicts diverse genomics tasks after fine-
tuning. We show the performance results across downstream tasks for fine-tuned transformer models. Error bars represent 2 SDs
derived from 10-fold cross-validation.*

Overall, our work provides novel insights related to the training and application
of language foundational models to genomics with ample opportunities of their applications in the field.

In this repository, you will find the following:

- Inference code for our models
- Pre-trained weights for all eight models
- Instructions for using the code and pre-trained models

## Get started 🚀

To use the code and pre-trained models, simply:

1. Clone the repository to your local machine.
2. Install the package by running `pip install .`.

You can then download and do the inference with any of our eight models in only a few lines of codes:
```python
import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="500M_human_ref",
    embeddings_layers_to_save=(20,),
    max_positions=32,
)
forward_fn = hk.transform(forward_fn)

# Get data and tokenize it
sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

# Initialize random key
random_key = jax.random.PRNGKey(0)

# Infer
outs = forward_fn.apply(parameters, random_key, tokens)

# Get embeddings at layer 20
print(outs["embeddings_20"].shape)
```
Supported model names are:
- **500M_human_ref** 
- **500M_1000G** 
- **2B5_1000G** 
- **2B5_multi_species**
- **50M_multi_species_v2** 
- **100M_multi_species_v2** 
- **250M_multi_species_v2** 
- **500M_multi_species_v2**

You can also run our models and find more example code in google colab [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/instadeepai/nucleotide-transformer/blob/main/examples/inference.ipynb)

The code runs both on GPU and TPU thanks to Jax!

## Nucleotide Transformers v2 models
Our second version Nucleotide Transformer v2 models include a series of architectural changes that proved more efficient: instead of using learned positional embeddings, we use Rotary Embeddings that are used at each attention layer and Gated Linear Units with swish activations without bias. These improved models also accept sequences of up to 2,048 tokens leading to a longer context window of 12kbp.
Inspired by Chinchilla scaling laws, we also trained our NT-v2 models on our multi-species dataset for longer duration (300B tokens for the 50M and 100M models; 1T tokens for the 250M and 500M model) compared to the v1 models (300B tokens for all four models).


## Embeddings retrieval
The transformer layers are 1-indexed, which means that calling `get_pretrained_model` with the arguments `model_name="500M_human_ref"` and `embeddings_layers_to_save=(1, 20,)` will result in extracting embeddings after the first and 20-th transformer layer. For transformers using the Roberta LM head, it is common practice to extract the final embeddings after the first layer norm of the LM head rather than after the last transformer block. Therefore, if `get_pretrained_model` is called with the following arguments `embeddings_layers_to_save=(24,)`, the embeddings will not be extracted after the final transformer layer but rather after the first layer norm of the LM head.

## Tokenization :abc:

The models are trained on sequences of length up to 1000 tokens, including the 
\<CLS> token prepended automatically to the beginning of the sequence. The tokenizer 
starts tokenizing from left to right by grouping the letters "A", "C", "G" and "T" in
6-mers. The "N" letter is chosen not to be grouped inside the k-mers, therefore 
whenever the tokenizer encounters a "N", or if the number of nucleotides in the sequence
is not a multiple of 6, it will tokenize the nucleotides without grouping them. Examples 
are given below: 

```python
dna_sequence_1 = "ACGTGTACGTGCACGGACGACTAGTCAGCA" 
tokenized_dna_sequence_1 = [<CLS>,<ACGTGT>,<ACGTGC>,<ACGGAC>,<GACTAG>,<TCAGCA>]

dna_sequence_2 = "ACGTGTACNTGCACGGANCGACTAGTCTGA" 
tokenized_dna_sequence_2 = [<CLS>,<ACGTGT>,<A>,<C>,<N>,<TGCACG>,<G>,<A>,<N>,<CGACTA>,<GTCTGA>]
```

All the transformers can therefore take sequences of up to 5994 nucleotides if there are
no "N" inside. 


## Acknowledgments 🙏

We thank Maša Roller, as well as members of the Rostlab, particularly Tobias Olenyi, Ivan Koludarov,
and Burkhard Rost for constructive discussions that helped identify interesting research directions.
Furthermore, we extend gratitude to all those who deposit experimental data in public databases, to
those who maintain these databases, and those who make analytical and predictive methods freely
available. We also thank the Jax development team.

## Citing the Nucleotide Transformer 📚

If you find this repository useful in your work, please add the following citation to our [associated paper](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2):

```bibtex
@article{dalla2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza Revilla, Javier and Lopez Carranza, Nicolas and Henryk Grywaczewski, Adam and Oteri, Francesco and Dallago, Christian and Trop, Evan and Sirelkhatim, Hassan and Richard, Guillaume and others},
  journal={bioRxiv},
  pages={2023--01},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

If you have any questions or feedback on the code and models, please feel free to reach out to us.

Thank you for your interest in our work!
