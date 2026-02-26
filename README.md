# Norwegian ML tagger

> [!CAUTION]

**Important:** This repository is no longer maintained. The tagger is maintained under the HuggingFace repositories with four different sizes. Please refer to these repositories. The list of these repositories according to the size of models is given below:

- [humit-tagger-xs (15M)](https://huggingface.co/Humit-Oslo/humit-tagger-xs)
- [humit-tagger-small (40M)](https://huggingface.co/Humit-Oslo/humit-tagger-small)
- [humit-tagger-base (123M)](https://huggingface.co/Humit-Oslo/humit-tagger-base)
- [humit-tagger-large (323M)](https://huggingface.co/Humit-Oslo/humit-tagger-large)


## The content of this page before moving to the new repositories.


The code requires pytorch GPU environment. It loads two models into the GPU RAM. These models also work well with multiple 12 GB GPUs. If want to you use them on the same GPU a minimum of 32 GB RAM is recommended. You can edit tag.py according to the GPU configuration and set int\_classification\_device int\_tokenization\_device to different device IDs (e.g. 0,1,2... and -1 for cpu).

The model files must be merged into one file using the following commands:

    cd models/classifier/
    cat xa* >pytorch_model.bin

    cd ../tokenization/
    cat f1/*> pytorch_model.bin
    cat f2/*> optimizer.pt

You can use tag.py to tag sentences. Give a filename as parameter which has a sentence per line.

    python3 tag.py file_name

Cite this work as the following:

    @inproceedings{haug-etal-2023-rules,
        title = {Rules and neural nets for morphological tagging of {N}orwegian - Results and challenges},
        author = {Haug, Dag  and
          Yildirim, Ahmet  and
          Hagen, Kristin  and
          N{\o}klestad, Anders},
        booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
        month = may,
        year = {2023},
        address = {T{\'o}rshavn, Faroe Islands},
        publisher = {University of Tartu Library},
        url = {https://aclanthology.org/2023.nodalida-1.43},
        pages = {425--435}
    }

# License

[MIT license](https://github.com/textlab/norwegian_ml_tagger/blob/master/LICENSE)
