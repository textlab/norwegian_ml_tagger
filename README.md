# Norwegian ML tagger

The code requires pytorch GPU environment. It loads two models into the GPU RAM. These models also work well with multiple 12 GB GPUs. If want to you use them on the same GPU a minimum of 32 GB RAM is recommended. You can edit tag.py according to the GPU configuration and set int\_classification\_device int\_tokenization\_device to different device IDs (e.g. 0,1,2... and -1 for cpu).

The model files must be merged into one file using the following commands:

    cd models/classifier/
    cat xa* >pytorch_model.bin

    cd ../tokenization/
    cat f1/*> pytorch_model.bin
    cat f2/*> optimizer.pt

You can use tag.py to tag sentences. Give a filename as parameter which has a sentence per line.

    python3 tag.py file_name

# License

[MIT license](https://github.com/textlab/norwegian_ml_tagger/blob/master/LICENSE)
