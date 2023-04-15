# Norwegian ML tagger

The code requires pytorch GPU environment. It loads two models into the GPU RAM. Each of these models work well with 12 GB GPUS. But if you use them on the same GPU a minimum of 32 GB GPU is recommended. You edit tag.py and set int\_classification\_device int\_tokenization\_device to different device IDs (e.g. 0 and 1). If you want use CPU you should edit the code accordingly.

The model files must be merged into one file using the following commands:

    cd models/classifier/
    cat xa* >pytorch_model.bin

    cd ../tokenization/
    cat f1/*> pytorch_model.bin
    cat f2/*> optimizer.pt

You can use tag.py to tag sentences. Give a filename as parameter which has a sentence per line.

    python3 tag.py file_name
