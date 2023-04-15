# Norwegian ML tagger

The code requires pytorch GPU environment with at least 32 GB RAM. If you have two 12 GB GPUs, you edit tag.py and set int\_classification\_device int\_tokenization\_device to different device IDs (e.g. 0 and 1). If you want use CPU you should edit the code accordingly.

The model files must be merged into one file using the following commands:

    cd models/classifier/
    cat xa* >pytorch_model.bin

    cd ../tokenization/
    cat f1/*> pytorch_model.bin
    cat f2/*> optimizer.pt

You can use tag.py to tag sentences. Give a filename as parameter which has a sentence per line.

    python3 tag.py file_name
