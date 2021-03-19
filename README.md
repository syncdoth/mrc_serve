# How to serve the mrc model from Pororo Library

## Download dependencies

First, download the python dependencies. As always, it is recommended to create
a virtual enviroment beforehand.

```
pip install -r requirements.txt
```

Also, you need a jdk 11 installed.

1. if you use conda environment, you can do

```
conda install openjdk -c conda-forge
```

2. Better, you can download from Oracle hompage the tar.gz file, and then
untar it:

```
tar -xvf <openjdk file name>.tar.gz
```
* Then, you will have to set up JAVA_HOME enviroment variable. Put the following
in your .bashrc (or .cshrc or .zshrc):

```
export JAVA_HOME=<openjdk folderpath>
export PATH="$JAVA_HOME/bin:$PATH"
```

## Download pororo pretrained files and dictionary

This can be done by calling `predownload.py` file.

```
python predownload.py
```

This will save the files to `~/.pororo` folder. Then, you should move all the
files to one directory for serving:

```
mkdir mrc_model
mv ~/.pororo/bert/brainbert.base.ko.korquad/* mrc_model
mv ~/.pororo/tokenizers/bpe32.ko/* mrc_model
```

## Modify handler.py

As because of some conflicts in the fairseq repo, you have to set the data path
manually. Go to line 42 of `handler.py` file, and change the path to the absolute
path to the folder that you store pretrained checkpoints (the folder you created
in the above step).

## create model archive

archive the model for serving.
```
torch-model-archiver --model-name mrc --version 1.0 \
--serialized-file mrc_model/model.pt \
--extra-files "mrc_model/merges.txt,mrc_model/vocab.json,,mrc_model/input0/dict.txt" \
--handler handler.py
```

This creates `mrc.mar` in your current directory.

## configurations

open `config.properties` file, and switch up some parts. e.g. change the port
numbers of the addresses.

## serve

First create a `model_store` directory. Then, move the `mrc.mar` file into the
folder.

```
mkdir model_store
mv mrc.mar model_store
```

Then, start the serving!

```
torchserve --start --ts-config config.properties --model-store model_store \
--models mrc=mrc.mar
```
