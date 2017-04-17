# insight-accent

**Repeating something you say in a different voice**

## Summary

My project for [Insight AI](http://insightdata.ai/). A three week project to explore applications of Style Transfer to speech data.


## Dependencies

Libraries: `tensorflow`, `numpy`, `matplotlib`

Reservoir sampling models can be run on a cpu with no setup or long training.
The encoders and classifier take prohibitively long to train on CPU, so a GPU is recommended.

## Usage

Their are three models, each requiring different data and setup.

### Reservoir Computing

Run `reservoir_transfer.py` replacing the argument with your content and style file, no pretraining required.

### Variational Autoencoder

Requires many samples of speech from content and style classes to train on. Training on a GPU is recommended.

The model will do a train,val, test split, train on the training data, checkpoint regularly, and output a few examples of inference on test data.

Run `autoencoder_transfer.py` replacing the argument with a directory to your audio files supplied. Currently supports two directories of `.wav` files (`male` and `female` for the provided example).

### Pretrained classifier

**This solution has not resulted in satisfying results, use at your own risk.**

Train a classifier (example supplied in `train_classifier.py`) and then use it in `pretrained_transfer.py`.

## Data source

[DAPS (Device and Produced Speech) Dataset](https://archive.org/details/daps_dataset)

[Alice in Wonderland Audioobook](http://etc.usf.edu/lit2go/1/alices-adventures-in-wonderland/)

## References/Inspiration

### Papers

[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

[Texture Synthesis Using Shallow Convolutional Networks with Random Filters](https://arxiv.org/abs/1606.00021)

[Discrete Variational Autoencoders](https://arxiv.org/abs/1609.02200)

### Online resources

[FF Labs Autoencoders](http://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html)

[VAE for video style transfer](http://int8.io/variational-autoencoder-in-tensorflow/)

[Paperspace Autopencoders](https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/#aae)

[Inspiration for speech classifier](https://yerevann.github.io/2015/10/11/spoken-language-identification-with-deep-convolutional-networks/)

[Using Reservoir Computing for Audio Style](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/)

[Using Reservoir Computing for Audio Style](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/)
