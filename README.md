# Image Captioning Using CNN/LSTM

This repository contains an implementation of image captioning based on neural network (CNN + RNN). The model first extracts the image feature by CNN and then generates captions by RNN. Here, CNN is ResNet-152 and RNN is LSTM .

Beam Search was used to predict the caption of images.

# Image Captioning Models in PyTorch

Some code is borrowed from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

Here are the implementations of Google-NIC[1] with PyTorch and Python3.

## Usage:

### Download the repositories:

```bash
# download coco Python API
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI
$ make
$ sudo make install
$ sudo python setup.py install
$ cd ../../

# download coco evaluation
$ git clone https://github.com/salaniz/pycocoevalcap

# download this respository
$ git clone https://github.com/kellyzhu11/Image-Captioning.git
$ cd ./image_captioning/
```
### Download and process the data

```bash
$ pip install -r requirements.txt
$ chmod +x download.sh
$ ./download.sh
$ python build_vocab.py   
```
### train the model
```bash
$ python train.py
```

### generate captions using trained model
```bash
$ python test.py
```

### calculate scores for generated captions
```bash
$ python evaluate.py
```
### manual inspection of captions and iamges

run visualize_caption.ipynb

## Results
| Beam Width | 1     | 3     | 5     | 10    |
|------------|-------|-------|-------|-------|
| Bleu_1     | 0.571 | 0.566 | 0.559 | 0.549 |
| Bleu_2     | 0.418 | 0.416 | 0.409 | 0.398 |
| Bleu_3     | 0.296 | 0.302 | 0.297 | 0.288 |
| Bleu_4     | 0.209 | 0.221 | 0.218 | 0.212 |
| METEOR     | 0.208 | 0.217 | 0.219 | 0.219 |
| ROUGE_L    | 0.467 | 0.474 | 0.472 | 0.468 |
| CIDEr      | 0.424 | 0.434 | 0.425 | 0.498 |
| SPICE      | 0.120 | 0.123 | 0.123 | 0.123 |


## References

[1] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
