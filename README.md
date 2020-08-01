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
| Beam Width | 1     | 3     | 5     |   |
|------------|-------|-------|-------|---|
| Bleu_1     | 0.570 | 0.563 | 0.555 |   |
| Bleu_2     | 0.418 | 0.413 | 0.405 |   |
| Bleu_3     | 0.296 | 0.300 | 0.293 |   |
| Bleu_4     | 0.209 | 0.219 | 0.215 |   |
| METEOR     | 0.211 | 0.219 | 0.220 |   |
| ROUGE_L    | 0.467 | 0.472 | 0.470 |   |
| CIDEr      | 0.432 | 0.436 | 0.423 |   |
| SPICE      | 0.122 | 0.126 | 0.125 |   |


## References

[1] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
