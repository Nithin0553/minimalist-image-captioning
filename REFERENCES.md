# References and source attribution

This project uses published methods through maintained Python libraries. It does not copy or
vendor source code from image-captioning tutorials or third-party repositories. Flickr8k images
and captions, pretrained ResNet-18 weights, checkpoints, and generated experiment files are kept
outside Git.

## Dataset

- Micah Hodosh, Peter Young, and Julia Hockenmaier. "Framing Image Description as a Ranking
  Task: Data, Models and Evaluation Metrics." *Journal of Artificial Intelligence Research*,
  47:853-899, 2013. <https://doi.org/10.1613/jair.3994>
- Flickr8k is downloaded from the
  [Kaggle dataset mirror](https://www.kaggle.com/datasets/adityajn105/flickr8k). Dataset images
  remain subject to their original Flickr terms, and the captions are credited to the Flickr8k
  authors.

## Architecture

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image
  Recognition." *CVPR*, 2016. <https://arxiv.org/abs/1512.03385>
- Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares,
  Holger Schwenk, and Yoshua Bengio. "Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation." *EMNLP*, 2014.
  <https://aclanthology.org/D14-1179/>

## Evaluation

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. "Bleu: a Method for Automatic
  Evaluation of Machine Translation." *ACL*, 2002. <https://aclanthology.org/P02-1040/>
- Satanjeev Banerjee and Alon Lavie. "METEOR: An Automatic Metric for MT Evaluation with
  Improved Correlation with Human Judgments." *ACL Workshop*, 2005.
  <https://aclanthology.org/W05-0909/>
- Laurens van der Maaten and Geoffrey Hinton. "Visualizing Data using t-SNE." *Journal of
  Machine Learning Research*, 9:2579-2605, 2008.
  <https://www.jmlr.org/papers/v9/vandermaaten08a.html>

## Software interfaces

- [PyTorch](https://docs.pytorch.org/docs/stable/index.html) supplies tensor operations,
  optimization, and the GRU implementation.
- [TorchVision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
  supplies the official ImageNet pretrained encoder weights.
- [NLTK translation metrics](https://www.nltk.org/api/nltk.translate.html) supplies corpus BLEU
  and METEOR.
- [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
  supplies the two-dimensional visualization transform.

The direct 512-to-6 bottleneck, project-specific preprocessing pipeline, data validation,
checkpoint handling, experiment orchestration, sensitivity analysis, plots, and tests in this
repository were implemented specifically for this COSC 6324 course project.
