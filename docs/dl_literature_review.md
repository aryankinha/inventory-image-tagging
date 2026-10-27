# Deep learning literature review (Phase 2)

Phase 1 used fixed descriptions of a picture (pixels, HOG) with classical classifiers. Phase 2 asks whether a small CNN, which learns its own descriptions, tells apart the categories that Phase 1 mixed up (Shirt, T-shirt, Pullover, Coat). Each source below is one idea we use. Each reference was checked for title, authors and venue, and we summarise its main idea; we did not re-run their experiments.

| Source | Idea | Where we use it | Limit when applying it |
|---|---|---|---|
| LeCun, Bottou, Bengio and Haffner (1998), [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791) | Convolution, weight sharing and pooling learn local picture features | The two or three convolution blocks | Digits and documents, not clothing |
| Krizhevsky, Sutskever and Hinton (2012), [ImageNet classification with deep convolutional neural networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | ReLU units, dropout and augmentation let a deep CNN generalise | ReLU after every layer; dropout; flips and shifts | Far larger network and dataset than ours |
| Srivastava et al. (2014), [Dropout](https://jmlr.org/papers/v15/srivastava14a.html), JMLR 15:1929-1958 | Randomly switching off units during training reduces overfitting | Dropout 0.4 in the dense layers (ablation: on and off) | Gains can be small; we measure them rather than assume them |
| Ioffe and Szegedy (2015), [Batch normalization](https://arxiv.org/abs/1502.03167) | Normalising layer inputs speeds up training and can regularise | Batch norm after each convolution (ablation: on and off) | Behaves differently in training and evaluation mode |
| Kingma and Ba (2015), [Adam](https://arxiv.org/abs/1412.6980), ICLR | Adaptive step sizes per weight | Optimiser, learning rate 0.001 | We tuned nothing else about it |
| Prechelt (1998), Early stopping, but when?, in *Neural Networks: Tricks of the Trade*, LNCS 1524 | Stop when validation error stops improving | Stop after 5 epochs without better validation loss; keep the best epoch | A patience value is a choice, not a law |
| Simard, Steinkraus and Platt (2003), [Best practices for convolutional neural networks](https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/), ICDAR | Small distortions of training pictures give the network more varied examples | Random shifts of up to 2 pixels and left-right flips | They studied digits; mirroring is only safe when the category survives it, which holds for these clothes |
| Razavian et al. (2014), [CNN features off-the-shelf](https://arxiv.org/abs/1403.6382), CVPR Workshops | Features from a CNN, fed to a linear SVM, make a strong baseline | The hybrid model: CNN features into a classical classifier | They used a large pretrained network; ours is trained on this dataset only |
| Tang (2013), [Deep learning using linear support vector machines](https://arxiv.org/abs/1306.0239) | Replacing the softmax layer with a linear SVM can help | Same idea, tested here with logistic regression, linear SVM and Random Forest | Tang trains the SVM together with the network; we fit it afterwards |
| Xiao, Rasul and Vollgraf (2017), [Fashion-MNIST](https://arxiv.org/abs/1708.07747) | The benchmark and its official split | Data and the official test set | A familiar benchmark: results do not transfer to a real store catalogue |

## What this review tells us to test

1. A small CNN with batch norm, dropout, augmentation and early stopping should beat fixed descriptions, but each ingredient is only *claimed* to help. We remove them one at a time and measure (ablation).
2. Reported gains from these ingredients on Fashion-MNIST are small and can sit inside seed-to-seed noise. We therefore train every configuration with three seeds and report the spread.
3. CNN features with a classical classifier on top is an established pairing. We test whether it beats the CNN's own final layer, and use features from an untrained CNN as a control.

## Not used, on purpose

Residual networks, pretrained models, learning-rate schedules, attention and generative augmentation. They add moving parts we could not explain in a viva, and a 28x28 grayscale problem does not need them.
