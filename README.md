<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

### What is semi-supervised learning?

Semi-supervised learning uses a small portion of labeled data and a large amount of unlabeled data to train a model. This is in contrast to supervised learning, in which all data is labeled (regression, random forest, k nearest neighbors) and unsupervised learning (k-means clustering, hierarchical clustering), where all data are unlabeled. 

### Why do we need semi-supervised learning?

Convolutional neural networks are the prevailing approach in computer vision, and require large amounts of labeled data, which can be tricky to acquire. Semi-supervised learning is often used for image classification, which is our topic of interest.

Approaches in image classification focus on the areas of consistency regularization and pseudo-labeling. Consistency regularization encourages the model to produce consistent predictions under different perturbations of the input, to improve the generalizability of the model. Pseudo-labeling generates labels for the unlabeled data that are used in the learning process, acting as true labels to use within the loss function. Early methods of pseudo-labeling use the network predictions as labels, but only use the pseudo labels during a warm-up stage (2). A later approach (3) uses the network predictions to create hard labels (uses the class with the highest probabability as the label), and adds an uncertainty weight for each sample loss based on distance from its k-nearest neighbors, a loss term to encourage compact and well-separated clusters, and a term to encourage consistency between perturbed samples. Recent work (4) introduces graph-based label propagation, in which the model is trained on the labeled and pseudo-labeled data, and a nearest-neighbor graph based on the network is created. The pseudo-labels are propagated from the labeled data points to the unlabeled points. Then, uncertainty scores are added based on the softmax predictions and class imbalance.

### The Paper

We will describe the pseudo-labeling methodology proposed in the paper "Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning" by Arazo et al. published in 2020. 

### Set up and Notation

For this implementation of pseudo-labeling, the authors specified data $$D$$ with $$N$$ observations split into labeled and unlabeled sets $$D_l=\{(x_i,y_i)\}^{N_l}_{i=1}$$ and $$D_u=\{x_i\}^{N_u}_{i=1}$$, respectively (Figure 1). Here, the $$y_i$$ values are one-hot encoded for $$C$$ possible classes. For example, if there are 3 possible classes and an observation is in the 3rd class, this would be represented as $$y=(0,0,1)$$. 

| ![Figure 1](Figure 1.png) | 
|:--:| 
|**Figure 1: Labeling Structure of Data** |

This method uses soft pseudo-labeling which differs from hard pseudo-labeling in that it does not store the predicted classes, but rather the predicted softmax probabilities of each class. The pseudo-labels are denoted as $$\tilde{y}$$, and $$\tilde{y}=y$$ for the labeled observations. For example, if there are 3 classes, and the model predicts an observation is in class 1 with probability 0.2, class 2 with probability 0.2, and class 3 with probability 0.6, the soft pseudo-label is $$\tilde{y}=(0.2,0.2,0.6)$$.

### Convolutional Neural Networks

The model used by the authors is a convolutional neural network (CNN) with a softmax outcome function, specified as $$h_{\theta}(x)$$, where $$\theta$$ represents the parameters of the network. 

The difference between a CNN and a standard feed-forward neural network we have seen comes in the formulation of the layers and how it attempts to classify images. CNNs work by first identifying small, local features that are then combined together to form broader features, which are then used to calculate the class probabilities. This creates a hierarchical structure in the network, which is illustrated in Figure 2, taken from ISL page 412 (5). In this figure, we can see that the network identifies areas of lines, shapes, and colors, which are then combined to identify larger features such as eyes and ears. 

| ![Figure 2](Figure 2.png) | 
|:--:|
|**Figure 2 (ISL Figure 10.6): Hierarchical Structure of CNNs** |

This process works by using two types of hidden layers: convolution layers and pooling layers. Convolution layers contain a series of convolution filters, which determine whether a small feature is present in the image by going through the entire image in sections the same size as the filter. This is achieved by matrix multiplying the the data in each section by the filter values to create a new matrix of values called the convolved image. If the convolved image has large values, this means that the image section contains features similar to the feature the filter is trying to identify. The convolved images are combined to create a feature map for each filter, which is then passed on to the next layer. The values in each filter, which represent the feature it is identifying,  are comparable to the weight matrices we have seen in feed-forward networks and are the parameters $$\theta$$ being optimized when training the CNN. The number of filters in a layer is analogous to the width of a layer in the networks we have previously seen (5).

Pooling layers, the second type of layer used in CNNs, are essentially a form of dimension reduction that reduce a large image into a smaller summary image. One common method is max pooling, which looks at each section in an image and stores only the maximum value found in that section. These layers always come after a convolution layer, although there may be multiple convolution layers before a pooling layer, and therefore reduce the size of the feature maps created by each filter. The combination of convolution and pooling layers is repeated until the feature maps have low dimension, at which point they are flattened into individual units and fed to fully-connected layers before classification with the softmax output layer. An example of a CNN architecture is shown in Figure 3 from ISL page 416 (5). 

| ![Figure 3](Figure 3.png) | 
|:--:| 
| **Figure 3 (ISL Figure 10.8): Sample CNN Architecture** |

As with standard feed-forward networks we saw in class, the CNN used in this paper has a feed-forward structure and uses backpropagation to calculate the gradients to optimize the $$\theta$$ parameters using mini-batch gradient descent. The CNN can also be tuned by adjusting the number, size, and type of the layers, as well as other features such as regularization and dropout terms, like we have previously seen. 

### Loss Function and Regularization
In order to train the CNN, the categorical cross-entropy loss function is used to optimize the parameters $$\theta$$. It has the form 

$$\ell^*=-\sum^N_{i=1}\tilde{y_i}^T\log(h_{\theta}(x_i))$$

where $$h_{\theta}(x)$$ are the predicted probabilities for each class, and an element-wise logarithm is used.

Additionally, 2 regularization terms are included in the loss function to address 2 different issues. In previous assignments, we have seen the tendency for predictions to all fall into the larger class, particularly with unbalanced data, which minimizes the amount of error. The first regularization aims prevent predictions from all being of the same class to minimize this issue, and it has the form 

$$R_A=\sum^C_{c=1}p_c\log\left(\frac{p_c}{\bar{h}_c}\right)$$

Here, $$p_c$$ represents the prior distribution of the probability of being in class $$c$$, which is assumed to be uniform with $$p_c=1/C$$. $$\bar{h}_c$$ is the softmax probability for class $$c$$ averaged across all observations, which is estimated by averaging the probabilities obtained within each mini-batch. If there is perfect agreement between the prior probability and the predicted probability for class $$c$$ ie an even probability distribution between classes, then $$\log(p_c/\bar{h}_c)=\log(1)=0$$, and this regularization term has no effect. If the average predicted probability for class $$c$$ is either very large or very small, ie the predicted probability of being in one class is much larger than the others, the $$\log(p_c/\bar{h}_c)$$ term becomes large, which then penalizes the loss function. 

The second regularization is needed when using soft pseudo-labels instead of hard labels. Without this regularization, the algorithm can get caught in local minima which prevents convergence to the global minima. This regularization avoids these local minima by concentrating the probability distribution of each label to one class, using the entropy averaged over all observations

$$R_H=-\frac{1}{N}\sum^N_{i=1}\sum^C_{c=1}h^c_{\theta}(x_i)\log(h^c_{\theta}(x_i))$$

where $$h_{\theta}^c(x_i)$$ is the $$c$$th softmax probability of from $$h_{\theta}(x_i)$$. Entropy represents the amount of uncertainty in predicting the class of the outcome and is given by $$-\sum^C_{c=1}h^c_{\theta}(x_i)\log(h^c_{\theta}(x_i))$$ portion of the regularization (6). When the predicted probabilities are very similar, indicating high uncertainty in prediction, the entropy will be high, and when there is less uncertainty, the entropy will be lower. For example, predicted probabilities $$h_{\theta}(x_i)=(.3,.3,.4)$$ have entropy equal to 1.09, whereas the predicted probabilities $$h_{\theta}(x_i)=(.9,.05,.05)$$ have entropy 0.394. Therefore, this regularization term will apply a larger penalty to the loss function when entropy is high, which encourages the probability of one class to be larger than the others. Like with the first regularization term, the average entropy of all observations is estimated by averaging over the observations in each mini-batch. 

It may seem like these regularizations are contradictory, but their weights are adjustable so they're effects don't just cancel out. When we combine the cross-entropy loss with the regularizations, we get the penalized loss function 

$$\ell=\ell^*+\lambda_AR_A+\lambda_HR_H$$

where the $$\lambda$$ values control the amount of regularization, which we have previously seen with L1 and L2 regularization. 

### The Algorithm

As previously stated, the model used in this paper is a convolution neural network (CNN) that trains similarly to the algorithms we have seen in class and in homework assignments. The network parameters are initialized randomly and then optimized using mini-batch gradient descent by training on the data. 

In order to optimize the network parameters, we first need to get initial soft pseudo-labels for the unlabeled data. To do so, the CNN is trained on the labeled data, $$D_l$$, for 10 epochs as a "warm-up". Then, the warm-up model is used to fit initial softmax predictions to the unlabeled data, $$D_u$$. The combined labeled and pseudo-labeled data are then used to further train the network. 

| ![Figure 4](Figure 4.png) | 
|:--:| 
| **Figure 4: Basic Pseudo-Labeling Algorithm** |

In each epoch, the parameters $$\theta$$ are updated using gradient descent on the loss function $$\ell$$ for each mini-batch, and the softmax predictions for each of the unlabeled observations are stored. Recall the basic loss function is the categorical cross-entropy loss

$$\ell^*=-\sum^N_{i=1}\tilde{y_i}^T\log(h_{\theta}(x_i))$$

As mentioned in the notation section, both $$\tilde{y}_i$$ and $$h_{\theta}(x_i)$$ are vectors, so multiplying the transpose of $$\tilde{y}_i$$ by the log of the softmax probability vector results in the dot product between the two. For labeled observations, $$\tilde{y}_i$$ is the true vector (eg $$\tilde{y}_i=(0,0,1)$$), so this just outputs the log of the predicted probability of being in the true class. For pseudo-labeled observations, $$\tilde{y}_i$$ is also a vector of softmax probabilities, which means the loss contribution is the dot product of the previous softmax predictions and the log of the new softmax predictions. The regularization terms $$R_H$$ and $$R_A$$ are also calculated (recall they are mini-batch averages), so that the gradient of $$\ell$$ can be calculated to obtain the stepping directions for $$\theta$$.

The new softmax predictions $$h_{\theta}(x_i)$$ for each of the pseudo-labeled observations are stored for each mini-batch in an epoch. At the end of the epoch, the soft pseudo-labels are updated using $$\tilde{y}_i=h_{\theta}(x_i)$$, and these new labels are used in the next epoch (Tanaka et al, 2018). These steps repeat until the specified number of epochs has been reached. An overview of the algorithm is visualized in Figure 4, and Figure 5 shows the general update procedure for the CNN parameters $$\theta$$ and the soft pseudo-labels $$\tilde{y}_i$$. 

| ![Figure 5](Figure 5.png) | 
|:--:| 
| **Figure 5: Updating Parameters and Pseudo-Labels** |


### Confirmation Bias

When network predictions are incorrect, these predictions are reinforced since the network predictions are used as labels for the unlabeled samples. Overfitting to these incorrect predictions is called confirmation bias. 

To deal with this confirmation bias, the authors use a method known as mixup data augmentation, which uses data augmentation (generating new data by warping existing data(8)) and label smoothing (a technique that increases label noise(9)). The mixup method trains the model on sample pairs ($$x_p$$ and $$x_q$$) and corresponding output labels ($$y_p$$ and $$y_q$$) (Figure 6)

$$x=\delta x_p + (1-\delta)x_q$$,
$$y=\delta y_p + (1-\delta)y_q$$

Where $$\delta$$ is randomly sampled from a beta distribution Be($$\alpha,\beta$$) with $$\alpha=\beta$$

| ![Figure 6](mixup.png) | 
|:--:| 
| **Figure 6: Implementing Mixup** |

The combined version of y can be included in the loss function, producing an updated loss equation:

$$
\ell^* = - \Sigma_{i=1}^N\delta[\tilde{y}_{i,p}^{T}log(h_{\theta}(x_i))] +  (1- \delta)[\tilde{y}_{i,q}^{T}log(h_{\theta}(x_i))]
$$

Combining $$y_p$$ and $$y_q$$ reduces prediction confidence, thus reducing overfitting to the predictions.

Including mixup in training generates softmax output $$h_{\theta}(x)$$ using the mixed input x, so a second forward pass using the original images is required to compute predictions that aren’t mixed. 

When there are few labeled samples, mixup may not be able to deal with confirmation bias effectively by itself. To improve the quality of the pseudo labels, the authors mention that previous studies have oversampled the labeled samples per mini batch to reduce confirmation bias and reinforce correct labels. 

This loss can be split into two sample terms, one based on the labeled samples and one based on the unlabeled samples. 

$$
\ell^* = N_l\bar{\ell_l} + N_u\bar{\ell_u}
$$

where $$N_l$$ and $$N_u$$ are the number of labeled and unlabelled samples, and 
$$
\bar{\ell}_l = \frac{1}{N_l}\Sigma_{i=1}^{N_l}\ell_l^{(i)}
$$
 is the average loss for the labeled samples, and $$l_u$$ is the average loss for the unlabeled samples (using the same form as above). 

When $$N_l$$ << $$N_u$$, the network focuses more on fitting the unlabeled samples correctly compared to the labeled samples. To counteract this, $$N_l$$ can be weighted more heavily or the labeled samples can be oversampled. The authors choose to oversample since it means that the model gets more chances to adjust its parameters to fit the labeled samples.

The authors tested the effect of mixup using the “two moons” data and showed that mixup, combined with oversampling, reduced confirmation bias, and gave a smooth rather than linear boundary (Figure 7). 

| ![Figure 7](Twomoon.png) | 
|:--:| 
| **Figure 7: Two Moons Data (1)** |

### Their Results and Conclusions
To compare the effectiveness of their pseudo labeling algorithm to previous methods, the authors evaluated their method on four datasets commonly used for testing image classification: CIFAR10, CIFAR100(10), SVHN(11), and Mini-ImageNet(12). 

The authors normalized the images to the dataset mean and standard deviation, which aids convergence, and then augmented the data by implementing random horizontal flips, pixel translations, and color jitter. For training, they used stochastic gradient descent (SGD), with momentum = 0.9, weight decay= 10^-4, and batch size of 100. They begin training with a high learning rate of 0.1 for CIFAR and SVHN and .2 for MiniImageNet, and then it is divided by 10 twice throughout the training process. CIFAR and MiniImageNet are trained for 400 epochs with a 10 epoch warmup, and SVHN is trained for 150 epochs with a 150 epoch warmup. The regularization weights we mentioned previously: $$\lambda_A$$ and $$\lambda_H$$ are set to 0.8 and 0.4, and they include dropout and weight normalization in their networks. 

For CIFAR 10/100, networks without mixup were overfitting on the predictions and had a high training accuracy. Error was reduced when mixup was included, and performed better when more labels were included. In past studies, network architecture played a role in the success of different approaches, so the authors tested across various architectures using pseudo labeling with mixup and at least 16 labeled samples per mini batch. They found that their methods work well across all of the tested architectures except one. 

Next, they compared their methods to state of the art approaches for SSL, which either used consistency regularization approaches or pseudo labeling approaches. The authors’ method outperforms consistency regularization methods and purely pseudo labeling methods, and it continues to be more effective even with a decreased number of labels. 

Overall, a semi-supervised learning approach using soft pseudo labels with mixup, a minimum number of labeled samples per mini batch, dropout, and data augmentation, outperforms other approaches in four datasets and across different network architectures. The authors conclude that it is a simple and accurate alternative to consistency regularization.



### References

1.	Arazo E, Ortego D, Albert P, O’Connor NE, McGuinness K. Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning. In: 2020 International Joint Conference on Neural Networks (IJCNN) [Internet]. 2020 [cited 2024 May 10]. p. 1–8. Available from: https://ieeexplore.ieee.org/abstract/document/9207304?casa_token=UIP9QUw65eQAAAAA:Cm1ronqlDtbMAWzI8-8t85b6oCP1c4EjZjO_jz_cPOb8_sjvse4HzKTzdgPqgWuNcU29QG6okDE
2.	Lee DH. Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. ICML 2013 Workshop Chall Represent Learn WREPL. 2013 Jul 10;
3.	Shi W, Gong Y, Ding C, Ma Z, Tao X, Zheng N. Transductive Semi-Supervised Deep Learning Using Min-Max Features. In: Ferrari V, Hebert M, Sminchisescu C, Weiss Y, editors. Computer Vision – ECCV 2018 [Internet]. Cham: Springer International Publishing; 2018 [cited 2024 May 9]. p. 311–27. (Lecture Notes in Computer Science; vol. 11209). Available from: https://link.springer.com/10.1007/978-3-030-01228-1_19
4.	Iscen A, Tolias G, Avrithis Y, Chum O. Label Propagation for Deep Semi-Supervised Learning. In 2019 [cited 2024 May 9]. p. 5070–9. Available from: https://openaccess.thecvf.com/content_CVPR_2019/html/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.html
5.	Gareth J, Witten D, Hastie T, Tibshirani R. An Introduction to Statistical Learning with Applications in R (2nd ed.). Second. 2023.
6.	Vedral V. The role of relative entropy in quantum information theory. Rev Mod Phys. 2002 Mar 8;74(1):197–234.
7.	Tanaka D, Ikami D, Yamasaki T, Aizawa K. Joint Optimization Framework for Learning with Noisy Labels [Internet]. arXiv; 2018 [cited 2024 May 9]. Available from: http://arxiv.org/abs/1803.11364
8.	Mumuni A, Mumuni F. Data augmentation: A comprehensive survey of modern approaches. Array. 2022 Dec 1;16:100258.
9.	Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the Inception Architecture for Computer Vision. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) [Internet]. 2016 [cited 2024 May 10]. p. 2818–26. Available from: https://ieeexplore.ieee.org/document/7780677
10.	Krizhevsky A. Learning Multiple Layers of Features from Tiny Images.
11.	Netzer Y, Wang T, Coates A, Bissacco A, Wu B, Ng AY. Reading Digits in Natural Images with Unsupervised Feature Learning.
12.	Vinyals O, Blundell C, Lillicrap T, Kavukcuoglu K, Wierstra D. Matching Networks for One Shot Learning [Internet]. arXiv; 2017 [cited 2024 May 10]. Available from: http://arxiv.org/abs/1606.04080
