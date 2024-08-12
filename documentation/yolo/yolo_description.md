# Model YOLOv8 classifier

## Introduction to YOLOv8

### Model arhitecture

The YOLOv8 model is popular object detection
algorithm, whose architecture is divided in three parts. First
part is called backbone, which is responsible for extracting
image features. Extracted features are passed to the next part
called neck. This component is used to fuse features from
different scales, which are then forwarded to prediction.
Prediction is carried out in part called head, which is used to
predict classes.

For backbone, YOLO uses Cross Stage Partial (CSP)
networks with convolutional layers. CSP networks addresses
multiple gradient problems, like vanishing gradient and
duplicate gradient, used in back-propagation algorithm for
training. CSP networks also encourage network to reuse
features and reduce number of parameters.
YOLO uses BiFNP neck architecture for feature fusion.
This network aims to aggregate features at different
resolutions. Design of this network is published in
„EfficientDet: Scalable and Efficient Object Detection” [5].
It improves previous methods by adding weighted feature
fusion, since different input features are at different
resolutions, and they usually contribute to the output feature
unequally.

Last component is called detection head, which is used
for predicting object classes and confidence
scores. It is composed from convolutional layers, that are
encoding features relevant to object detection, and spatial
reduction layers, that reduce dimensions and increase each
node perceptual field.

### Loss functions

In learning process YOLO uses multiple loss function
for classification like BCELoss and
FocalLoss. BCELoss or Binary Cross Entropy Loss is loss function used to
measure the difference between predicted binary outcomes and actual binary
labels. Following formula describes BCELoss function.
$$-\frac{1}{N}\sum_i^N \sum_j^M y_{ij} log(p_{ij})$$
Where N is number of examples, M number of classes,
y one hot encoding of ground truth and
p predicted probability of class j for example i.
Focal loss is loss function used to address the issue of imbalanced number of classes.
It assignes more weight to smaller classes and harder examples.
With adding this weight model avoids being biased toward larger classes.
Focall loss can be expressed with following formula:
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma log(p_t)$$
Where $p_t$ is predicted probability for given class,
$\alpha_t$ is the weight assigned to class, and $\gamma$ is the focusing parameter
that reduces relative well-classified examples, and puts more focus on hard examples.

## Model training data augmentation

Data augmentation is used to reduce over-fitting when
having low availability of original data. This process
involves altering the training images to generate synthetic
dataset. On our dataset, we used augmentation with scaling,
HSV, translation, flip and mosaic.
Scaling will improve recognising close ups and wide shots  of images
containing a dog. Changing image
Hue Saturation and Value (HSV) is commonly used to
generate images with different color, brightness and
saturation, which helps model to become more robust to
changes in lighting condition, shadows and color variants in
input images. Flipping and translation stops model from
learning point where most objects from interest are located in
images, this also acts as method of regularization. Mosaic
makes model better at detecting objects that are located
behind other objects, and encourages model to learn more
distinguishing features about each class.

## Training process

For model training, we used stochastic gradient descent
SGD. This is type of online learning algorithm, which makes
weight correction after each batch (in our case batch
contained 16 images). Since function that is being optimized
has high modularity, this means that function contains many
local optima. If gradient would be applied after all samples it
would lead to closest local optima, so by using
approximation after each batch algorithm is avoiding local
optima while reducing error imposed by given loss function.
When starting, YOLO uses warmup parameters. These
parameters are used for given time before optimization
process starts using given hyperparameters. We used 3
epochs for warmup, and learning rate of 0.01, and momentum
of 0.9 for training with SGD.

Learning rate is hyperparameter that controls rate at which algoritm is learning (adjusting weights with gradients). Learning rate can be described with formula:
$$w \leftarrow w + \eta \nabla_w \mathcal{L}$$
Where $\eta$ is learning rate, and $\nabla_w \mathcal{L}$
gradient of loss function by w. If we set learning rate
to high model will diverge and wouldn't learn anything,
but with smaller learning rate model is learning slower.

Momentum is used to define exponential weighted
moving average of gradients. Next gradient would be
calculated by multiplying gradient for current step with
parameter ($\beta$ – 1) with addition of previous gradient sum
multiplied by $\beta$. Representing weighted average of gradients
with V, method could be expressed with following formula.
$V_t = (1 - \beta)\nabla_w L(W, X,y) + \beta V_{t-1}$ . This method
improves SGD convergence, by using larger gradients in
direction that most recent gradients point, and lower for ones
deviating from average.

## Results

### Metrics used to validate model

For describing model performace following metrics will be used: precision, recall
and f1. To calculate those metrics reader first needs to introduce itself with
true positives (TP), false positives (FP) and false negatives (FN).

First lets consider this mesures on binary exampe with classes 1 and 0.
True positives are examples in test set that model marked as
positive (1) and they are in fact positive (1). False positives are examples that
were marked as positive (1), but are in fact negative (0). While flase negatives
are examples that are marked as negative (0), but are in fact positive (1).
For determening this metrics in problem with multiple classes
we focse on single class. For example lets take Afghan,
then TP are all examples that are in
fact Afghan, and are marked as Afghan, FP are all examples that are marked as
Afghan while in fact they belong to other dog breed and FN are all examples
that are in fact Afghan, but are marked as other dog breed.

Precision is defined as TP / (TP + FP), which is percentage of correctly marked
examples from all examples that model classified in considered class.
Recall is defined as TP / (TP + FN), which is percentage of correctly marked
examples from all examples that truly belong considered class. F1 is harmonic
mean of precision and recall, giving single number to mesure performance of
model.

When training of modle was finised, model was tested on test dataset which
was not previously seen by model. For each class,  previously stated mesures,
were calculated. Following table displays mesures for each class.

|Class               |   p    |   r    |   f1  |
| ------------------ | ------ | ------ | ----- |
|Afghan              | 1.0000 | 1.0000 | 1.0000|
|African Wild Dog    | 1.0000 | 1.0000 | 1.0000|
|Airedale            | 1.0000 | 1.0000 | 1.0000|
|American Hairless   | 0.9091 | 1.0000 | 0.9524|
|American Spaniel    | 0.7500 | 0.6000 | 0.6667|
|Basenji             | 1.0000 | 1.0000 | 1.0000|
|Basset              | 1.0000 | 0.9000 | 0.9474|
|Beagle              | 1.0000 | 0.9000 | 0.9474|
|Bearded Collie      | 1.0000 | 1.0000 | 1.0000|
|Bermaise            | 1.0000 | 1.0000 | 1.0000|
|Bichon Frise        | 1.0000 | 1.0000 | 1.0000|
|Blenheim            | 1.0000 | 1.0000 | 1.0000|
|Bloodhound          | 1.0000 | 1.0000 | 1.0000|
|Bluetick            | 0.9091 | 1.0000 | 0.9524|
|Border Collie       | 1.0000 | 1.0000 | 1.0000|
|Borzoi              | 1.0000 | 1.0000 | 1.0000|
|Boston Terrier      | 0.7692 | 1.0000 | 0.8696|
|Boxer               | 1.0000 | 1.0000 | 1.0000|
|Bull Mastiff        | 0.9091 | 1.0000 | 0.9524|
|Bull Terrier        | 1.0000 | 1.0000 | 1.0000|
|Bulldog             | 1.0000 | 0.7000 | 0.8235|
|Cairn               | 0.7692 | 1.0000 | 0.8696|
|Chihuahua           | 1.0000 | 0.8000 | 0.8889|
|Chinese Crested     | 1.0000 | 0.9000 | 0.9474|
|Chow                | 0.9091 | 1.0000 | 0.9524|
|Clumber             | 1.0000 | 1.0000 | 1.0000|
|Cockapoo            | 0.8182 | 0.9000 | 0.8571|
|Cocker              | 0.9091 | 1.0000 | 0.9524|
|Collie              | 1.0000 | 1.0000 | 1.0000|
|Corgi               | 1.0000 | 1.0000 | 1.0000|
|Coyote              | 1.0000 | 1.0000 | 1.0000|
|Dalmation           | 1.0000 | 1.0000 | 1.0000|
|Dhole               | 1.0000 | 1.0000 | 1.0000|
|Dingo               | 0.8333 | 1.0000 | 0.9091|
|Doberman            | 1.0000 | 1.0000 | 1.0000|
|Elk Hound           | 1.0000 | 0.9000 | 0.9474|
|French Bulldog      | 0.9000 | 0.9000 | 0.9000|
|German Sheperd      | 0.9091 | 1.0000 | 0.9524|
|Golden Retriever    | 1.0000 | 1.0000 | 1.0000|
|Great Dane          | 1.0000 | 1.0000 | 1.0000|
|Great Perenees      | 1.0000 | 1.0000 | 1.0000|
|Greyhound           | 0.9091 | 1.0000 | 0.9524|
|Groenendael         | 1.0000 | 1.0000 | 1.0000|
|Irish Spaniel       | 0.7500 | 0.9000 | 0.8182|
|Irish Wolfhound     | 1.0000 | 1.0000 | 1.0000|
|Japanese Spaniel    | 1.0000 | 0.9000 | 0.9474|
|Komondor            | 1.0000 | 1.0000 | 1.0000|
|Labradoodle         | 0.8889 | 0.8000 | 0.8421|
|Labrador            | 1.0000 | 0.9000 | 0.9474|
|Lhasa               | 0.8333 | 1.0000 | 0.9091|
|Malinois            | 1.0000 | 0.7000 | 0.8235|
|Maltese             | 1.0000 | 1.0000 | 1.0000|
|Mex Hairless        | 1.0000 | 1.0000 | 1.0000|
|Newfoundland        | 1.0000 | 0.9000 | 0.9474|
|Pekinese            | 0.9000 | 0.9000 | 0.9000|
|Pit Bull            | 1.0000 | 1.0000 | 1.0000|
|Pomeranian          | 1.0000 | 1.0000 | 1.0000|
|Poodle              | 1.0000 | 1.0000 | 1.0000|
|Pug                 | 1.0000 | 0.9000 | 0.9474|
|Rhodesian           | 0.9000 | 0.9000 | 0.9000|
|Rottweiler          | 1.0000 | 0.9000 | 0.9474|
|Saint Bernard       | 0.9091 | 1.0000 | 0.9524|
|Schnauzer           | 1.0000 | 1.0000 | 1.0000|
|Scotch Terrier      | 0.8333 | 1.0000 | 0.9091|
|Shar_Pei            | 1.0000 | 1.0000 | 1.0000|
|Shiba Inu           | 1.0000 | 0.9000 | 0.9474|
|Shih-Tzu            | 1.0000 | 0.9000 | 0.9474|
|Siberian Husky      | 1.0000 | 1.0000 | 1.0000|
|Vizsla              | 0.9091 | 1.0000 | 0.9524|
|Yorkie              | 1.0000 | 0.8000 | 0.8889|

For classification only class with highest probability was
considered, which is often called top 1.
As we can se model achieved precision and recall of 100% on 37 of 70 classes.
Average precision was 95.896%, recall was 95.429% and f1 was 95.383%,
this is also called makro mesure. Total accuracy on test set was 95.429%.
