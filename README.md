# Deep Learning Scratch Implementations

## Why Scratch Implemantation??

Watch https://youtu.be/l9RWTMNnvi4?t=130,
The God Father of AI Geoffrey Hinton said "If you want to understand a really complicated device like brain, then you should build one".

Same goes for the Deep Learning technologies, if you really (like reeaally) want to understand such technologies then you should build them. Indeed Keras exists, but these kind of frameworks which takes not more than 5 lines to build a model actually hides everything from you, that makes you completely unknown what's going on under the hood. The purpose of such frameworks is rapid deployment and production, they are not meant to be used by people who are beginner in the field. If you already understand the technology, then be my guest and use such frameworks like a champ, otherwise if you are a beginner in the field......... you better listen to The God Father.

## List of topics implemented from scratch.

### Intro to Deep Learning.
- [Understanding Percepron - Linear Regression.](01._Intro_To_Deep_Learning/01._Linear_Regression_MXNet_(From_Scratch).ipynb)
- [Binary Classification - Logistic Regression.](01._Intro_To_Deep_Learning/02._Logistic_Regression_MXNet_(From_Scratch).ipynb)
- [Multiclass Classification - Softmax Regression.](01._Intro_To_Deep_Learning/03._Multiclass_Softmax_Classification_MXNet_(From_Scratch).ipynb)
- [The Modern Deep Learning - Neural Networks.](01._Intro_To_Deep_Learning/04._Multilayer_Neural_Networks_MXNet_(From_Scratch).ipynb)

### Gradient Descent Variants. 
- [Momentum Optimizer.](02._Gradient_Descent_Variants/01._Momentum_Optimizer_MXNet_(From_Scratch).ipynb)
- [Nesterov Accelerated Gradient Optimizer.](02._Gradient_Descent_Variants/02._Nesterov_Accelerated_Gradient_Optimizer.ipynb)
- [AdaGrad Optimizer.](02._Gradient_Descent_Variants/03._AdaGrad_Optimizer_MXNet_(From_Scratch).ipynb)
- [AdaDelta Optimizer.](02._Gradient_Descent_Variants/04._AdaDelta_Optimizer_MXNet_(From_Scratch).ipynb)
- [RMSProp Optimizer.](02._Gradient_Descent_Variants/05._RMSProp_Optimizer_MXNet_(From_scratch).ipynb)
- [Adam Optimizer.](02._Gradient_Descent_Variants/06._Adam_Optimizer_MXNet_(From_Scratch).ipynb)
- [AMSGrad Optimizer.](02._Gradient_Descent_Variants/07._AMSGrad_Optimizer_MXNet_(From_Scratch).ipynb)
- [Nadam Optimizer.](02._Gradient_Descent_Variants/08._Nadam_Optimizer_MXNet_(From_Scratch).ipynb)
- [AdaMax Optimizer.](02._Gradient_Descent_Variants/09._AdaMax_Optimizer_MXNet_(From_Scratch).ipynb)

### Preventing Overfitting.
- [L1 Regularization.](03._Preventing_Overfitting/01._Regularization_MXNet_(From_scratch).ipynb)
- [L2 Regularization.](03._Preventing_Overfitting/01._Regularization_MXNet_(From_scratch).ipynb)
- [Dropout.](03._Preventing_Overfitting/02._Dropout_MXNet_(From_Scratch).ipynb)
- [DropConnect.](03._Preventing_Overfitting/03._DropConnect_MXNet_(From_Scratch).ipynb)

### Making Deep Learning Learn Faster.
- [Batch Normalization.](04._Making_Deep_Learing_Learn_Faster/01._Batch_Normalization_MXNet_(From_Scratch).ipynb)
- [Layer Normalization.](04._Making_Deep_Learing_Learn_Faster/02._Layer_Normalization_MXNet_(From_Scratch).ipynb)

### Recurrent Neural Networks. 
- [Recurrent Neural Networks (RNN).](05._Recurrent_Neural_Networks/01._Recurrent_Neural_Networks_MXNet_(From_Scratch)_original.ipynb)
- [Long Short Term Memory Cell (LSTM).](05._Recurrent_Neural_Networks/02._Long_Short_Term_Memory_MXNet_(From_Scratch).ipynb)
- [Gated Recurrent Unit (GRU).](05._Recurrent_Neural_Networks/03._Gated_Recurrent_Unit_MXNet_(From_Scratch).ipynb)
- [Sequence to Sequence.](05._Recurrent_Neural_Networks/04._Sequence_to_Sequence_MXNet_(From_Scratch).ipynb)
- [Multilayer Recurrent Neural Networks.]()
- Bidirectional Recurrent Neural Networks. __(To be implemented)__

### Convolutional Neural Networks.
- [Convolutional Neural Networks (CNN).](06._Convolutional_Neural_Networks/01._Convolutional_Neural_Networks_MXNet_(From_Scratch).ipynb)
- Capsule Networks. __(To be implemented)__

### Generative Models.
- [Autoencoder.](07._Generative_models/01._Autoencoder_MXNet_(From_Scratch).ipynb)
- [Denoising Autoencoder.](07._Generative_models/02._Denoising_Autoencoder_MXNet_(From_Scratch).ipynb)
- [Variational Autoencoder.](07._Generative_models/03._Variational_Autoencoder.ipynb)
- [Generative Adversarial Networks.](07._Generative_models/04._Generative_Adversarial_Networks_MXNet_(From_Scratch).ipynb)
- Style Transfer. __(To be implemented)__
- Pixel2Pixel. __(To be implemented)__
- Boltzmann Machines. __(To be implemented)__
- Restricted Boltzmann Machines. __(To be implemented)__
- Deep Belief Networks. __(To be implemented)__


## How to use notebooks provided.
It is recommended to run notebooks locally on your computer only if you have GPU(CUDA) support, otherwise it'd be quit painfull :O

If you don't have the GPU, then follow the steps as bellow:-

- Go to https://colab.research.google.com
- Sign In using your google account.
- From the menu top left, Go to "__File -> Upload notebook -> Choose File__".
- Then select the notebook you want to run located in your computer and click "__Open__".
- Wait for a while.
- When notebook is successfully loaded, from the menu top left go to "__Runtime -> Change runtime type__".
- Select GPU as your "__Hardware accelerator__".
- Click "__SAVE__".
- Now you are ready to roll.


## TODO

- Implement all above algorithms from scratch in Tensorflow.
- Implement all above algorithms from scratch in Pytorch.
- Provide resources to learn topics mentioned above.
- Add more explanations to notebooks for better understanding.
