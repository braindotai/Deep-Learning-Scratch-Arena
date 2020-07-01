# Welcome to Deep Learning Scratch Arena.


## What is Deep Learning Scratch Arena?

My goal is to provide high quality Scratch Implementations of the fundamentals of Deep Learning and its applications, with interactive well documentated jupyter notebooks. All notebooks come along with implementations using Tensorflow, MXNet and Pytorch.


## Why Scratch Implemantation?

- It helps us to understand the core working of an algorithm better.
- We could try to improve an existing algorithm both for accuracy and efficiency.
- We could implement our own version of an existing algorithm with some new features via experiments.
- We could even invent new algorithms that could outperform previous algorithms achieving SOTA(state of the art) results.
- Thus we could actually contribute to the field.

Watch https://youtu.be/l9RWTMNnvi4?t=130,
The God Father of AI Geoffrey Hinton said "If you want to understand a really complicated device like brain, then you should build one".

Same goes for the Deep Learning technologies, if you really (like reeaally) want to understand such technologies then you should build them. Indeed Keras exists, but these kind of frameworks which takes not more than 5 lines to build a model actually hides everything from you, that makes you completely unknown what's going on under the hood. The purpose of such frameworks is rapid deployment and production, they are not meant to be used by people who are beginner in the field. If you already understand the technology, then be my guest and use such frameworks like a champ, otherwise if you are a beginner in the field......... you better listen to The God Father.


## List of topics implemented from scratch.

### Intro to Deep Learning.
- [Understanding Percepron - Linear Regression](01.%20Intro%20To%20Deep%20Learning/01.%20Linear%20Regression)
- [Binary Classification - Logistic Regression](01.%20Intro%20To%20Deep%20Learning/02.%20Logistic%20Regression)
- [Multiclass Classification - Softmax Regression](01.%20Intro%20To%20Deep%20Learning/03.%20Multiclass%20Softmax%20Classification)
- [The Modern Deep Learning - Neural Networks](01.%20Intro%20To%20Deep%20Learning/04.%20Multilayer%20Neural%20Networks)

### Activation Functions.
- [Sigmoid](02.%20Activation%20Functions/01.%20Sigmoid-%20MXNet.ipynb)
- [Tanh](02.%20Activation%20Functions/02.%20Tanh-%20MXNet.ipynb)
- [ArcTan](02.%20Activation%20Functions/03.%20ArcTan)
- [Softsign](02.%20Activation%20Functions/04.%20Softsign)
- [ISRU](02.%20Activation%20Functions/05.%20ISRU)
- [ISRLU](02.%20Activation%20Functions/06.%20ISRL)
- [PLU](02.%20Activation%20Functions/07.%20PLU)
- [ReLU](02.%20Activation%20Functions/08.%20ReLu)
- [SoftPlus](02.%20Activation%20Functions/09.%20SoftPlus)
- [Leaky ReLU](02.%20Activation%20Functions/10.%20Leaky%20ReLU)
- [PReLU](02.%20Activation%20Functions/11.%20PReLU)
- [RReLU](02.%20Activation%20Functions/12.%20RReLU)
- [ELU](02.%20Activation%20Functions/13.%20ELU)
- [SELU](02.%20Activation%20Functions/14.%20SELU)
- [Swish](02.%20Activation%20Functions/15.%20Swish)
- [Bent Identity](02.%20Activation%20Functions/16.%20Bent%20Identity)

### Gradient Descent Variants. 
- [Momentum Optimizer](03.%20Gradient%20Descent%20Variants/01.%20Momentum)
- [Nesterov Accelerated Gradient Optimizer](03.%20Gradient%20Descent%20Variants/02.%20Nesterov%20Accelerated%20Gradient)
- [AdaGrad Optimizer](03.%20Gradient%20Descent%20Variants/03.%20AdaGrad)
- [AdaDelta Optimizer](03.%20Gradient%20Descent%20Variants/04.%20AdaDelta)
- [RMSProp Optimizer](03.%20Gradient%20Descent%20Variants/05.%20RMSProp)
- [Adam Optimizer](03.%20Gradient%20Descent%20Variants/06.%20Adam)
- [AMSGrad Optimizer](03.%20Gradient%20Descent%20Variants/07.%20AMSGrad)
- [Nadam Optimizer](03.%20Gradient%20Descent%20Variants/08.%20Nadam)
- [AdaMax Optimizer](03.%20Gradient%20Descent%20Variants/09.%20AdaMax)

### Learning Rate Decay Methods.
- [Time Decay](04.%20Learning%20Rate%20Decay%20Methods/01.%20Time%20Decay)
- [Step Decay](04.%20Learning%20Rate%20Decay%20Methods/02.%20Step%20Decay)
- [Exponential Decay](04.%20Learning%20Rate%20Decay%20Methods/03.%20Exponential%20Decay)
- [Cyclic Decay](04.%20Learning%20Rate%20Decay%20Methods/04.%20Cyclic%20Decay)

### Preventing Overfitting.
- [L1 Regularization](05.%20Preventing%20Overfitting/01.%20Regularization)
- [L2 Regularization](05.%20Preventing%20Overfitting/01.%20Regularization)
- [Dropout](05.%20Preventing%20Overfitting/02.%20Dropout)
- [DropConnect](05.%20Preventing%20Overfitting/03.%20DropConnect)

### Making Deep Learning Learn Faster.
- [Batch Normalization](06.%20Making%20Deep%20Learing%20Learn%20Faster/01.%20Batch%20Normalization)
- [Layer Normalization](06.%20Making%20Deep%20Learing%20Learn%20Faster/02.%20Layer%20Normalization)

### Recurrent Neural Networks. 
- [Recurrent Neural Networks (RNN)](07.%20Recurrent%20Neural%20Networks/01.%20Recurrent%20Neural%20Networks)
- [Long Short Term Memory Cell (LSTM)](07.%20Recurrent%20Neural%20Networks/02.%20Long%20Short%20Term%20Memory)
- [Gated Recurrent Unit (GRU)](07.%20Recurrent%20Neural%20Networks/03.%20Gated%20Recurrent%20Unit)
- [Sequence to Sequence](07.%20Recurrent%20Neural%20Networks/04.%20Sequence%20to%20Sequence)
- [Multilayer Recurrent Neural Networks](07.%20Recurrent%20Neural%20Networks/05.%20Multilayer%20Recurrent%20Neural%20Networks%20MXNet.ipynb)
- [Bidirectional Recurrent Neural Networks](07.%20Recurrent%20Neural%20Networks/06.%20Bidirectional%20Recurrent%20Neural%20Networks)

### Convolutional Neural Networks.
- [Understanding Convolutional](08.%20Convolutional%20Neural%20Networks/01.%20Understanding%20Convolutional)
- [Convolution with Padding](08.%20Convolutional%20Neural%20Networks/03.%20Convolution%20with%20Padding)
- [Convolution with Strides](08.%20Convolutional%20Neural%20Networks/02.%20Convolution%20with%20Strides)
- [Max Pooling](08.%20Convolutional%20Neural%20Networks/04.%20Max%20Pooling)
- [Average Pooling](08.%20Convolutional%20Neural%20Networks/05.%20Average%20Pooling)
- [Understanding Deconvolution](08.%20Convolutional%20Neural%20Networks/06.%20Understanding%20Deconvolution)
- [Deconvolution with Strides](08.%20Convolutional%20Neural%20Networks/07.%20Deconvolution%20with%20Strides)
- [Deconvolution with Padding](08.%20Convolutional%20Neural%20Networks/08.%20Deconvolution%20with%20Padding)

### Generative Models.
- [Autoencoder](09.%20Generative%20Models/01.%20Autoencoder%20MXNet.ipynb)
- [Denoising Autoencoder](09.%20Generative%20Models/02.%20Denoising%20Autoencoder)
- [Sparse Autoencoder](09.%20Generative%20Models/03.%20Sparse%20Autoencoder)
- [Variational Autoencoder](09.%20Generative%20Models/04.%20Variational%20Autoencoder)
- [Generative Adversarial Networks](09.%20Generative%20Models/05.%20Generative%20Adversarial%20Networks)
- [Deep Convolutional GAN](09.%20Generative%20Models/06.%20Deep%20Convolutional%20GAN)
- [Conditional GAN](09.%20Generative%20Models/07.%20Conditional%20GAN)
- [AC GAN](09.%20Generative%20Models/08.%20AC%20GAN)
- [LS GAN](09.%20Generative%20Models/09.%20LS%20GAN)
- [Info GAN](09.%20Generative%20Models/10.%20Info%20GAN)
- Cyclic GAN. __(To be implemented)__
- Disco GAN. __(To be implemented)__
- Bi GAN. __(To be implemented)__
- W GAN. __(To be implemented)__

### Computer Vision
- [LeNet](10.%20Computer%20Vision/01.%20LeNet)
- [AlexNet](10.%20Computer%20Vision/02.%20AlexNet)
- [VGG](10.%20Computer%20Vision/03.%20VGG%20Net)
- [Network in Network](10.%20Computer%20Vision/04.%20Network%20in%20Network)
- [Inception Network](10.%20Computer%20Vision/05.%20Inception%20Network)
- [ResNet](10.%20Computer%20Vision/06.%20Residual%20Network)
- [DenseNet](10.%20Computer%20Vision/07.%20DenseNet)
- [Data Augmentation](10.%20Computer%20Vision/08.%20Data%20Augmentation)

### Applications of Computer Vision
- [Image Classification](11.%20Applications%20of%20Computer%20Vision/01.%20Image%20Classification)
- [Transfer Learning](11.%20Applications%20of%20Computer%20Vision/02.%20Transfer%20Learning)
- [Image Denoising](11.%20Applications%20of%20Computer%20Vision/03.%20Image%20Denoising)
- Image Resolution Inhancer. __(To be implemented)__
- Image Reconstruction. __(To be implemented)__
- Image Colorization. __(To be implemented)__
- Object Detection. __(To be implemented)__
- Objection Segmentation. __(To be implemented)__
- Style Transfer. __(To be implemented)__
- Text to Image. __(To be implemented)__
- Image Captioning. __(To be implemented)__

### Natural Language Processing
- Word Embeddings. __(To be implemented)__
- Embedding Matrix. __(To be implemented)__
- Word to Vec. __(To be implemented)__
- GloVe. __(To be implemented)__
- Negative Sampling. __(To be implemented)__
- Attention Mechanism. __(To be implemented)__

### Applications of Natural Language Processing
- Sentiment Analysis. __(To be implemented)__
- Named Entity Recognition. __(To be implemented)__
- Machine Translation. __(To be implemented)__
- Text Summarization. __(To be implemented)__
- Visual Question Answering. __(To be implemented)__
- Image Captioning. __(To be implemented)__
- Music Generation. __(To be implemented)__


## How to run provided notebooks?
It is recommended to run notebooks locally on your computer only if you have GPU(CUDA) support, otherwise it'd be quit painfull :O

If you don't have the GPU, then follow the steps as bellow:-

- Go to https://colab.research.google.com
- Sign In using your google account.
- Clone the repository by running `git clone`
- From the menu top left, Go to "__File -> Upload notebook -> Choose File__".
- Then select the notebook you want to run located in your computer and click "__Open__".
- Wait for a while.
- When notebook is successfully loaded, from the menu top left go to "__Runtime -> Change runtime type__".
- Select GPU as your "__Hardware accelerator__".
- Click "__SAVE__".
- Now you are ready to roll.


## What are some good resources to learn Deep Learning?

Here are some very useful resources I've curated so far. First we will learn some basic fundamentals of Deep Learning, then we will learn the core math required to understand a bit complex Deep Learning approaches, so watch and read in the following order. 


1. [Deep Learning For Beginners by Andrew Ng](https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)

	Its a full course for beginners by Prof Andrew Ng, this course provides the glimps to the world of AI.

2. [Deep Learning For Intermediates by Andrew Ng](https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)

	Its a full course for intermediates by Prof Andrew Ng, this course provides techniques to improve Deep Learning models.

3. [Convolutional Neural Networks by Andrew Ng](https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

	Its a full course on Convolutional Neural Networks by Prof Andrew Ng, this course provides great explanations and examples for the topic with great simplicity and helpful guides.

4. [Recurrent Neural Networks by Andrew Ng](https://www.youtube.com/watch?v=efWlOCE_6HY&list=PL1w8k37X_6L_s4ncq-swTBvKDWnRSrinI&index=2)

	Its a full course on Recurrent Neural Networks by Prof Andrew Ng, this course provides explanations to why we need sequence models and what are their use cases and how do they work.

5. [An Outstanding course for Linear Algebra by Prof W. Gilbert Strange](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PLE7DDD91010BC51F8)

	This course is by far my best course on Linear Algebra on the planet. Prof Strange has done a phenomenal job, just watch his videos and I bet you'd see a rainbow with a unicorn.

6. [An Another Outstanding Course on Linear Algera By 3Blue1Brown](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PL0-GT3co4r2y2YErbmuJw2L5tW4Ew2O5B)

	This course introduces the "Essence of linear algebra" series, aimed at animating the geometric intuitions underlying many of the topics taught in a standard linear algebra course.

7. [Course on Calculus by 3Blue1Brown](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PL0-GT3co4r2wlh6UHTUeQsrf3mlS2lk6x)

	What might it feel like to invent calculus?


8. [A Great Course on Deep Learning by MIT](https://www.youtube.com/watch?v=5v1JnYv_yWs&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1)

	Its a full course on Deep Learning by MIT, this course provides great explanations to various deep learning topics like GANs... etc. Note: only watch first 4 lectures because afterwards they are explaining an another AI field know as Reinforcement Learning, which we will cover later.

9. [Deep Learning Course by Stanford](https://www.youtube.com/watch?v=PySo_6S4ZAg&list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)

	This course covers introduction to deep learning and its applications like Adversarial Attacks, Interpretability and Health Care... etc. Note: skip the Reinforcement Learning part for now, we will cover that later.

10. [An Awesome Book on Deep Learning](https://www.d2l.ai/)

	This book is an interactive deep learning book with codes, maths, and discussions, for beginners as well as for experts. It provides crystal clear explaination for all the topics along with hands on coding tutorials. If you got question regarding the topic then you can ask it to the forum as well.

11. [Another Great Book on Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)

	This book introduces to Deep Learning by first teaching the crucial math required for research in Deep Learning, for example:-

	- Linear Algebra
	- Probability
	- Numerical Computation... etc

	This book would definitely help you to build the foundational understanding of mathematics for research in Deep Learning.


## TODO

- Implement all above algorithms from scratch in Tensorflow.
- Implement all above algorithms from scratch in Pytorch.
- Provide resources to learn topics mentioned above.
- Add more explanations to notebooks for better understanding.

# Author

## __Rishik Mourya__
