# Welcome to Deep Learning Scratch Arena.


## What is Deep Learning Scratch Arena?

My goal is to provide high quality Scratch Implementations of the fundamentals of Deep Learning and its applications, with interactive well documentated jupyter notebooks. All notebooks comes along with implementations using Tensorflow, MXNet and Pytorch.


## Why Scratch Implemantation?

Watch https://youtu.be/l9RWTMNnvi4?t=130,
The God Father of AI Geoffrey Hinton said "If you want to understand a really complicated device like brain, then you should build one".

Same goes for the Deep Learning technologies, if you really (like reeaally) want to understand such technologies then you should build them. Indeed Keras exists, but these kind of frameworks which takes not more than 5 lines to build a model actually hides everything from you, that makes you completely unknown what's going on under the hood. The purpose of such frameworks is rapid deployment and production, they are not meant to be used by people who are beginner in the field. If you already understand the technology, then be my guest and use such frameworks like a champ, otherwise if you are a beginner in the field......... you better listen to The God Father.


## List of topics implemented from scratch.

### Intro to Deep Learning.
- [Understanding Percepron - Linear Regression.](01.%20Intro%20To%20Deep%20Learning/01.%20Linear_Regression_MXNet_(From_Scratch).ipynb)
- [Binary Classification - Logistic Regression.](01.%20Intro%20To%20Deep%20Learning/02.%20Logistic_Regression_MXNet_(From_Scratch).ipynb)
- [Multiclass Classification - Softmax Regression.](01.%20Intro%20To%20Deep%20Learning/03.%20Multiclass_Softmax_Classification_MXNet_(From_Scratch).ipynb)
- [The Modern Deep Learning - Neural Networks.](01.%20Intro%20To%20Deep%20Learning/04.%20Multilayer_Neural_Networks_MXNet_(From_Scratch).ipynb)

### Gradient Descent Variants. 
- [Momentum Optimizer.](02.%20Gradient%20Descent%20Variants/01.%20Momentum_Optimizer_MXNet_(From_Scratch).ipynb)
- [Nesterov Accelerated Gradient Optimizer.](02.%20Gradient%20Descent%20Variants/02.%20Nesterov_Accelerated_Gradient_Optimizer.ipynb)
- [AdaGrad Optimizer.](02.%20Gradient%20Descent%20Variants/03.%20AdaGrad_Optimizer_MXNet_(From_Scratch).ipynb)
- [AdaDelta Optimizer.](02.%20Gradient%20Descent%20Variants/04.%20AdaDelta_Optimizer_MXNet_(From_Scratch).ipynb)
- [RMSProp Optimizer.](02.%20Gradient%20Descent%20Variants/05.%20RMSProp_Optimizer_MXNet_(From_scratch).ipynb)
- [Adam Optimizer.](02.%20Gradient%20Descent%20Variants/06.%20Adam_Optimizer_MXNet_(From_Scratch).ipynb)
- [AMSGrad Optimizer.](02.%20Gradient%20Descent%20Variants/07.%20AMSGrad_Optimizer_MXNet_(From_Scratch).ipynb)
- [Nadam Optimizer.](02.%20Gradient%20Descent%20Variants/08.%20Nadam_Optimizer_MXNet_(From_Scratch).ipynb)
- [AdaMax Optimizer.](02.%20Gradient%20Descent%20Variants/09.%20AdaMax_Optimizer_MXNet_(From_Scratch).ipynb)

### Preventing Overfitting.
- [L1 Regularization.](03.%20Preventing%20Overfitting/01.%20Regularization_MXNet_(From_scratch).ipynb)
- [L2 Regularization.](03.%20Preventing%20Overfitting/01.%20Regularization_MXNet_(From_scratch).ipynb)
- [Dropout.](03.%20Preventing%20Overfitting/02.%20Dropout_MXNet_(From_Scratch).ipynb)
- [DropConnect.](03.%20Preventing%20Overfitting/03.%20DropConnect_MXNet_(From_Scratch).ipynb)

### Making Deep Learning Learn Faster.
- [Batch Normalization.](04.%20Making%20Deep%20Learing%20Learn%20Faster/01.%20Batch_Normalization_MXNet_(From_Scratch).ipynb)
- [Layer Normalization.](04.%20Making%20Deep%20Learing%20Learn%20Faster/02.%20Layer_Normalization_MXNet_(From_Scratch).ipynb)

### Recurrent Neural Networks. 
- [Recurrent Neural Networks (RNN).](05.%20Recurrent%20Neural%20Networks/01.%20Recurrent_Neural_Networks_MXNet_(From_Scratch)_original.ipynb)
- [Long Short Term Memory Cell (LSTM).](05.%20Recurrent%20Neural%20Networks/02.%20Long_Short_Term_Memory_MXNet_(From_Scratch).ipynb)
- [Gated Recurrent Unit (GRU).](05.%20Recurrent%20Neural%20Networks/03.%20Gated_Recurrent_Unit_MXNet_(From_Scratch).ipynb)
- [Sequence to Sequence.](05.%20Recurrent%20Neural%20Networks/04.%20Sequence_to_Sequence_MXNet_(From_Scratch).ipynb)
- [Multilayer Recurrent Neural Networks.](05.%20Recurrent%20Neural%20Networks/05.%20Multilayer_Recurrent_Neural_Networks_MXNet_(From_Scratch).ipynb)
- Bidirectional Recurrent Neural Networks. __(To be implemented)__

### Convolutional Neural Networks.
- [Convolutional Neural Networks (CNN).](06.%20Convolutional%20Neural%20Networks/01.%20Convolutional_Neural_Networks_MXNet_(From_Scratch).ipynb)
- Capsule Networks. __(To be implemented)__

### Generative Models.
- [Autoencoder.](07.%20Generative%20models/01.%20Autoencoder_MXNet_(From_Scratch).ipynb)
- [Denoising Autoencoder.](07.%20Generative%20models/02.%20Denoising_Autoencoder_MXNet_(From_Scratch).ipynb)
- [Variational Autoencoder.](07.%20Generative%20models/03.%20Variational_Autoencoder.ipynb)
- [Generative Adversarial Networks.](07.%20Generative%20models/04.%20Generative_Adversarial_Networks_MXNet_(From_Scratch).ipynb)
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

This book is an interactive deep learning book with code, math, and discussions, for beginners as well as for experts. It provides crystal clear explaination for all the topics along with hands on coding tutorials. If you got question regarding the topic then you can ask it to the forum.

11. [Another Great Book on Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)

This book introduces to Deep Learning by first teaching the crucial math required for Deep Learning(mainly for research), for example:-

- Linear Algebra
- Probability
- Numerical Computation... etc

This book would definitely help you to build the foundational understanding mathematics for research in Deep Learning.


## TODO

- Implement all above algorithms from scratch in Tensorflow.
- Implement all above algorithms from scratch in Pytorch.
- Provide resources to learn topics mentioned above.
- Add more explanations to notebooks for better understanding.
