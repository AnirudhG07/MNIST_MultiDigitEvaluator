# MNIST MULTI DIGIT EVALUATION

MNIST is a famous dataset which is used by everyone to train models to identify handwritten images. 
However, the images contains only 1 digit rather than many.
<br> Hence, this repository contains code on how to make your own dataset and how to evaluate your model

# Image Classifications 
Two datasets of 10000 images has been uploaded, one containing images with 2 digit numbers and other with 3 digit numbers. You can unzip and see the jpeg images.
<br> You can also create your own dataset of n digit numbers using data_creator_n.py which will load the MNIST dataset and randomly select n of 60000 training images
(or testing, depending on what you would like) and concatenate them to form a 28 x 28*n size image.
<br> <h4> For Example: </h4>
See the below image of number 726766291, a 9 nine digit number. <br>
<img width="613" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/19a25a5d-5ca8-4178-8cfb-99bdf7f0cb6c">

For a human, it is trivial to identify it, but for a machine we need to segregate the image into pieces so that it can evaluate using out model trained.
<br> **IMPORTANT NOTE:** 
While saving the image, matplotlib may change the dimension of image from 28x252 to 55x496 as well, which is dangerous as model is trained on 28*28=784 input size.
Hence PIL is used to save the image.

# Model Evaluation
Model using PyTorch is used here, which will detect the number in each 28x28 image and finally output desired result.<br>
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/ed279fc6-0f49-4f34-a058-1044b0b3872b"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/d96cae34-2b39-4980-956f-1e35ffe577b4">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/0f7151be-b7ce-474c-af13-264e1c9d7cad"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/fc2eb384-02e4-4795-9fd7-e456176da046">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/46427fa1-06c5-427e-82b2-8e3ccd8bfdd8"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/7a3629d0-5850-461d-ac27-b4d2d3ff8ded">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/8f249b7f-3512-4824-9bee-7c4351f553b0"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/a086e297-32d3-4773-8843-12401996ee69">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/20cbaa2e-2148-41c9-9908-b6d27601ae7a"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/7e8239f5-a586-45df-bfab-51806aae80e9">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/dea47c78-0803-4175-8f2c-a37b3f8ef6f8"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/65b87df5-8c9d-4bd2-bd4e-acdefbf1c67e">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/36a1aaee-1512-4f22-a625-5792019966bc"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/aef4a0ae-795b-4618-b496-972a0931d687">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/adf2ea5c-1b74-4c78-bcc5-88c0ebf718e3"> <img width="30" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/ab1453e6-9635-4169-829c-0a5f6c9f709f">
<img width="68" alt="image" src="https://github.com/AnirudhG07/MNIST_MultiDigitEvaluator/assets/146579014/20a399a7-8b0c-4b01-b873-ea8404fa167d"> <h2>**= 726766291**</h2>


