# Plant Coverage Rate Estimation Using Classification

### Introduction:

This project was part of my internship at Shanghai Roots & Shoots: Million Tree Project, a non-profit oganization that plants trees in inner mongolia in China. The goal is to use drone images of tree plantations to estimate plant coverage rates to help better monitor tree growth. I used an SVM model to classify the pixels based on RGB values into two categories: "tree" and "soil", and used the percentage of "tree" pixels as the plant coverage rate for each image.

original image:\
<img src="https://user-images.githubusercontent.com/80374850/200893490-f7614751-1e76-47ed-9508-0130eafed159.JPG" width="400">


image after classification:\
![clf_DJI_0206](https://user-images.githubusercontent.com/80374850/200893564-9533c752-ef20-45c8-9322-67b100ef73af.JPG)

### Methodology:

#### Preprocessing: 
Because the images were taken under different lighting conditions, the color tone differs from image to image. This makes it difficult to generalize model results to unseen images. To solve this problem, I centered the RGB values of each image, so that all the images would have the same mean value for each of the RGB chanels.

unprocssed image:\
<img src="https://user-images.githubusercontent.com/80374850/200897450-d8afcc6a-a182-4b1a-9f97-60e427fe880c.JPG" width="400">

processed image:\
<img src="https://user-images.githubusercontent.com/80374850/200897502-321ce376-1977-4e45-a683-39dd822db182.JPG" width="400">
