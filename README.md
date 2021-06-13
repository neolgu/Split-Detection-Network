# 🔍Split-Detection-Network📷

## 🎓 Graduation Project   
During 2020 Spring ~ 2021 Fall   
> Gachon Univ, AI·Software department   
> Author :  Lee Subin, Yu Jeongjae, Seo Sooyoung, Jang Hwijun   
> Prof. Jung Yongju   
    
## 📕 Contents
* Introdution
* Structure Design
* Model Description 
* Testing & Result
* Open Source & Environment
* Web Service Publication and Simulation


## Introdution
The theme of our graduation project is a deepfake detection model using **Split-Detection-Network**, a network that 'learns and discriminates according to image generation method'.   

Currently, videos and photographs using Deepfake are emerging in many areas of society. It has a positive effect, but it is also abused for illegal adult material, fake news, and financial fraud. Therefore, many deepfake detection technologies are developing to solve this problem. Our team became interested in these social problems and developed a more effective deepfake detection model so that anyone can use it and develop it to contribute to social problems.   

To achieve this goal, we designed a Split Detection method to not only construct a detection model that tells us if Deepfake image is a forged image or not, but also to perform better than other models. Detailed design structures and ideas will be discussed in the following table of contents.

Also, the model on this page can be applied to web pages, applications, etc., so you can use it conveniently. [This Page](https://github.com/neolgu/Split-Detection-Network/wiki/Model-used-on-Web-Page) is an example of a video that combines our model with a web page and an explanation of it, so if you're interested in it, please refer to it.




## Structure Design
Our initial goal was simply to achieve superior performance in a model learned for discrimination, such as the Deepfake Detection Challenge at Kaggle. While investigating the design of the model with this goal, we found that the images modulated through Deepfake can be divided into two main categories depending on the way they are produced.

Most Deepafake images can be divided into Gan and Non-Gan methods, and the images created through them are characterized by the way they are produced. Most models use these features to determine whether they are Deepfake or not. For example, in general, Non-Gan images are used to determine whether a character is Deepfake through the eyes and nose, and in Gan images, the people's skin is used to determine whether it is Deepfake..

Noting that existing models learn these images made of Gan and Non-gan without distinction, we came up with the idea that **Divided and learned according to the way images were created and determined** would have a better effect on Deepfake Detection.

## Model Description 

![image](https://user-images.githubusercontent.com/32592754/118757968-1074b580-b8a9-11eb-8d81-241af2d56e4d.png)   
  
Please refer to [Here](https://github.com/neolgu/Split-Detection-Network/wiki/%E2%9A%99Model-Description) for a detailed model description.

## Testing & Result
![image](https://user-images.githubusercontent.com/32592754/118758037-3437fb80-b8a9-11eb-8095-383c7be8a6c2.png)   
    
For a description of the datasets and baseline models used for testing & result, please refer to [Here](https://github.com/neolgu/Split-Detection-Network/wiki/Testing-Result-&-Dataset).
***

## Configure Environment
### Clone Repository
```
git clone https://github.com/neolgu/Split-Detection-Network
```

### Requirements
```
pip3 install -r requirements.txt  // or pip, conda, ...
```

### Dataset
* [Dataset](https://github.com/neolgu/Split-Detection-Network/wiki/Testing-Result-&-Dataset#dataset)

## Getting Started
### Edit config: config/config.yaml
edit config.yaml argument you want!

example)

To train: ```mode: train```

To test: ```mode: test```

### Run
To run application:
```
python main.py  // after edit config.yaml!!
```

## Open Source & Tools used
__Open source & Tools used  :__   
* OpenCV   
* Dlib   
* Python   
* Pytorch
* face forensics++ (provided on paper)   

__Computing environmnet :__      
* CPU: Intel® Xeon® CPU E3-1231   
* GPU: Nvidia GEFORCE TITAN XP   

## Citations
```bibtex
@misc{rössler2019faceforensics,
      title={FaceForensics++: Learning to Detect Manipulated Facial Images}, 
      author={Andreas Rössler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nießner},
      year={2019},
      eprint={1901.08971},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```bibtex
@misc{chollet2017xception,
      title={Xception: Deep Learning with Depthwise Separable Convolutions}, 
      author={François Chollet},
      year={2017},
      eprint={1610.02357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```bibtex
@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Web Service Publication and Simulation
* For publication, we used Google Cloud
* Backend was developed with Flask server and Frontend with Vue.js
* To experience publication and simulation, Click [Here](https://github.com/neolgu/Split-Detection-Network/wiki/Web-Service-Publication)
