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


## Introdution
저희 졸업 작품 프로젝트의 주제는 '이미지 생성방식에 따라 학습 후 판별하는 네트워크'인 **Split-Detection-Network**를 활용한 deepfake detection model입니다.   

현재 사회적으로 많은 분야에서 Deepfake를 활용한 영상과 사진들이 생겨나고 있습니다. 이는 긍정적인 영향도 있으나 불법 성인물, 가짜 뉴스, 금융 사기 등에 악용되기도 합니다. 따라서 이를 해결하기 위한 수많은 deepfake detection 기술들이 발전하고 있는데요. 저희 팀은 이러한 사회적 문제에 관심을 가지게되어 더욱 효과적인 deepfake detection model을 개발하여 누구나 사용하고 발전시킬 수 있도록하여 사회적 문제에 기여하는 것을 최종 목표로 삼았습니다.

이러한 목표를 달성하기 위해 저희는 Deepfake image를 입력하면 해당 이미지가 위조된 이미지인지, 위조되지 않은 이미지인지를 알려주는 Detection Model을 구성하는 것에 그치지 않고, 타 모델에 비하여 더욱 좋은 성능을 내기 위해 Split Detection 방식을 고안하였습니다. 자세한 설계 구조 및 아이디어는 다음 목차에서 설명하겠습니다. 

또한, 이 페이지에 올라온 model은 웹페이지, 어플리케이션 등에 접목할 수 있기에 편리하게 누구나 사용하실 수 있습니다. [이 페이지](https://github.com/neolgu/Split-Detection-Network/wiki/Model-used-on-Web-Page)는 예시로 저희 모델을 웹페이지에 접목해본 영상과 그에 대한 설명이 담긴 곳으로, 이에 관심 있으신 분들은 참고하시기 바랍니다.

The theme of our graduation project is a deepfake detection model using **Split-Detection-Network**, a network that 'learns and discriminates according to image generation method'.   

Currently, videos and photographs using Deepfake are emerging in many areas of society. It has a positive effect, but it is also abused for illegal adult material, fake news, and financial fraud. Therefore, many deepfake detection technologies are developing to solve this problem. Our team became interested in these social problems and developed a more effective deepfake detection model so that anyone can use it and develop it to contribute to social problems.   

To achieve this goal, we designed a Split Detection method to not only construct a detection model that tells us if Deepfake image is a forged image or not, but also to perform better than other models. Detailed design structures and ideas will be discussed in the following table of contents.

Also, the model on this page can be applied to web pages, applications, etc., so you can use it conveniently. [This Page](https://github.com/neolgu/Split-Detection-Network/wiki/Model-used-on-Web-Page) is an example of a video that combines our model with a web page and an explanation of it, so if you're interested in it, please refer to it.




## Structure Design

저희 팀의 초기 목표는 Kaggle에서 진행된 Deepfake Detection Challange처럼 학습된 모델이 단순히 우수한 성능을 내자는 것이었습니다. 저희는 이러한 목표를 가지고 모델 설계를 위한 조사를 하다가 Deepfake를 통해 변조된 이미지들은 제작 방식에 따라 크게 2가지로 나눌 수 있다는 점을 확인했습니다.

대다수의 Deepafake를 통해 제작된 이미지들은 Gan과 Non-Gan 방식으로 나눌 수 있으며 이를 통해 만들어진 이미지들을 각기 제작된 방식에 따라 특징이 있습니다. 대부분의 모델은 이러한 특징을 통해 Deepfake인지, 아닌지를 판별해냅니다. 예를 들어, 대체적으로 Non-Gan 이미지는 인물의 눈과 코를 통해 Deepfake 여부를 판별하며, Gan 이미지에서는 인물의 피부를 통해 판별하는 편입니다.   

저희는 기존의 모델들이 이러한 Gan과 Non-gan으로 제작된 이미지들을 구분하지 않고 학습하는 점에 주목하여, **이미지의 생성방식에 따라 나누어서 학습시킨 후 판별**한다면 Deepfake Detection에 더욱 좋은 효과를 보일 것이라는 아이디어를 내었습니다.


Our initial goal was simply to achieve superior performance in a model learned for discrimination, such as the Deepfake Detection Challenge at Kaggle. While investigating the design of the model with this goal, we found that the images modulated through Deepfake can be divided into two main categories depending on the way they are produced.

Most Deepafake images can be divided into Gan and Non-Gan methods, and the images created through them are characterized by the way they are produced. Most models use these features to determine whether they are Deepfake or not. For example, in general, Non-Gan images are used to determine whether a character is Deepfake through the eyes and nose, and in Gan images, the people's skin is used to determine whether it is Deepfake..

Noting that existing models learn these images made of Gan and Non-gan without distinction, we came up with the idea that **Divided and learned according to the way images were created and determined** would have a better effect on Deepfake Detection.

## Model Description 

![image](https://user-images.githubusercontent.com/32592754/118757968-1074b580-b8a9-11eb-8d81-241af2d56e4d.png)   

자세한 모델 설명은 [이 곳](https://github.com/neolgu/Split-Detection-Network/wiki/%E2%9A%99Model-Description)을 참고해주시길 바랍니다.   
Please refer to [Here](https://github.com/neolgu/Split-Detection-Network/wiki/%E2%9A%99Model-Description) for a detailed model description.

## Testing & Result
![image](https://user-images.githubusercontent.com/32592754/118758037-3437fb80-b8a9-11eb-8095-383c7be8a6c2.png)   

테스트를 위한 사용된 데이터셋 및 베이스 라인 모델과 결과에 대한 설명은 [이 곳](https://github.com/neolgu/Split-Detection-Network/wiki/Testing-Result-&-Dataset)을 참고해주시길 바랍니다.      
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


