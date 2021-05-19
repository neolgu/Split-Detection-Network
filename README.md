# 🔍Split-Detection-Network📷

## 🎓 Graduation Project   
During 2020 Spring ~ 2021 Fall   
> Gachon Univ, AI·Software department   
> Author :  이수빈, 유정재, 서수영, 장휘준   
> Prof. Jung Yongju   
    
## 📕 Contents
* Introdution
* Structure Design
* Development Environment
* Description Model
* Testing & Result
* Open Source & Tools used


## Introdution
저희 졸업 작품 프로젝트의 주제는 '이미지 생성방식에 따라 학습 후 판별하는 네트워크'인 **Split-Detection-Network**를 활용한 deepfake detection model입니다.   

현재 사회적으로 많은 분야에서 Deepfake를 활용한 영상과 사진들이 생겨나고 있습니다. 이는 긍정적인 영향도 있으나 불법 성인물, 가짜 뉴스, 금융 사기 등에 악용되기도 합니다. 따라서 이를 해결하기 위한 수많은 deepfake detection 기술들이 발전하고 있는데요. 저희 팀은 이러한 사회적 문제에 관심을 가지게되어 더욱 효과적인 deepfake detection model을 개발하여 누구나 사용하고 발전시킬 수 있도록하여 사회적 문제에 기여하는 것을 최종 목표로 삼았습니다.

이러한 목표를 달성하기 위해 저희는 Deepfake image를 입력하면 해당 이미지가 위조된 이미지인지, 위조되지 않은 이미지인지를 알려주는 Detection Model을 구성하는 것에 그치지 않고, 타 모델에 비하여 더욱 좋은 성능을 내기 위해 Split Detection 방식을 고안하였습니다. 자세한 설계 구조 및 아이디어는 다음 목차에서 설명하겠습니다. 

또한, 이 깃헙 페이지에 올라온 model은 웹페이지, 어플리케이션 등을 접목할 수 있기에 편리하게 누구나 사용하실 수 있습니다. [이 페이지](https://github.com/neolgu/Split-Detection-Network/wiki/Model-used-on-Web-Page)는 예시로 저희 모델을 웹페이지에 접목해본 영상과 그에 대한 설명이 담긴 곳으로, 이에 관심 있으신 분들은 참고하시기 바랍니다.


## Structure Design

저희 팀의 초기 목표는 Kaggle에 Deepfake Detection Challange처럼 학습된 모델이 단순히 우수한 성능을 내자는 것이었습니다. 저희는 이러한 목표를 가지고 모델 설계를 위한 조사를 하다가 Deepfake를 통해 변조된 이미지들은 제작 방식에 따라 크게 2가지로 나눌 수 있다는 점을 알 수 있었습니다.

대다수의 Deepafake를 통해 변조된 이미지들은 Gan, Non-Gan 방식으로 나눌 수 있으며 이를 통해 만들어진 이미지들을 각기 제작된 방식에 따라 특징이 있으며, 대부분의 모델은 이러한 점을 통해 Deepfake인지, 아닌지를 판별해내는 것을 알 수 있었습니다. 예를 들어서 대체적으로 Non-Gan으로 제작된 이미지 속 인물의 눈과 코를 통해 판별하며, Gan 이미지에서는 인물의 피부를 통해 판별하는 편입니다.   

저희는 기존의 모델들은 이러한 Gan과 Non-gan으로 제작된 이미지들을 구분하지 않고 학습하는 점에 주목하여, **이미지의 생성방식에 따라 나누어서 학습**한다면 Deepfake Detection에 더욱 좋은 효과를 보일 것이라는 아이디어를 내었고, 이를 발전시키어 저희 모델에 접목시키고자 하였습니다.


## Development Environment

## Description Model

![image](https://user-images.githubusercontent.com/32592754/118757968-1074b580-b8a9-11eb-8d81-241af2d56e4d.png)   

자세한 모델 설명은 [이 곳](https://github.com/neolgu/Split-Detection-Network/wiki/Model-Description)을 참고해주시길 바랍니다.

## Testing & Result
![image](https://user-images.githubusercontent.com/32592754/118758037-3437fb80-b8a9-11eb-8095-383c7be8a6c2.png)   

테스트를 위한 사용된 데이터셋 및 베이스 라인 모델에 대한 설명은 [이 곳](https://github.com/neolgu/Split-Detection-Network/wiki/Testing-Result-&-Dataset)을 참고해주시길 바랍니다.   


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

