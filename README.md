# 🔍Split-Detection-Network🔍

![image](https://user-images.githubusercontent.com/32592754/118756673-6300a280-b8a6-11eb-92a0-1df4ae9f776a.png)   
   
**Graduation Project**   
2020 Spring ~ 2021 Fall   
> Gachon Univ, AI·Software department   
> Author - 이수빈, 유정재, 서수영, 장휘준   
> Prof. Jung Yongju   

***
## 목차
* 소개
* 설계과정 (주제 선정 이유 포함)
* 모델 설명 + 설치 방법 + 결과
* 사용된 오픈소스 및 툴, 환경


## 소개
저희 졸업 작품 프로젝트의 주제는 '이미지 생성방식에 따라 학습 후 판별하는 네트워크'인 **Split-Detection-Network**를 활용한 deepfake detection model입니다. 현재 사회적으로 많은 분야에 deepfake를 활용한 영상과 사진들이 생겨나고 있습니다. 이는 긍정적인 방면 뿐만 아니라 불법 성인물, 가짜 뉴스, 금융 사기 등에 악용되기도 합니다. 또한, 이를 해결하기 위한 deepfake detection 기술들이 발전하고 있는데요. 저희 팀은 이러한 문제에 관심을 가지고, 더욱 효과적인 deepfake detection model을 개발하여 사회적 문제에 기여하는 것을 최종 목표로 삼았습니다.
Deepfake image를 입력하면 해당 이미지가 위조된 이미지인지 위조되지 않은 이미지인지를 알려주는 모델을 구성하였습니다. 저희 모델은 웹페이지, 어플리케이션 등을 접목할 수 있기에 편리하게 누구나 사용하실 수 있습니다. 

[이 페이지](https://github.com/neolgu/Split-Detection-Network/wiki/Specific-Description-Video)는 예시로 저희 모델을 웹페이지에 접목해본 영상과 그에 대한 설명이 담긴 곳으로, 이에 관심 있으신 분들은 참고하시기 바랍니다.


## 설계과정

초기 목표는 Kaggle에 Deepfake Detection Challange처럼 학습된 모델이 단순히 우수한 성능을 내자는 것이었습니다. 저희는 이러한 목표를 가지고 모델 설계를 위한 조사를 하다가 Deepfake를 통해 변조된 이미지들은 제작 방식에 따라 크게 2가지인 Gan, Non-Gan 방식 나눌 수 있으며 이렇게 제조된 이미지들을 각기 제작된 방식에 따라 특징이 있으며, 대부분의 모델은 이러한 점을 통해 Deepfake인지, 아닌지를 판별해내는 것을 알 수 있었습니다. 

예를 들어서 대체적으로 Non-Gan으로 제작된 이미지의 눈과 코를 통해 판별하며, Gan 이미지에서는 피부를 통해 판별하는 편입니다. 저희는 기존의 모델들은 이러한 Gan과 Non-gan으로 제작된 이미지들을 구분하지 않고 학습하는 점에 주목하여, 이미지의 생성방식에 따라 나누어서 학습한다면 Deepfake Detection에 더욱 좋은 효과를 보일 것이라는 아이디어를 내었습니다.


## 모델 설명

이를 위해 저희는 입력된 이미지가 Gan으로 제작되었는지, Non-Gan으로 제작된 이미지인지를 먼저 구분하는 판별기를 학습시킨 후, 이를 통해 판별된 이미지들을 그 생성방식에 따라 최적화된 Classifier model로 이미지를 보내서 최종적으로 판별하는 구조를 만들었습니다. 전체적인 구조는 다음과 같습니다.   
![image](https://user-images.githubusercontent.com/32592754/118757968-1074b580-b8a9-11eb-8d81-241af2d56e4d.png)   

자세한 모델 설명은 [이 곳](https://github.com/neolgu/Split-Detection-Network/wiki/Model-Description)을 참고해주시길 바랍니다.

## 테스팅 결과
![image](https://user-images.githubusercontent.com/32592754/118758037-3437fb80-b8a9-11eb-8095-383c7be8a6c2.png)   

Deepfake를 통한 변조 영상들이 인터넷 상에 유포될 때는 압축되거나 해상도가 낮아지는 부분이 있습니다. 그래서 저희도 이러한 상황을 가정하여 JPEG압축, downSampling을 테스트과정에 추가하였습니다. JPEG압축은 55%, 75%의 압축률로, downsampling 64x64 사이즈로 진행하였습니다.   
raw 데이터와 JPEG 55 압축에선 baseline model과 비슷한 결과를 보였고, JPEG 75압축에서는 1정도 낮은 결과를 보였지만, downscale에서는 눈에 띄게 좋은 결과를 얻을 수 있었습니다. 결론적으로 저희 모델은 원본 이미지 파일 형식 판별에서도 좋은 성능을 보이며, 특히 DownSampling된 파일에서 기존 모델보다 강한 성능을 보이는 모델을 개발하였습니다.

테스트를 위한 데이터셋과 베이스 라인 모델에 대한 설명은 [이 곳](https://github.com/neolgu/Split-Detection-Network/wiki/Dataset)을 참고해주시길 바랍니다.   


## 사용된 오픈소스 및 툴, 환경

__Open source & Tools used  :__   
* OpenCV   
* Dlib   
* Python   
* Pytorch   
* face forensics++ (provided on paper)   

__Computing environmnet :__      
* CPU: Intel® Xeon® CPU E3-1231   
* GPU: Nvidia GEFORCE TITAN XP   

