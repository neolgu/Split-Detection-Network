# Split-Detection-Network

![image](https://user-images.githubusercontent.com/32592754/118756673-6300a280-b8a6-11eb-92a0-1df4ae9f776a.png)   
__Graduation Project__

## 기여자

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
저희 졸업 작품 프로젝트의 주제는 '이미지 생성방식에 따라 학습 후 판별하는 네트워크'인 split-detection network를 활용한 deepfake detection model입니다. 현재 사회적으로 많은 분야에 deepfake를 활용한 영상과 사진들이 생겨나고 있습니다. 이는 긍정적인 방면 뿐만 아니라 불법 성인물, 가짜 뉴스, 금융 사기 등에 악용되기도 합니다. 또한, 이를 해결하기 위한 deepfake detection 기술들이 발전하고 있는데요. 저희 팀은 이러한 문제에 관심을 가지고, 더욱 효과적인 deepfake detection model을 개발하여 사회적 문제에 기여하는 것을 최종 목표로 삼았습니다.
Deepfake image를 입력하면 해당 이미지가 위조된 이미지인지 위조되지 않은 이미지인지를 알려주는 모델을 구성하였습니다. 
해당 모델을 웹페이지 등을 통해서 사용하면 편리하게 이용할 수 있습니다.
[Wep ]

## 설계과정

초기 목표는 Kaggle에 Deepfake Detection Challange처럼 단순히 학습된 모델이 우수한 성능을 내자는 것이었습니다. 저희는 이러한 목표를 가지고 모델 설계를 위한 조사를 하다가, Deepfake를 통해 변조된 이미지들은 제작 방식에 따라 크게 2가지인 Gan, Non-Gan 방식 나눌 수 있으며 이렇게 제조된 이미지들을 각기 제작된 방식에 따라 특징이 있다는 것을 알 수 있었습니다.
각각 서로 다른 특징에 맞게 좋은 퍼포먼스를 보이는 모델이 있는데 저희는 어떤 이미지에 대해서도 하나의 모델을 통해 분류할 수 있도록 모델을 구성하였습니다.
예를 들어서 대체적으로 Non-Gan으로 제작된 이미지의 눈과 코를 통해 판별하며, Gan 이미지에서는 피부를 통해 판별하는 편입니다. 저희는 기존의 모델들은 이러한 Gan과 Non-gan으로 제작된 이미지들을 구분하지 않고 학습하는 점에 주목하여, 이미지의 생성방식에 따라 나누어서 학습한다면 Deepfake Detection에 더욱 좋은 효과를 보일 것이라는 아이디어를 내어서, 이러한 모델을 설계하고 발전시키었습니다.


## 모델 설명

이를 위해 저희는 입력된 이미지가 Gan으로 제작되었는지, Non-Gan으로 제작된 이미지인지를 먼저 구분하는 판별기를 학습시킨 후, 이를 통해 판별된 이미지들을 그 생성방식에 따라 최적화된 Classifier model로 이미지를 보내서 최종적으로 판별하는 구조를 만들었습니다. 전체적인 구조는 다음과 같습니다.   
![image](https://user-images.githubusercontent.com/32592754/118757968-1074b580-b8a9-11eb-8d81-241af2d56e4d.png)   

(이하 모델 설명)   

만들어진 모델을 테스팅하였을 때...   
![image](https://user-images.githubusercontent.com/32592754/118758037-3437fb80-b8a9-11eb-8095-383c7be8a6c2.png)   
raw 데이터와 JPEG 55 압축에선 baseline model과 비슷한 결과를 보였고, JPEG 75압축에서는 1정도 낮은 결과를 보였지만, downscale에서는 눈에 띄게 좋은 결과를 얻을 수 있었습니다. 결론적으로 저희 모델은 원본 이미지 파일 형식 판별에서도 좋은 성능을 보이며, 특히 DownSampling된 파일에서 기존 모델보다 강한 성능을 보이는 모델을 개발하였습니다.   

## 데이터셋
[Dataset](https://github.com/neolgu/Split-Detection-Network/wiki/Dataset)

## 시연 및 소개 영상
[Specific Description Video](https://github.com/neolgu/Split-Detection-Network/wiki/Specific-Description-Video)

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

