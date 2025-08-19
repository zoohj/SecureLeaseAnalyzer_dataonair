# data_onair
# 전세 매물 이상탐지 모델을 통한 전세사기 유형 분석 및 예방 방안 제시
2023 데이터 청년 캠퍼스 상명대학교 2분반 4조


## 프로젝트 설명

**[목적]** 💡

근래에 부동산 시장에서 전세사기 피해가 급증하고 있다. 

그중에서도 전세계약과 관련된 지식의 빈틈을 노린 사기 비중이 높다.

따라서 전세계약 이전에 올바른 정보를 제공함으로써 사전에 피해를 예방하고자 한다.

전세사기 피해 중에서도 사회초년생을 집중적으로 노리는 전세사기 비중이 높다.


➡️ 전세사기로 의심되는 **전세매물 이상탐지 모델**을 구축하여 사전에 전세사기 행위를 탐지하여 전세사기 피해를 줄이고자 전세사기 **데이터 분석** 및 **예측**, 전세사기 **유형 분석 및 예방방안** 제시를 하고자 한다.

**[프로젝트 구조]** 

    | 전세사기 의심 탐지 모델 | + | 전세사기 유형 분석 | + | 전세사기 탐지 웹 | + | 예방 방안 제시 |


**[기대효과]**

전세사기 의심 탐지 모델이 포함된 웹 플랫폼을 통해, 사회초년생과 일반대중들을 향한  전세사기로부터 위험을 감소시키고 피해를 예방한다.


------------------


## 1. 프로젝트 환경 및 라이브러리 설치목록

**[프로젝트 환경]**

<데이터 분석>

jupyter notebook

<웹>

Django, HTML, CSS, JAVA Script


**[라이브러리 설치 목록]**

|라이브러리 이름|설치 코드|
|------|---|
|numpy|!pip install numpy|
|pandas|!pip install pandas|
|matplotlib|!pip install matplotlib|
|seaborn|!pip install seaborn|
|sklearn|!pip install scikit-learn|
|statsmodels|!pip install statsmodels|
|imblearn|!pip install -U imbalanced-learn|
|selenium|!pip install selenium|
|webdriver|!pip install webdriver|
|pdf2image|! pip install pdf2image|
|opencv-python|! pip install opencv-python|
|pytesseract|! pip install pytesseract|
|yellowbrick|!pip install yellowbrick|

**[크롤링을 위한 업데이트]**

pip install selenium --upgrade

※ 크롬 드라이버가 현재로서 가장 최신 버전인 버전 116.0인 경우입니다. (2023.08.28 기준)

**[OCR을 위한 설치]**

1. Tesseract 설치방법
   https://www.youtube.com/watch?v=NbkHxYwPAiA

   유튜브 영상 7분 10초까지 똑같이 따라하면됩니다.

3. poppler 설치 경로
    https://github.com/oschwartz10612/poppler-windows/releases/
   
   Release- 23.08-0.zip을 설치합니다. (버전이 업데이트 되었다면 상황에 맞게 다른 것을 설치합니다.)



## 2. 코드 설명

**--------[코드 설명]--------**

1. 전처리 코드
   
2. 모델링 코드
   
3. 웹 코드

※ 자세한 코드 설명은 함께 제출한 코드파일의 주석을 참고하시면 됩니다.

**--------[평가 지표]--------**

1. Precision
2. Recall
3. F1 Score
4. AUC & ROC CURVE

## 3. 코드 사용방법

**3-1.** 주피터 환경에서 [1. 프로젝트 환경 및 라이브러리 설치목록] 의 [라이브러리 설치목록] 설치코드를 이용해 라이브러리를 설치합니다.

**3-2.** [1. 프로젝트 환경 및 라이브러리 설치목록]의 [OCR를 위한 설치] 링크를 따라 설치를 진행합니다. 

**3-3.** 제출한 파일을 모두 다운을 받습니다.

**3-4.** **<< 전처리제출용.ipynb파일 >>**


   **3-4-1.**  
   
   <img width="482" height="115" alt="1" src="https://github.com/user-attachments/assets/c7844da5-5d52-4721-ba24-c284a393b535" />


   
  fraud 데이터를 불러오는 곳에 '전세사기raw데이터' 엑셀파일 경로를 넣습니다.

  **3-4-2.**
  
  <img width="465" height="119" alt="2" src="https://github.com/user-attachments/assets/934f7cda-17ef-4b49-b1f9-4ba30e392bb5" />

   
   nfraud 데이터를 불러오는 곳에 '전세사기아닌raw데이터' 엑셀파일 경로를 넣습니다.

   * 전처리 엑셀 저장 부분에 excel형식과 csv형식으로 저장할 수 있는 코드를 주석처리 해놨습니다.

     

   **<< 2분반 4조 모델링 코드 최종본.ipynb >>**

   **3-4-3.** 
   
   <img width="546" height="71" alt="3" src="https://github.com/user-attachments/assets/29a1a4dd-7ec0-4741-bc48-737985af33f6" />

   위의 사진 부분에 **'0822data.csv'** 파일 주소를 넣습니다.
   
 전처리를 통해 저장한 파일과 0822data는 인덱스 유무만 다릅니다.
   
   
   크롤링 부분 코드를 돌린 후, 아래와 같은 입력란이 생기면 **'서희스타힐스아파트'** 라고 쳐야 합니다.

<img width="532" height="43" alt="4" src="https://github.com/user-attachments/assets/66b5ba45-2ee9-4701-b666-f4219551b40a" />

   

   
   **3-4-4.** 다운 받은 **완료데이터.csv**파일을 3-4-1과 3-4-2 방법처럼 큰따옴표 사이에 경로주소를 넣습니다.
   
   **3-4-5.** OCR(Optuxal Character Recognition)에서 첫번째 주석, **# Tesseract 실행 경로 지정**부분 바로 밑의 코드 큰따옴표 부분에 설치한 tesseract.exe 경로를 복사해 붙여 넣습니다.
   
   **3-4-6.** OCR(Optuxal Character Recognition)에서 두번째 주석, **# PDF 파일 경로 및 Poppler 실행 경로 지정**부분 바로 밑의 코드 작은따옴표 부분에 다운받은 등기부등본 경로를 복사해 붙여 넣습니다.

   **3-4-7.** 4-5부분의 바로 밑 코드인 poppler_path의  작은따옴표 부분에 설치한 poppler 경로를 복사해 붙여 넣습니다.
       이 때, poppler 경로는 설치한 poppler 파일의 Library에 들어가고 **bin파일의 경로**를 복사해 붙여넣어야 합니다.

   
   


## 4. 프로젝트 시현

<img width="1266" height="703" alt="5" src="https://github.com/user-attachments/assets/c5491dc7-4c1b-4969-9fd1-6f09fae48e43" />




<img width="1266" height="701" alt="8" src="https://github.com/user-attachments/assets/8075d7ee-e5a9-4218-b863-43037e0328d0" />


<img width="959" height="440" alt="6" src="https://github.com/user-attachments/assets/ab6bbdf8-4bf6-4569-8016-509ae69a1453" />


<img width="949" height="430" alt="7" src="https://github.com/user-attachments/assets/aed8f93f-8b65-46a0-b73d-de5b3dabe73f" />



## 5. 팀원명 및 팀원이름

팀원명: 2023 데이터 청년 캠퍼스 상명대 2분반 4조

팀원이름 : 고석훈, 김도윤, 김서영, 송수현, 양준석, 이수현, 주현지
