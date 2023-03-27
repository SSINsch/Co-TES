
# To do  
- 기본 실험 : 진행 완료
  - ```CNN + CNN```, ```CNN + LSTM```, ```CNN + FCN```, ```LSTM + FCN```, ```LSTM + LSTM```, ```FCN + FCN```  (100 epoch) * 5
  - ```CNN + CNN```, ```CNN + LSTM```, ```CNN + FCN```, ```LSTM + FCN```, ```LSTM + LSTM```, ```FCN + FCN```  (200 epoch) * 3
  - ```LSTM + LSTM``` 모델 ```50```, ```100```, ```300``` 차원에 대해 조합 실험 (60 epoch) * 3
  - 그리고 위의 실험들 Loss 그래프

- 모델 파라미터 개수(모델 크기)에 따른 조합 결과 : 진행 중
  - ```CNN + FCN50```, ```CNN + FCN300```, ```CNN + FCN1500``` (200 epoch) * 1
  - (실험 ing) ```CNN + LSTM50```, ```CNN + LSTM300```, ```CNN + LSTM400``` (200 epoch) * 1

- ```~20 epoch coteaching w/ Co-TES``` , ```(20~100 epoch coteaching+ w/ Co-TES)``` : 진행 예정
  - 성능 상의 차이를 보이는지 보고자

- 모델 크기에 따른 파라미터 개수 조사 or 초기 weigth vector L2 norm 조사
  - Decoupling 논문의 upper bound 관련 내용