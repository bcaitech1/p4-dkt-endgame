
# Stage 4 - DKT  
> 2021 boostcamp AI Tech  
  
## Task  
### **📖 Knowledge Tracing란?**  
  
**Knowledge Training은 사람의 지식 상태를 추적하는 방법론입니다.**  
  
- 시험을 보는 것은 동일하지만 단순히 우리가 수학을 80점 맞았다고 알려주는 것을 넘어서 우리가 수학이라는 과목을 얼마만큼 이해하고 있는지 측정합니다. 추가적으로 이런 이해도를 활용하여 우리가 아직 풀지 않은 미래의 문제에 대해 우리가 맞을지 틀릴지 예측이 가능합니다!  
- 대회는 **미래의 문제에 대해서 맞출지 틀릴지 예측**하는 것에 집중되어 있습니다.  
- 대회를 벗어나 저희는 주어진 문제를 맞히는 데 있어 어떠한 경험들, 즉 **학생의 성장에 있어 중요한 요소가 무엇인지**를 확인하는 것에 초점을 맞추었습니다.  
- 따라서 저희는 기존의 예측만 해주는 Knowledge Tracing 모델에서 벗어나 **친절한** 모델을 만들고자 했습니다. 저희는 **[Context-Aware Attentive Knowledge Tracing (AKT)](https://arxiv.org/abs/2007.12324)** 모델을 기반으로 [**ELO rating system**](https://www.fi.muni.cz/~xpelanek/publications/CAE-elo.pdf) 을 적용하여 발전시켰습니다.  
  
  
# Model Architecture  
based by [Context-Aware Attentive Knowledge Tracing (AKT)](https://arxiv.org/abs/2007.12324)  
  
![pipeline2](https://user-images.githubusercontent.com/56197411/122345523-e8997000-cf82-11eb-968b-33c11b7b304d.PNG)  
  
[AKT](https://github.com/arghosh/AKT)는 Question Transformer Encoder와 Knowledge Transformer Encoder를 통해 문제와 유형에 대한 학습, 유형과 사용자의 응답에 대한 학습을  
각각 진행합니다. Encoder를 통해 재구성한 Sequence를 Monotonic Attention 구조를 통해 다음 문제에 대한 응답을 예측합니다.  
  
## Monotonic Attention  
![monotonic](https://user-images.githubusercontent.com/56197411/122346770-4aa6a500-cf84-11eb-95c5-56228be6759e.PNG)  
각 Transformer Layer에 사용하는 Monotonic Attention은 확장된 Attention 구조입니다. 비슷한 유형일수록, 최근에 배운 유형일수록 더 강하게  
작용합니다.  
  
  
# Usage  
  
## AKT_ELO  
  
### Train & Inference  
  
 ```shell  $ python main.py ```  
 ```shell  $ python inference.py ```  
## ELO  
### Train & Inference  
  
 ```shell  $ python elo.py ```  
## Other Models Available
- SAINT
별도 폴더에 코드와 사용예시가 존재합니다.  

# Members  
## Team ENDGAME
| 김한마루 | 소재열 | 이대훈 | 정지훈 | 최이서 | 홍승우 |  
| :-: | :-: | :-: | :-: | :-: | :-: |
|github|github|github|github|github|github|
