# ⚽ Fitogether Soccer Object Detection Project

이 프로젝트는 축구 경기 영상에서 **선수(Players)**, **공(Ball)**, 그리고 **기타 객체(Others)**를 실시간으로 탐지하기 위한 YOLO 기반의 영상 분석 플랫폼 구축 과정과 모델별 성능 분석 결과를 담고 있습니다.

---

## 1. 프로젝트 개요
* **목적**: 축구 경기 영상 데이터셋을 활용하여 고성능 객체 탐지 모델 학습 및 비교 분석
* **대상 클래스**: `players` (0), `ball` (1), `others` (2)
* **프레임워크**: `Ultralytics YOLO`, `PyTorch`

---

## 2. 개발 환경 (Development Environment)

| 구분 | 상세 사양 | 비고 |
| :--- | :--- | :--- |
| **Cloud (Colab)** | NVIDIA L4 (22GB) / T4 GPU | 데이터 전처리 및 Nano 모델 학습 |
| **Local GPU** | NVIDIA GeForce RTX 4070 Ti | Small 모델 정밀 학습 (Batch 64) |
| **OS/Language** | Ubuntu (Colab) / Windows (Local), Python 3.12.12 | |
| **Main Libraries** | `ultralytics 8.3.250`, `torch 2.9.0+cu126`, `opencv-python` | |

---

## 3. 데이터 파이프라인 (Data Pipeline)

### 🛠️ 데이터 전처리 및 변환
원본 JSON 어노테이션 데이터를 YOLO 학습에 적합한 TXT 포맷으로 변환하는 과정을 거쳤습니다.

* **BBox 변환**: JSON의 다각형(points) 데이터를 기반으로 YOLO 형식의 정규화된 좌표로 변환하였습니다.
    * $x_{center} = \frac{x_{min} + x_{max}}{2w}$
    * $y_{center} = \frac{y_{min} + y_{max}}{2h}$
    * $width = \frac{x_{max} - x_{min}}{w}$
    * $height = \frac{y_{max} - y_{min}}{h}$
* **데이터 분할 (Split)**: `random.seed(2026)`를 사용하여 `Train 80%`, `Valid 10%`, `Test 10%` 비율로 무작위 분할하였습니다.

---

## 4. 모델 성능 비교 분석 (Performance Analysis)

학습된 두 가지 모델(YOLO Nano vs Small)의 검증 지표 결과입니다.

### 📊 Metric Table

| 지표 | 모델 1: Nano (val4) | 모델 2: Small (val5) | 향상도 (Improvement) |
| :--- | :---: | :---: | :---: |
| **Parameters** | 3,006,233 (3M) | 11,126,745 (11M) | 약 3.7배 모델 크기 증가 |
| **mAP50 (전체)** | 0.559 | **0.732** | **+30.9%** |
| **mAP50-95 (전체)** | 0.308 | **0.493** | **+60.0%** |
| **Inference Speed** | 2.2ms | **2.0ms** | L4 GPU 환경에서 차이 미비 |

### 🔍 주요 분석 결과 요약

1.  **전반적인 정밀도 향상**: 모델 파라미터가 3.7배 큰 Small 모델이 전 영역에서 압도적인 성능 우위를 보입니다. 특히 `mAP50-95`가 60% 상승한 점은 **객체의 경계(Bounding Box)를 훨씬 더 정교하게 예측**하고 있음을 나타냅니다.
2.  **공(Ball) 검출 능력의 차이**:
    * **Nano 모델**: 작은 객체인 `ball`을 전혀 찾지 못함 (mAP50: 0).
    * **Small 모델**: `ball` 클래스 검출을 시작함 (mAP50: 0.232). 이는 작은 객체 특징 추출을 위해 일정 수준 이상의 모델 복잡도가 필수적임을 시사합니다.
3.  **실시간성 확보**: 고성능 GPU(L4) 환경에서는 모델 크기가 커져도 추론 속도가 2ms 내외로 유지되므로, 실시간 분석 환경에서도 Small 모델 사용이 매우 효율적입니다.

---

## 5. 학습 및 검증 설정 (Config)

### [Nano Model - Colab]
```python
model_n = YOLO('yolov8n.pt')
results_n = model_n.train(
    data='/content/fitogether.yaml',
    epochs=30,
    imgsz=416,
    batch=16,
    device=0, 
    name='fit_nano_results'
)
```
### [Small Model - Local]
```python
model_s = YOLO('yolov8s.pt')
results_s = model_s.train(
    data='local_fitogether.yaml',
    epochs=40,
    imgsz=640,
    batch=64,
    device=0,
    amp=True,
    name='fit_small_results'
)
```
![val_batch2_pred](https://github.com/user-attachments/assets/67d66879-ae4b-45ac-9855-b8871e943911)
![val_batch1_labels](https://github.com/user-attachments/assets/e2e70530-a903-4c22-825e-1aaced3a4c2f)

## 6. 결론 및 향후 과제
* 결론:
  * 축구와 같이 객체가 작고 움직임이 빠른 환경에서는 모델의 파라미터 수와 이미지 해상도(imgsz)가 성능에 결정적인 영향을 미침을 확인했습니다.
  * 해상동의 차이도 있겠지만 모델의 기본 성능에서도 차이를 느꼈다.
  * 로컬과 코랩 두 개의 환경에서 실행하다보디 Yeml파일의 경로 설정 에러로 인해 골머리를 잠깐 썼지만, 이번 계기로 어떻게 하면 되는지, 무엇을 수정하면 되는지 알게 되는 계기가 되기도 했다. 
 
**작성자: [Park Yeonggon/GitHub ID] 프로젝트 기간: 2026.01.09**
