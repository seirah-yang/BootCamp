# 피부질환 분류와 ResNext 학습
  Github URL: https://github.com/seirah-yang/BootCamp.git

1. 피부질환 분류 학습 프로젝트 
  본 프로젝트는 ImageNet 사전학습(1k) 가중치를 사용한 ResNeXt 아키텍처를 기반으로 전이학습을 수행하였다. 
  
  ResNeXt는 잔차 연결 위에 다중 병렬 변환의 집계를 통해 성능을 향상 시키는 구조이며, learning rate schedule은 ResNet과 ResNeXt 계열에서 보편적으로 사용하는 step decay을 일반화 한 모델이다. 
   
  본 프로젝트를 통해 다음과 같은 세부사항을 수행 할 것을 계획 하였다. 

   1) AWS S3를 활용하여 데이터를 업로드 및 다운로드
  
   2) Kaggle 피부질환 데이터를 이용하여 ResNeXt 분류모델 학습

  ResNext 분류 모델 학습 후, 결과 개선을 위해 다음과 같이 계획하였다. 
   1) loss가 가장 많이 감소하는 epochs= "10", "30", "60"을(Xie et al., 2017)를 참고하여 epochs를 조절 하여 학습을 수행한 후 결과를 비교한다. 
  
   2) ResNeXt계열에서 보편적으로 사용하는 step-decay관행을 에폭의 50%, 75% 지점에서 1/10배 감소(Xie et al., 2017)하도록 적용하고, 결과를 비교한다. 
      
2. 일시: 2025년 8월 28일, 알파코 End-to-End AI 개발자과정 2기 부트캠프

3. train/test Data  
  Kaggle Skin Disease Classification [Image Dataset]: https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset
  
  - Kaggle에서 다운로드 받은 skin Disease Classification [Image Dataset]을 train, test를 위해 사용하였다.
     
4. 기술스택

  1) boto3를 설치하여 AWS S3를 활용하여 데이터베이스를 생성하여 데이터를 공유 및 다운로드 받고 업로드 하였다.
  ```bash
    !pip install boto3
  ```   
  2) class CustomDataset을 사용하고, def len, getitem을 이용하여 파일명을 생성하여 저장 할 수 있또록 반복문을 활용하였다. 
    
  ```bash
  class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.data = []
        for label in range(len(self.classes)):
            class_folder = os.path.join(root_dir, self.classes[label])
          
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            self.data.append((img_path, label))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
            return image, label
  ```
    
  3) 학습을 위한 모델은 2017년 발간된 "Aggregated Residual Transformations for Deep Neural Networks"을 사용하였다.

  ```bash
    model = models.resnext50_32x4d(pretrained=True) 
    model.fc = torch.nn.Linear(model.fc.in_features, 
                              len(train_dataset.classes))
  ```

5. 결과

 - 학습결과 비교
  1) epochs = 10
      :  
   ![img1.epochs10](https:///Users/gom3ku/Documents/BootCamp/Skin_Disease/epochs10.png)
  
  2) epochs = 30
      :   
   ![](https://)
   
  3) epochs = 60 
      :   
   ![](https://)
   
  4) learning rate scheduler 적용
  
   ![img4.lr_scheduler](https://Users/gom3ku/Documents/BootCamp/Skin_Disease/lr_schedule.png)

   - epochs = 10 설정하여 학습을 수행 했을 때의 결과 

   ![graph1.Accuracy](https://github.com/seirah-yang/BootCamp/blob/main/Skin_Disease/lr_scheduler_result1.png)

   - validation Accuracy: epochs = 4 이후 약간 하향 후 유지하는 양상을 보인다. 

   - train Accyracy: 상승하는 양상을 보이다가 epochs = 5이후로 평탄한 양상을 보인다. 

   - 이를 통해 과적합의 위험을 예측 해 볼 수 있다. 
    
   ![graph2.Loss]()

   - validation Accuracy: 하향하다가 epochs = 4 이후로 정체되는 양상을 보인다.  

   - train Accuracy: 하향하다가 epochs = 4 이후로 정체되는 양상을 보인다.  

   ![graph3.Learning_Rate](https://Users/gom3ku/Documents/BootCamp/Skin_Disease/lr_scheduler_result3.png)


6.결론 및 제언 

  - 조기 종료(early stopping) 시점을 epochs = 4 이후로 두고 학습을 중단 하는 것이 적합할 것으로 사료된다.
  
  - 피부 질병의 경우 병변의 연속성과 진단의 모호성을 고려하여 정상피부에 기반한 병변과  병변 주변의 이상 탐지 및 레이블 없는 데이터를 활용(Lu & Xu (2018))하는 학습 방법을 고려 할 수 있다. 

7. 참고문헌

  Lu, Y., & Xu, J. (2018). Anomaly detection for skin disease images using variational autoencoder. arXiv preprint arXiv:1807.01349. https://arxiv.org/abs/1807.01349
  
  Shaju, R. E. (2022). Skin disease classification [Image dataset]. Kaggle. https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset
  
  Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated residual transformations for deep neural networks. arXiv. https://arxiv.org/abs/1611.05431

