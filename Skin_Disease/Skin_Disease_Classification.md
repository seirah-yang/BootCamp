# 피부질환 분류와 ResNext 학습
  Github URL: https://github.com/seirah-yang/BootCamp.git

1. 피부질환 분류 학습 프로젝트 

    - AWS S3를 활용하여 데이터를 업로드 및 다운로드 할 수 있다. 
    - 다운로드한 Kaggle 피부질환 데이터를 이용하여 ResNext 분류 모델 학습을 수행한다.
      
3. 일시: 2025년 8월 28일 

   - 장소: 알파코 딥러닝 부트캠프

5. 데이터
  Kaggle Skin Disease Classification [Image Dataset] : https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset
   - Kaggle에서 다운로드 받은 skin Disease Classification Image DAtaset train/valid 데이터를 사용하였다.
     
6. 기술스택
   - !pip install boto3
     : boto3를 설치하여 AWS S3를 활용하여 데이터베이스를 생성하여 데이터를 공유 및 다운로드 받고 업로드 하였다.
  - class CustomDataset을 사용하고, def len, getitem을 이용하여 파일명을 생성하여 저장 할 수 있또록 반복문을 활용하였다. 
      ''' class CustomDataset(Dataset):
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
                return image, label '''
- 모델은 2017년 발간된 "Aggregated Residual Transformations for Deep Neural Networks" 을 사용하였다.

  ''' model = models.resnext50_32x4d(pretrained=True) 
      model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.classes))'''
  
7. 결과 
   - 설계도:
   
   ![](https://)

   - 결과: 

   ![](https://)

8.결론 및 제언 

   - 
   
   -  

6. 참고문헌

Shaju, R. E. (2022). Skin disease classification [Image dataset]. Kaggle. https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated residual transformations for deep neural networks. arXiv. https://arxiv.org/abs/1611.05431
