# Scikit-learn에 있는 유방암 데이터셋을 이용한 데이터 셋 분리 학습 

from sklearn.model_selection import train_test_split # scikit-learn 라이브러리에서 train_test_split 불러오기
from sklearn.datasets import load_breast_cancer 

cancer = load_breast_cancer() 
x = cancer.data 
y = cancer.target

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

# load_breast_cancer()를 이용하여 유방암 데이터셋을 불러오고 x, y에 feature값과 target값을 저장 
# train_test_split을 이용하여 y의 클래스 비율을 맞추고 random_state는 42, 데이터를 섞어 train test를 8:2로 분할
