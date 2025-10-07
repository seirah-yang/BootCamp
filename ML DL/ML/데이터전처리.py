# 데이터를 일정한 범위로 변환하거나 스케일을 조정하여 모델의 성능을 향상시키 데이터 정규화 -> 알고리즘 안정성이 향상됨
# 데이터의 스케일이 다르면 일부 머신 러닝 알고리즘은 수렴하기 어려워질 수 있기 때문이다. 

# 데이터 준비 
import pandas as pd 
import numpy as np 

df = pd.DataFrame({'x1' : np.arange(11), 'x2' : np.arange(11) ** 2}) 

# 정규화 방법 
# 1. Z-score standardization - 평균과 표준 편차를 사용하여 표준화
  # 수식: (X - 평균) / 표준 편차
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()  # StandardScaler()를 통해 객체를 생성
df_std = scaler.fit_transform(df)  #scaler.fit_transform(df)를 통해 변환

pd.DataFrame(df_std, columns = ['x1_std', 'x2_std']) #변환된 데이터를 dataframe으로 만들어 준다. 

# 2. Min-Max Scaling (Normalization) - [0,1] or [-1,1] 범위로 조정 
  # 수식: (X-최솟값)/(최댓값-최솟값)
from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()  #MinMaxScaler()를 통해 객체를 생성
df_minmax = scaler.fit_transform(df) #scaler.fit_transform(df)를 통해 변환
pd.DataFrame(df_minmax, columns = ['x1_minmax', 'x2_minmax']) #변환된 데이터를 dataframe으로 만들어 준다. 
