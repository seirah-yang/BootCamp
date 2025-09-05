
import random
import numpy as np
import torch 

#python 기본난수 
random.seed(42)
print(random.randint(1,100))

#Numpy 난수
np.random.seed(42)
print(np.random.rand(3))

#pytorch 난수
torch.manual_seed(42)
print(torch.randn(2,2))

#위 코드를 여러번 실행해도 결과가 항상 똑같은 
#colab에서는 seed가 매번 바뀌기 떄문에 random.seed 가 사실상 영향력이 없음 

#🔹 예시 (R)
set.seed(123)
runif(5) 
#set.seed(123)을 고정하면, 항상 같은 난수가 생성됨.

