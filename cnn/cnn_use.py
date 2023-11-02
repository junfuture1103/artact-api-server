import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from CNN import CNN  # Import your CNN model from the 'CNN' module (if not already done).
from CNN import DEVICE, train, evaluate

# 모델 클래스 레이블 설정
class_labels = ["detail1","detail2","detail3","detail4","detail5","detail6","detail7","detail8","detail9","detail10","detail11","detail12"]

model = CNN().to(DEVICE)
model.load_state_dict(torch.load('../../../artact-api-server/cnn/model/model.pth'))
model.eval()

# 이미지를 읽어옵니다. 이미지 파일 경로를 지정해야 합니다.
image_path = 'uploads/'+sys.argv[1]  # 이미지 파일 경로를 적절히 지정하세요.
# image_path = 'uploads/test2.jpg'
# image_path = 'uploads/1698705997054-photo.jpg'
print("image_path: ",image_path)
image = Image.open(image_path)

# 이미지 변환을 정의합니다. 모델을 훈련할 때와 동일한 변환을 사용해야 합니다.
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 이미지 변환을 적용합니다.
image = transform(image)

# 이미지를 모델에 전달하여 예측합니다.
with torch.no_grad():
    image = image.unsqueeze(0)  # 배치 차원을 추가합니다.
    output = model(image)

# 예측 결과 중 가장 높은 확률을 가지는 클래스 인덱스를 찾습니다.
predicted_class_index = torch.argmax(output, dim=1).item()

# 해당 인덱스를 사용하여 클래스 레이블을 가져옵니다.
predicted_class = class_labels[predicted_class_index]

# 예측 결과 출력
print(f"resultURL:{predicted_class}")

# 이미지를 삭제
if os.path.exists(image_path):
    os.remove(image_path)
    print(f"이미지 삭제: {image_path}")
else:
    print(f"이미지를 찾을 수 없음: {image_path}")
    
sys.exit(0)
