import uvicorn
import torchvision.transforms as transforms
from PIL import Image
import torch
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
import os

torch.manual_seed(777)

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = torch.nn.Linear(56 * 56 * 64, 2, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def hello(request: Request):
    return templates.TemplateResponse("index.html", {'request': request, 'a':2})

@app.post('/uploader')
async def uploader_file(request: Request,file: UploadFile = File(...)):
    content = await file.read()
    file_save_folder = './'
    with open(os.path.join(file_save_folder, file.filename), "wb") as fp:
        fp.write(content)
    output = infer(file_save_folder+file.filename)
    return templates.TemplateResponse("CNN_result.html", {'request': request,'result':output})


def infer(filename):
    model = CNN()
    model.load_state_dict(torch.load('cnn_model.pt', map_location=torch.device('cpu')))  

    img = Image.open(filename)
    transform = transforms.Compose([ # torchvision의 전처리 코드
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

    img_tensor = transform(img) 
    img_tensor = img_tensor.unsqueeze(0) 
    print(img_tensor.shape)

    prediction = model(img_tensor)
    result = torch.argmax(prediction, 1) 
    result = result.tolist()[0] 
    return result 


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)