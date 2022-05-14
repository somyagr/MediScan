import json
import re
import flask
import io
import string
import time
import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify
import numpy as np

#Helper functions
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
        
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
                
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential( 
            nn.MaxPool2d(2,2),
            DoubleConv(in_ch, out_ch)
         )
    def forward(self, x):
        x = self.mpconv(x)
        return x

#model
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

def prepare_image(image_bytes):

    my_transforms = transforms.Compose([
            Resize(256, 256),
            transforms.ToTensor()
        ])

    image = Image.open(io.BytesIO(image_bytes))
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return my_transforms(image=np.array(image_np))['image'].unsqueeze(0)

model = UNet(3, 1)
state_dict = torch.load('UNet.pth', map_location=torch.device('cpu'))['model']
model.load_state_dict(state_dict)
model.eval()

def predict_result(image_bytes):
    """For given image bytes, predict the label using the pretrained DenseNet
    """
    tensor = prepare_image(image_bytes)
    mask = model(tensor)
    transform = transforms.ToPILImage()
    output = transform(mask)
    return output


app = Flask(__name__)
ALLOWED_EXTENSIONS= {'png','jpg','jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def infer_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error':'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            prediction = predict_result(img_bytes)
            image = {'prediction': prediction}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
