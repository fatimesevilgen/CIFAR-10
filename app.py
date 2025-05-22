import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from predict import Net

@st.cache_resource
def load_model():
    model = Net()
    state = torch.load('cifar_net.pth', map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

net = load_model()
classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

st.markdown(
    "<h1 style='text-align:center;'>🎨 CIFAR-10 Görüntü Sınıflandırıcı</h1>"
    "<hr style='width:90%;margin:auto;border:1px solid #ccc; margin: 0 auto 40px;'>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📂 Bir resim yükleyin", type=['png','jpg','jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Yüklenen Görsel', use_container_width=True)

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = net(input_tensor)
        _, pred = torch.max(outputs, 1)
    st.markdown(f"**Tahmin edilen sınıf:** {classes[pred.item()]}")

