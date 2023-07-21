import streamlit as st
import torchxrayvision as xrv
import skimage, torch, torchvision

def get_image(image_file):
    img = skimage.io.imread(image_file)
    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...] # Make single color channel
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    img = transform(img)
    img = torch.from_numpy(img)
    return img

def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    return model

def process_image(model, img):
    outputs = model(img[None,...]) # or model.features(img[None,...]) 
    results = dict(zip(model.pathologies,outputs[0].detach().numpy()))
    return results


st.title('X-Ray Vision Application')
st.write('Upload an X-Ray image and get the results.')

image_file = st.file_uploader('Upload Image', type=['jpg', 'png'])
if image_file is not None:
    img = get_image(image_file)
    model = load_model()
    results = process_image(model, img)
    
    st.header('Results')
    for pathology, score in results.items():
        st.write(f"{pathology}: {score}")
