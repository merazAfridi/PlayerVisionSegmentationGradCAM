import os
import uuid
import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#import model 
from models.unet_resnet50 import get_unet_resnet50
from models.unet_resnet50_3epoch import get_unet_resnet50_3epoch
#from models.linknet_vgg16 import get_linknet_vgg16

app = Flask(__name__)

#configure directories
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
MODEL_FOLDER = 'checkpoints'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

#img preprocessing
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#mapping from model name to model creation function
MODEL_MAPPING = {
    "unet_resnet50": get_unet_resnet50,
    "unet_resnet50(3epoch)": get_unet_resnet50_3epoch,
    #"linknet_vgg16": get_linknet_vgg16,
}

def remove_module_prefix(state_dict):
    #remove 'module'.prefix if the checkpoint was saved using DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    return new_state_dict

def load_model(model_path, model_name):
    #load selected model from checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    #creating correct model based on  selection
    model = MODEL_MAPPING[model_name]()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    #handle checkpoint formats that either use 'model_state_dict'/ are the state dict themselves
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    #remove any 'module' prefix if present
    state_dict = remove_module_prefix(state_dict)
    
    #loading the state dictionary; using strict=False to ignore minor mismatches
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def predict_mask(model, image_path):
  
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize((512, 512))
    
    #save resized input img
    resized_filename = f"{uuid.uuid4().hex}_resized.png"
    resized_path = os.path.join(RESULT_FOLDER, resized_filename)
    resized_image.save(resized_path)
    
    #Preprocess image
    input_tensor = preprocess(resized_image).unsqueeze(0)
    
    #run inference
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype('uint8') * 255  # Convert to binary mask

    #s@ving predicted mask
    mask_filename = f"{uuid.uuid4().hex}_mask.png"
    mask_path = os.path.join(RESULT_FOLDER, mask_filename)
    Image.fromarray(mask).save(mask_path)
    
    return resized_filename, mask_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        model_name = request.form['model']
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        #predicting & get filenames
        original_filename, mask_filename = predict_mask(
            load_model(os.path.join(MODEL_FOLDER, f"{model_name}.pth"), model_name),
            image_path
        )
        return render_template(
            'result.html',
            original_image=original_filename,
            mask_image=mask_filename
        )
    return render_template('index.html')

def process_image(image_path, model_name):
    #loading & process img
    image = Image.open(image_path).convert('RGB')
    processed = preprocess(image).unsqueeze(0)
    
    #Load model& predict
    model = load_model(os.path.join(MODEL_FOLDER, f"{model_name}.pth"), model_name)
    model.eval()
    
    with torch.no_grad():
        prediction = model(processed)
        mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype('uint8') * 255  # Convert to binary mask

    #generate unique filename for results
    result_id = uuid.uuid4().hex

    #saving results
    results = {
        'original': f'static/results/{result_id}_original.png',
        'mask': f'static/results/{result_id}_mask.png'
    }

    #saving original
    image.save(os.path.join(app.root_path, results['original']))

    #saving mask
    Image.fromarray(mask).save(os.path.join(app.root_path, results['mask']))

    return results

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)