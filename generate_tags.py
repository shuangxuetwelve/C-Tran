import sys
import torch
import torchvision
import PIL
from models import CTranModel

num_labels = 20
use_lmt = True
pos_emb = False
layers = 3
heads = 4
dropout = 0.0
no_x_features = False
image_scale_size = 576
num_labels = 20
category_info = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
                 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
                 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

def generate_tags():
    # Read the image.
    image_path = get_image_path()
    image = PIL.Image.open(image_path)

    # Transform the image.
    image_transformations = compose_image_transformations()
    image_transformed = image_transformations(image)

    # Compose the mask.
    mask = torch.full((num_labels,), -1)

    # Compose the inputs
    images = image_transformed[None, :]
    masks = mask[None, :]
    
    # Load a pre-trained model.
    model = CTranModel(num_labels, use_lmt, pos_emb, layers, heads, dropout, no_x_features)
    model = model.cuda()
    model = load_saved_model("results/voc.3layer.bsz_8.adam1e-05.lmt.unk_loss/best_model.pt", model)
    model.eval()

    # Predict.
    pred, int_pred, attns = model(images.cuda(), masks.cuda())
    confidence = torch.sigmoid(pred)

    # Organize and print results.
    results = {}
    for i in range(num_labels):
        results[category_info[i]] = confidence[0][i].item()
    result_str = ""
    for i in range(num_labels):
        key, value = get_largest_from_dict(results)
        del results[key]
        result_str += key + " (" + str(value) + ") "
    print(result_str)

def get_image_path():
    if len(sys.argv) < 2:
        sys.exit("No image path is given!")
    
    return sys.argv[1]

def compose_image_transformations():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_scale_size, image_scale_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def load_saved_model(saved_model_name,model):
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def get_largest_from_dict(dict):
    max = sys.float_info.min
    key_max = None
    value_max = None
    for key, value in dict.items():
        if value > max:
            key_max = key
            value_max = value
            max = value
    return key_max, value_max

generate_tags()
