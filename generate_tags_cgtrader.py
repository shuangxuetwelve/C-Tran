import sys
import torch
import torchvision
import PIL
from models import CTranModel

use_lmt = True
pos_emb = False
layers = 3
heads = 4
dropout = 0.1
no_x_features = False
image_scale_size = 576
num_labels = 1000

def generate_tags():
    category_info = read_category_info()

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
    model = load_saved_model("results/cgtrader-3000.3layer.bsz_16.adam1e-05.lmt.unk_loss/best_model.pt", model)

    # Predict.
    pred, int_pred, attns = model(images.cuda(), masks.cuda())
    confidence = torch.sigmoid(pred)

    # Organize and print results.
    results = {}
    for i in range(num_labels):
        results[category_info[i]] = confidence[0][i].item()
    result_str = ""
    for i in range(5):
        key, value = get_largest_from_dict(results)
        del results[key]
        result_str += key + " (" + str(value) + ") "
    print(result_str)

def read_category_info():
    # 获取所有互不相同的标签，为每一个标签赋予从0开始递增的标签，并储存在self.category_info中。
    with open('data/cgtrader/cgtrader-3000/tags_needed.txt') as file:
        lines = file.read().split('\n')[:-1]
        category_indices = {}
        for i in range(num_labels):
            category_indices[lines[i]] = i
        file.close()
    category_info = {}
    for key, value  in category_indices.items():
        category_info[value] = key
    return category_info

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
