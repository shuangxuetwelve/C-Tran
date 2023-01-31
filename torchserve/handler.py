# Third-party imports
import ts.torch_handler.vision_handler
import ts.utils.util
import torch
import torchvision

class Handler(ts.torch_handler.vision_handler.VisionHandler):

    # Constants
    image_scale_size = 576
    topk = 5

    image_processing = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_scale_size, image_scale_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    def postprocess(self, data):
        confidences = torch.sigmoid(data)
        probs, classes = torch.topk(confidences, self.topk, dim=1)
        return ts.utils.util.map_class_to_label(probs.tolist(), self.mapping, classes.tolist())
