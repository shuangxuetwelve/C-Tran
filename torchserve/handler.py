# Third-party imports
import ts.torch_handler.vision_handler
import ts.utils.util
import torch
import torchvision
from ts.utils.util import PredictionException

class Handler(ts.torch_handler.vision_handler.VisionHandler):

    # Constants
    image_scale_size = 576
    topk = 5
    API_KEY = "InFFPMYxeH/GLNcK2X+DO+ilNgHQBf1W0jASPaamvHoWFJyPOrczAozGr8+3u7J3BE/ygzTNzTHlAKoqsG6Rjg=="

    image_processing = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_scale_size, image_scale_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    def handle(self, data, context):
        # Authorize the API key.
        api_key = context.get_request_header(0, 'X-API-Key')
        if api_key != self.API_KEY:
            raise PredictionException("Unautorized", 403)

        return super().handle(data, context)

    def postprocess(self, data):
        confidences = torch.sigmoid(data)
        probs, classes = torch.topk(confidences, self.topk, dim=1)
        return ts.utils.util.map_class_to_label(probs.tolist(), self.mapping, classes.tolist())
