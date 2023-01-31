# Third-party imports.
import torch

# Self imports.
from CTran import CTranModel

class Model(CTranModel):
    
    num_labels = 20
    use_lmt = True
    dropout = 0

    def __init__(self):
        super(Model, self).__init__(num_labels=self.num_labels, use_lmt=self.use_lmt, dropout=self.dropout)

    def forward(self, images):
        mask = torch.full((self.num_labels,), -1)
        masks = mask[None, :]
        predictions, _, _ = super(Model, self).forward(images, masks.cuda())
        return predictions
