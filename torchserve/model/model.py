from CTran import CTranModel

class Model(CTranModel):
    
    num_labels = 1000
    use_lmt = True
    dropout = 0

    def __init__(self):
        super(Model, self).__init__(num_labels=self.num_labels, use_lmt=self.use_lmt, dropout=self.dropout)
