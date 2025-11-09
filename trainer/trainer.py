import torch 
from diffusion.optimizer import get_scheduler
from diffusion.loss import get_loss

class Trainer:
    def __init__(self, train_loader, test_loader, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None # TODO
        self.train_loader = train_loader # TODO
        self.test_loader = test_loader # TODO

        # self.optimizer = Adam(self.model.parameters(), lr=config.lr)

    def _testing(self):
        # scheduler = get_scheduler(self.optimizer, self.config)
        # print(scheduler)

        loss = get_loss(self.config.get("loss_type", "l1"))
        print(loss)
