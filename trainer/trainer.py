import torch 
from diffusion.optimizer import get_scheduler
from diffusion.loss import get_loss
from torch.optim import Adam

class Trainer:
    def __init__(self, config, train_loader, test_loader, model):
        self.config = config
        self.device = config["device"]

        self.model = model
        self.train_loader = train_loader # TODO
        self.test_loader = test_loader # TODO

        self.optimizer = Adam(self.model.parameters(), lr=float(config["lr"]))

    def train(self):
        self.model.train()

        x = torch.randn(int(self.config["batch_size"]), int(self.config["in_channels"]), int(self.config["image_size"]), self.config["image_size"], device=self.device)
        t = torch.randint(0, 1000, (int(self.config["batch_size"]),), device=self.device)
        
        for epoch in range(int(self.config["n_epochs"])):
            self.optimizer.zero_grad()
            out = self.model(x, t)
            assert out.shape == x.shape, "Shape mismatch"

            loss_fn = get_loss(self.config["loss_type"])
            loss = loss_fn(out, x)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch [{epoch+1}/{self.config['n_epochs']}], Loss: {loss.item():.4f}")

        print("Test successful.")


