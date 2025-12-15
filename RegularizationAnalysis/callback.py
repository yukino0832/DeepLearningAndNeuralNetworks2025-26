import torch

class StepLossAccInfoTorch:
    def __init__(self, model, eval_loader, print_iter, early_stop, cfg):
        self.model = model
        self.eval_loader = eval_loader
        self.print_iter = print_iter
        self.early_stop = early_stop
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss = []
        self.acc = []

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        for x, y in self.eval_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        self.model.train()
        return correct / total

    def step_end(self, cur_epoch, cur_step, loss_value):
        if self.cfg.stop:
            return

        if cur_step % self.print_iter == 0:
            acc = self.evaluate()

            self.loss.append(loss_value)
            self.acc.append(acc)

            print(
                f'epoch: {cur_epoch}, step: {cur_step}, '
                f'loss_train: {loss_value:.6f}, acc_test: {acc:.4f}'
            )

            # Early Stop
            if self.early_stop:
                if loss_value > self.cfg.min_val_loss:
                    self.cfg.count += 1
                else:
                    self.cfg.min_val_loss = loss_value
                    self.cfg.count = 0

                if self.cfg.count == self.cfg.MAX_COUNT:
                    self.cfg.stop = True
                    print('=' * 10, 'early stopped', '=' * 10)
