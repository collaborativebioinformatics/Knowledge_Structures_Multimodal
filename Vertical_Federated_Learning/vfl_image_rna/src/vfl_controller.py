
from nvflare.apis.impl.controller import Controller


class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)

    def forward(self, h_img, h_rna):
        z = torch.cat([h_img, h_rna], dim=1)
        return self.fc(z)


class VFLController(Controller):
    def __init__(self):
        super().__init__()

    def start_controller(self, fl_ctx):
        num_samples = 1000
        labels = torch.randint(0, 2, (num_samples,))

        dataset = TensorDataset(labels)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.data_iter = iter(self.loader)

        self.model = ServerModel()
        self.opt = torch.optim.Adam(self.model.parameters(), 1e-3)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def control_flow(self, abort_signal, fl_ctx):
        epochs = 10
        for i in range(epochs):
            data_shareable = Shareable()
            task = Task(name="vfl_forward", data=data_shareable, received_cb=self._process_result)
            self.broadcast_and_wait(
                task=task,
                min_responses=2,
                wait_time_after_min_received=0,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
            new_grad = self._server_side_training()
    
            task2 = Task(name="vfl_backward", data=new_grad)
            self.broadcast_and_wait(
                task=task2,
                min_responses=2,
                wait_time_after_min_received=0,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
        

    def _process_result(self, client_task, fl_ctx):
        task = client_task.task
        result = client_task.result

        if "h_img" in result:
            self.h_img = shareable["h_img"]
        else:
            self.h_rna = shareable["h_rna"]

    def _server_side_training(self):
        try:
            (y,) = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            (y,) = next(self.data_iter)
        
        logits = self.model(self.h_img, self.h_rna).squeeze()
        loss = self.loss_fn(logits, y.float())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return Shareable({
            "grad_img": h_img.grad,
            "grad_rna": h_rna.grad,
            "loss": loss.item()
        })
