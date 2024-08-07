
import torch
import torch.nn as nn
import neurogym as ngym


def make_model_for_task(model_type, task, *args, **kwargs):
    n_inp = task.obs_dims
    n_out = task.action_dims
    model = model_type(n_inp, *args, n_out, **kwargs)
    return model


def train_model_on_task(
    model, task, batch_size=16, seq_len=None, lr=1e-3, num_steps=2000,
):
    dataset = ngym.Dataset(task, batch_size=batch_size, seq_len=seq_len)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(p.to(device) for p in model.parameters()), lr=lr)
    running_loss = 0.0
    for i in range(num_steps):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels).type(torch.float).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
            running_loss = 0.0
    return dataset

