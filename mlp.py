import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pdb
import os
import argparse
from utils import progress_bar
import time

from src.rigl_torch.rigl_constant_fan import RigLConstFanScheduler

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        
        self.relu = nn.ReLU()
        
        self.recurrent_layer1 = nn.Linear(input_dim, hidden_dim)
        #self.recurrent_layer2 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        #x = self.relu(x)
        x = x.view(x.size(0), -1)

        for _ in range(self.num_layers):
            x = self.recurrent_layer1(x)
            x = self.relu(x)
        """ for _ in range(self.num_layers):
            x = self.recurrent_layer2(x)
            x = self.relu(x)
         """
        x = self.fc1(x)
        
        return x
    

""" model = MLP(3072, 3072, 10, 1)

optimizer = torch.optim.SGD(params=model.parameters(), *args, **kwargs)  # SGD or AdamW supported currently

#Dataloader for Cifar10

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_epochs = 100

# Recommended kwargs
scheduler_kwargs = dict(
    dense_allocation = 0.1,  # (1-sparsity) up to you to set 
    T_end = int(len(trainloader)*num_epochs*0.75),   # optim step number at which to stop mask mutation, 75% is what we used. 
    # Will need to adjust for distributed training or grad accumulation, see https://github.com/calgaryml/condensed-sparsity/blob/main/src/rigl_torch/utils/rigl_utils.py#L239 for an example of how this can be calculated for more complex training runs. 
    dynamic_ablation = True , # ablate low saliency neurons
    min_salient_weights_per_neuron = 0.3,  # 30% of sparse weights must be salient or else neuron is pruned
    no_ablation_module_names = list(model.named_modules())[-1][0],  # Important to not ablate your last layer as this would remove entire classes from consideration. 
)
scheduler = RigLConstFanScheduler(model, optimizer, **scheduler_kwargs)

criterion = nn.CrossEntropyLoss()


use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.model}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
model.cuda()
maxAcc = 0
for epoch in range(0, num_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    scheduler.step(epoch-1) # step cosine scheduling

    list_loss.append(val_loss)
    list_acc.append(acc)

    #Save the maximum acc to a file
    if acc > maxAcc:
        maxAcc = acc

    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})
    # Write out csv..
    with open(f'log/log_{args.model}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)
    print(list_loss)


# train loop
for data, labels in data_loader:
    ....
    loss.backward()
    optimizer.step()
    scheduler()  # we use __call__ to step the SRigL scheduler, returns True if mask was modified, False if not
    optimizer.zero_grad() """