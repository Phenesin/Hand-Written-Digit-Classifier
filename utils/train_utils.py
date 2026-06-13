import torch

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device
):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim = 1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_accuracy = correct / total

    return avg_loss, avg_accuracy



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim = 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss/len(loader)
    avg_accuracy = correct / total
    return avg_loss, avg_accuracy