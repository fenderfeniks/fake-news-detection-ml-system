import torch


def binary_accuracy(preds, y):
    probs = torch.sigmoid(preds)
    rounded = torch.round(probs)
    correct = (rounded == y).float()
    return correct.sum() / len(correct)


def train_model(model, iterator, optimizer, criterion, device):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for texts, labels in iterator:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_model(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for texts, labels in iterator:
            texts = texts.to(device)
            labels = labels.to(device)

            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            probs = torch.sigmoid(predictions)
            preds = torch.round(probs)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_labels, all_probs, all_preds


def model_fitting(model, train_iterator, test_iterator, optimizer, criterion, device, n_epochs=10):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_model(model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc, test_labels, test_probs, test_preds = evaluate_model(
            model, test_iterator, criterion, device
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}')

    return train_losses, train_accs, test_losses, test_accs, test_labels, test_probs, test_preds