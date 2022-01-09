from loaders import get_loaders

if __name__ == "__main__":

    device = "cuda"

    #net = LeNetStudent().to(device)

    batch_size_train = 512
    batch_size_test = 1024
    nb_epoch = 60

    train_loader, test_loader = get_loaders(batch_size_train, batch_size_test)

    print(train_loader)
    print(test_loader)
