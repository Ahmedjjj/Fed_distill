import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR_10_MAPPER = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

CIFAR10_TEST_TRANSFORM = T.Compose(
    [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
)
CIFAR10_TRAIN_TRANSFORM = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
CIFAR10_INVERSION_TRANSFORM = T.Compose(
    [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
)
