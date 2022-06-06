import logging
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from fed_distill.resnet import ResNet18
from fed_distill.data import (
    DeepInversionDataset,
    GrowingDatasetDataLoader,
    RandomSampler,
)
from fed_distill.cifar10 import CIFAR10_TEST_TRANSFORM, CIFAR10_INVERSION_TRANSFORM
from fed_distill.deep_inv import AdaptiveDeepInversion, ADILoss, DILoss, DeepInversion
from fed_distill.train import AccuracyTester, Trainer


logger = logging.getLogger("fed_distill")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main() -> None:
    teacher = ResNet18().to("cuda")
    teacher.load_state_dict(torch.load("/mlodata1/jellouli/experiment0/teacher0.pt")["best_model"])

    test_set = CIFAR10("/mlodata1/jellouli", train=False, transform=CIFAR10_TEST_TRANSFORM)
    tester = AccuracyTester(DataLoader(test_set, batch_size=2048))
    logger.info("Teacher accuracy %f", tester(teacher))

    # Deep Inversion
    batch_size = 256
    di_loss = DILoss(l2_scale=0.0, var_scale=1e-3, bn_scale=10)
    inputs = torch.randn((batch_size, 3, 32, 32), device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([inputs], lr=0.1)
    sampler = RandomSampler(256, 10)
    di = DeepInversion(di_loss, optimizer, teacher, 1000, True) 

    di_base = DeepInversionDataset(
        di.iterator_from_sampler(sampler),
        new_batch_transform=CIFAR10_INVERSION_TRANSFORM,
    )

    if not di_base.load("/mlodata1/jellouli/new_exp0/initial_batches.pt"):
        for i in range(50):
            logger.info("Generating initial batch %i", i)
            di_base.grow()
        di_base.save("/mlodata1/jellouli/new_exp0/initial_batches.pt")

    # student Training
    student = ResNet18().to("cuda")
    adi_loss = ADILoss(l2_scale=0.0, var_scale=1e-3, bn_scale=10, comp_scale=10)

    adi = AdaptiveDeepInversion(adi_loss, optimizer, teacher, student, 1000, True) 
    adi_dataset = DeepInversionDataset(
        adi.iterator_from_sampler(sampler), di_base, CIFAR10_INVERSION_TRANSFORM
    )
    loader = GrowingDatasetDataLoader(adi_dataset, 50000, 2, 256, True)

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 250], 0.1)
    trainer = Trainer(student, criterion, optimizer, scheduler, loader, tester)

    trainer.train(300)

    adi_dataset.save("/mlodata1/jellouli/new_exp0/batches.pt")

    torch.save(trainer.metrics, "/mlodata1/jellouli/new_exp0/result.pt")

if __name__ == "__main__":
    main()