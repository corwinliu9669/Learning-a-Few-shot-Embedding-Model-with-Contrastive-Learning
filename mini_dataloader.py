##### from cross attention network https://github.com/blue-blue272/fewshot-CAN
from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

import transforms as T
import mini_dataset
import sampler.mini_sampler_test as sample_test
import sampler.mini_sampler_train as sample_train

class DataManager(object):
    """
    Few shot data manager
    """
    def __init__(self, args, use_gpu):
        super(DataManager, self).__init__()
        self.args = args
        self.use_gpu = use_gpu

        print("Initializing dataset {}".format(args.dataset))
        dataset = mini_dataset.miniImageNet_load()
        transform_train = T.Compose([
                T.RandomCrop(84, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            ])
        transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        pin_memory = True if use_gpu else False

        self.trainloader = DataLoader(
                sample_train.FewShotDataset_train(name='train_loader',
                    dataset=dataset.train,
                    labels2inds=dataset.train_labels2inds,
                    labelIds=dataset.train_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.train_nTestNovel,
                    epoch_size=args.train_epoch_size,
                    transform=transform_train,
                    load=args.load,
                ),
                batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=True,
            )
        self.testloader = DataLoader(
                sample_test.FewShotDataset_test(name='test_loader',
                    dataset=dataset.test,
                    labels2inds=dataset.test_labels2inds,
                    labelIds=dataset.test_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.nTestNovel,
                    epoch_size=args.epoch_size,
                    transform=transform_test,
                    load=args.load,
                ),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=False,
        )
    def return_dataloaders(self):
        return self.trainloader, self.testloader
