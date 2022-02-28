class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'webvision':
            return '/data/webvision/'
        elif dataset == 'clothing':
            return '/data/clothing1M/'
        elif dataset == 'miniimagenet_preset':
            return 'miniImagenet/miniimagenet_web/dataset/mini-imagenet/'
        elif dataset == 'cifar100':
            return 'samples/cifar100/'
        elif dataset == 'imagenet32':
            return 'samples/'
        else:
            raise NotImplementedError('Dataset {} not available.'.format(dataset))
        
