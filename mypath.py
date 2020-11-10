class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'webvision':
            return '/home/paul/Documents/data/webvision/'
        elif dataset == 'clothing':
            return '/home/paul/Documents/data/clothing1M/'
        elif dataset == 'miniimagenet_preset':
            return '/home/paul/Documents/miniImagenet/miniimagenet_web/dataset/mini-imagenet/'        
        else:
            raise NotImplementedError('Dataset {} not available.'.format(dataset))
        
