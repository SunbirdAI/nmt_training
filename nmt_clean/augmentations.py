class Augmentations():

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def add(self, augmentation):
        ## TODO add type check for augmentations here
        #aug.__class__.__bases__[0] == nlpaug.augmenter.word.word_augmenter.WordAugmenter or whatever Char, etc
        self.augmentations.append(augmentation)

    def __call__(self,X, *args, **kwds):
        for augmentation in self.augmentations:
            X = augmentation.augment(X)
        return X