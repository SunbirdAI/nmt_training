class Augmentations():

    """
    A class to apply a list of augmentations to a given textual input.
    We use nlpaug augmentations, but any text --> text augmentation is supported.

    Args:
    augmentations (list): A list of augmentation objects to be applied in order.

    Methods:
    add(augmentation): Adds an augmentation object to the list of augmentations.

    Returns:
    The input after applying all augmentations in the specified order.

    Example Usage:

    augmentations = Augmentations([augmentation1, augmentation2])
    augmented_text = augmentations(text)

    """

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