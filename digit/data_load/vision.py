import os
from typing import Any, Optional, Tuple, Union, List, LiteralString
import torch.utils.data as data


class VisionDataset(data.Dataset):
    """
    A base class for vision-related datasets.


    Attributes:
        _repr_indent (int): Indentation level for the representation of the dataset.
        root (str): The root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    _repr_indent = 4

    def __init__(self, root: str, transforms: Optional[callable] = None, transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None):
        """

        :param root: Root directory of the dataset.
        :param transforms: A function/transform that takes input sample and its target as entry and returns a transformed version.
        :param transform: A function/transform that takes in an PIL image and returns a transformed version.
        :param target_transform: A function/transform that takes in the target and transforms it.
        """
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: Index

        :return: tuple: (sample, target) where target is class_index of the target class.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        :return: The length of the dataset.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        :return: The string representation of the Dataset object.
        """
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @staticmethod
    def _format_transform_repr(transform: callable, head: str) -> list[LiteralString | str]:
        """
        Helper method to format the transform representation.

        :param transform: The transform function.
        :param head: The header string for the representation.

        :return: The formatted representation string.
        """
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        """

        :return: Extra representation string to be added to the standard representation string.
        """
        return ""


class StandardTransform(object):
    """
    A standard transform to be used with the Dataset class.

    :param transform: A function/transform that takes in an PIL image and returns a transformed version.
    :param target_transform: A function/transform that takes in the target and transforms it.
    """

    def __init__(self, transform: Optional[callable] = None, target_transform: Optional[callable] = None):
        """

        :param transform: A function/transform that takes in an PIL image and returns a transformed version.
        :param target_transform: A function/transform that takes in the target and transforms it.
        """
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input_: Any, target: Any) -> Tuple[Any, Any]:
        """
        Apply the transform to the input and target.

        :param input_: The input data to be transformed.
        :param target: The target data to be transformed.

        :return: tuple: The transformed input and target data.
        """
        if self.transform is not None:
            input_ = self.transform(input_)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input_, target

    @staticmethod
    def _format_transform_repr(transform: callable, head: str) -> list[LiteralString | str]:
        """
        Helper method to format the transform representation.

        :param transform: The transform function.
        :param head: The header string for the representation.

        :return: The formatted representation string.
        """
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        """

        :return: The string representation of the StandardTransform object.
        """
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return '\n'.join(body)
