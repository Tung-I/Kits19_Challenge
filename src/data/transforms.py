import torch
import random
import functools
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from skimage.morphology import label

import src.data.transforms


def compose(transforms=None):
    """Compose several transforms together.
    Args:
        transforms (Box): The preprocessing and augmentation techniques applied to the data (default: None, only contain the default transform ToTensor).

    Returns:
        transforms (Compose): The list of BaseTransform.
    """
    if transforms is None:
        return Compose([ToTensor()])

    _transforms = []
    for transform in transforms:
        cls = getattr(src.data.transforms, transform.name)
        kwargs = transform.get('kwargs')
        _transforms.append(cls(**kwargs) if kwargs else cls())

    transforms = Compose(_transforms)
    return transforms


class BaseTransform:
    """The base class for all transforms.
    """
    def __call__(self, *imgs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class Compose(BaseTransform):
    """Compose several transforms together.
    Args:
         transforms (Box): The preprocessing and augmentation techniques applied to the data.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transformed.

        Returns:
            imgs (tuple of torch.Tensor): The transformed images.
        """
        for transform in self.transforms:
            imgs = transform(*imgs, **kwargs)

        # Returns the torch.Tensor instead of a tuple of torch.Tensor if there is only one image.
        if len(imgs) == 1:
            imgs = imgs[0]
        return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(BaseTransform):
    """Convert a tuple of numpy.ndarray to a tuple of torch.Tensor.
    """
    def __call__(self, *imgs, dtypes=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be converted to tensor.
            dtypes (sequence of torch.dtype, optional): The corresponding dtype of the images (default: None, transform all the images' dtype to torch.float).

        Returns:
            imgs (tuple of torch.Tensor): The converted images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if dtypes:
            if not all(isinstance(dtype, torch.dtype) for dtype in dtypes):
                raise TypeError('All of the dtypes should be torch.dtype.')
            if len(dtypes) != len(imgs):
                raise ValueError('The number of the dtypes should be the same as the images.')
            imgs = tuple(img.to(dtype) for img, dtype in zip(map(torch.from_numpy, imgs), dtypes))
            #imgs = tuple(img.copy().to(dtype) for img, dtype in zip(map(torch.from_numpy, imgs), dtypes))
        else:
            imgs = tuple(img.float() for img in map(torch.from_numpy, imgs))
        return imgs


class Normalize(BaseTransform):
    """Normalize a tuple of images with the means and the standard deviations.
    Args:
        means (list, optional): A sequence of means for each channel (default: None).
        stds (list, optional): A sequence of standard deviations for each channel (default: None).
    """
    def __init__(self, means=None, stds=None):
        if means is None and stds is None:
            pass
        elif means is not None and stds is not None:
            if len(means) != len(stds):
                raise ValueError('The number of the means should be the same as the standard deviations.')
        else:
            raise ValueError('Both the means and the standard deviations should have values or None.')

        self.means = means
        self.stds = stds

    def __call__(self, *imgs, normalize_tags=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be normalized.
            normalize_tags (sequence of bool, optional): The corresponding tags of the images (default: None, normalize all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The normalized images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if normalize_tags:
            if len(normalize_tags) != len(imgs):
                raise ValueError('The number of the tags should be the same as the images.')
            if not all(normalize_tag in [True, False] for normalize_tag in normalize_tags):
                raise ValueError("All of the tags should be either True or False.")
        else:
            normalize_tags = [None] * len(imgs)

        _imgs = []
        for img, normalize_tag in zip(imgs, normalize_tags):
            if normalize_tag is None or normalize_tag is True:
                if self.means is None and self.stds is None: # Apply image-level normalization.
                    axis = tuple(range(img.ndim - 1))
                    means = img.mean(axis=axis)
                    stds = img.std(axis=axis)
                    img = self._normalize(img, means, stds)
                else:
                    img = self._normalize(img, self.means, self.stds)
            elif normalize_tag is False:
                pass
            _imgs.append(img)
        imgs = tuple(_imgs)
        return imgs

    @staticmethod
    def _normalize(img, means, stds):
        """Normalize the image with the means and the standard deviations.
        Args:
            img (numpy.ndarray): The image to be normalized.
            means (list): A sequence of means for each channel.
            stds (list): A sequence of standard deviations for each channel.

        Returns:
            img (numpy.ndarray): The normalized image.
        """
        img = img.copy()
        for c, mean, std in zip(range(img.shape[-1]), means, stds):
            img[..., c] = (img[..., c] - mean) / (std + 1e-10)
        return img


class PositiveCrop(BaseTransform):
    """Set a probability (positive_sampling_rate) where we ensure the sampled images at least put a target pixel in the middle during training.
    Args:
        positive_sampling_rate (float): The probability to select the sample around the target pixel.
        size (list): The desired output size of the cropped images.
    """
    def __init__(self, positive_sampling_rate, size):
        self.positive_sampling_rate = positive_sampling_rate
        self.size = size

    def __call__(self, *imgs, target=None, **kwargs):
        """
        Args:
            target (numpy.ndarray): The reference target to determine sampling area.
            imgs (tuple of numpy.ndarray): The images to be cropped.

        Returns:
            imgs (tuple of numpy.ndarray): The cropped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if not all(target.shape == img.shape for img in imgs):
            raise ValueError("All of the images' shapes should be same as the target image.")
            
        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
            raise ValueError(f'The dimensions of the cropped size should be the same as the image ({ndim - 1}). Got {len(self.size)}')
        
        starts, ends = self._get_coordinates(target, self.positive_sampling_rate, self.size)
        if ndim == 3:
            imgs = tuple([img[starts[0]:ends[0]+1, starts[1]:ends[1]+1] for img in imgs])
        elif ndim == 4:
            imgs = tuple([img[starts[0]:ends[0]+1, starts[1]:ends[1]+1, starts[2]:ends[2]+1] for img in imgs])
        return imgs

    @staticmethod
    def _get_coordinates(target, positive_sampling_rate, size):
        """Compute the coordinates of the cropped image.
        Args:
            target (numpy.ndarray): The referenced image.
            size (list): The desired output size of the cropped image.

        Returns:
            coordinates (tuple): The coordinates of the cropped image.
        """
        if any(i - j < 0 for i, j in zip(target.shape, size)):
            raise ValueError(f'The target image ({target.shape}) is smaller than the cropped size ({size}). Please use a smaller cropped size.')

        sample_rate = random.uniform(0, 1)
        starts, ends = [], []
        if sample_rate <= positive_sampling_rate:
            # sample a target object
            label_target = label(target.squeeze(axis=-1), connectivity=target.ndim-1)
            target_list = np.unique(label_target)[1:]
            target_id = random.choice(target_list)
            
            # sample a pixel from the selected target
            positive_list = np.where(label_target == target_id)
            positive_index = random.choice(range(len(positive_list[0])))
            for i in range(target.ndim-1):
                start = positive_list[i][positive_index] - size[i] // 2
                start = max(0, start)
                end = start + size[i] - 1
                if end >= target.shape[i]:
                    end = target.shape[i] - 1
                    start = end - size[i] + 1
                starts.append(start)
                ends.append(end)
        else:
            for i in range(target.ndim-1):
                start = random.randint(0, target.shape[i] - size[i])
                end = start + size[i] - 1
                starts.append(start)
                ends.append(end)
        return starts, ends

class Resize(BaseTransform):
    """Resize a tuple of images to the same size.
    Args:
        size (list): The desired output size of the resized images.
    """
    def __init__(self, size):
        self.size = size
        self._resize = functools.partial(resize, mode='constant', preserve_range=True)

    def __call__(self, *imgs, resize_orders=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be resized.
            resize_orders (sequence of int, optional): The corresponding interpolation order of the images (default: None, the interpolation order would be 1 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The resized images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
            raise ValueError(f'The dimensions of the resized size should be the same as the image ({ndim - 1}). Got {len(self.size)}')

        if resize_orders:
            imgs = tuple(self._resize(img, self.size, order).astype(img.dtype) for img, order in zip(imgs, resize_orders))
        else:
            imgs = tuple(self._resize(img, self.size) for img in imgs)
        return imgs


class RandomCrop(BaseTransform):
    """Crop a tuple of images at the same random location.
    Args:
        size (list): The desired output size of the cropped images.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be cropped.

        Returns:
            imgs (tuple of numpy.ndarray): The cropped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        #print(imgs[0].shape)
        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
        #if ndim != len(self.size):
            raise ValueError(f'The dimensions of the cropped size should be the same as the image ({ndim - 1}). Got {len(self.size)}')

        if ndim == 3:
            h0, hn, w0, wn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple([img[h0:hn, w0:wn].copy() for img in imgs])
        elif ndim == 4:
            h0, hn, w0, wn, d0, dn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple([img[h0:hn, w0:wn, d0:dn].copy() for img in imgs])
        return imgs

    @staticmethod
    def _get_coordinates(img, size):
        """Compute the coordinates of the cropped image.
        Args:
            img (numpy.ndarray): The image to be cropped.
            size (list): The desired output size of the cropped image.

        Returns:
            coordinates (tuple): The coordinates of the cropped image.
        """
        if any(i - j < 0 for i, j in zip(img.shape, size)):
            raise ValueError(f'The image ({img.shape}) is smaller than the cropped size ({size}). Please use a smaller cropped size.')

        if img.ndim == 3:
        #if img.ndim == 2:
            h, w = img.shape[:-1]
            ht, wt = size
            h0, w0 = random.randint(0, h - ht), random.randint(0, w - wt)
            return h0, h0 + ht, w0, w0 + wt
        elif img.ndim == 4:
        #elif img.ndim == 3:
            h, w, d = img.shape[:-1]
            ht, wt, dt = size
            h0, w0, d0 = random.randint(0, h - ht), random.randint(0, w - wt), random.randint(0, d - dt)
            return h0, h0 + ht, w0, w0 + wt, d0, d0 + dt


class RandomElasticDeformation(BaseTransform):
    """Do the random elastic deformation as used in U-Net and V-Net by using the bspline transform.
    Args:
        do_z_deformation (bool, optional): Whether to apply the deformation along the z dimension (default: False).
        num_ctrl_points (int, optional): The number of the control points to form the control point grid (default: 4).
        sigma (int or float, optional): The number to determine the extent of deformation (default: 15).
        prob (float, optional): The probability of applying the deformation (default: 0.5).
    """
    def __init__(self, do_z_deformation=False, num_ctrl_points=4, sigma=15, prob=0.5):
        self.do_z_deformation = do_z_deformation
        self.num_ctrl_points = max(num_ctrl_points, 2)
        self.sigma = max(sigma, 1)
        self.prob = max(0, min(prob, 1))
        self.bspline_transform = None

    def __call__(self, *imgs, elastic_deformation_orders=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be deformed.
            elastic_deformation_orders (sequence of int, optional): The corresponding interpolation order of the images (default: None, the interpolation order would be 3 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The deformed images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            self._init_bspline_transform(imgs[0].shape)
            if elastic_deformation_orders:
                imgs = tuple(self._apply_bspline_transform(img, order) for img, order in zip(imgs, elastic_deformation_orders))
            else:
                imgs = map(self._apply_bspline_transform, imgs)
        return imgs

    def _init_bspline_transform(self, shape):
        """Initialize the bspline transform.
        Args:
            shape (tuple): The size of the control point grid.
        """
        # Remove the channel dimension.
        shape = shape[:-1]

        # Initialize the control point grid.
        img = sitk.GetImageFromArray(np.zeros(shape))
        mesh_size = [self.num_ctrl_points] * img.GetDimension()
        self.bspline_transform = sitk.BSplineTransformInitializer(img, mesh_size)

        # Set the parameters of the bspline transform randomly.
        params = self.bspline_transform.GetParameters()
        params = np.asarray(params, dtype=np.float64)
        params = params + np.random.randn(params.shape[0]) * self.sigma
        if len(shape) == 3 and not self.do_z_deformation:
            params[0: len(params) // 3] = 0
        params = tuple(params)
        self.bspline_transform.SetParameters(params)

    def _apply_bspline_transform(self, img, order=3):
        """Apply the bspline transform.
        Args:
            img (np.ndarray): The image to be deformed.
            order (int, optional): The interpolation order (default: 3, should be 0, 1 or 3).

        Returns:
            img (np.ndarray): The deformed image.
        """
        # Create the resampler.
        resampler = sitk.ResampleImageFilter()
        if order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif order == 3:
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            raise ValueError(f'The interpolation order should be 0, 1 or 3. Got {order}.')

        # Apply the bspline transform.
        shape = img.shape
        img = sitk.GetImageFromArray(np.squeeze(img))
        resampler.SetReferenceImage(img)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(self.bspline_transform)
        img = resampler.Execute(img)
        img = sitk.GetArrayFromImage(img).reshape(shape)
        return img


class RandomHorizontalFlip(BaseTransform):
    """Do the random flip horizontally.
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be flipped.

        Returns:
            imgs (tuple of numpy.ndarray): The flipped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            imgs = tuple([np.flip(img, 1) for img in imgs])
        return imgs


class RandomVerticalFlip(BaseTransform):
    """Do the random flip vertically.
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be flipped.

        Returns:
            imgs (tuple of numpy.ndarray): The flipped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            imgs = tuple([np.flip(img, 0) for img in imgs])
        return imgs


class RandomCropPatch(BaseTransform):
    """Crop a tuple of LR images at the same random location and a tuple of HR images at the corresponding location.

    Note that it expects the first half of the images are LR, and the remaining images are HR.

    Args:
        size (list): The desired output size of the cropped LR images.
        ratio (int): The ratio between the HR images and the LR images.
    """
    def __init__(self, size, ratio):
        self.size = size
        self.ratio = ratio

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be cropped.

        Returns:
            imgs (tuple of numpy.ndarray): The cropped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
            raise ValueError(f'The dimensions of the cropped size should be the same as the image ({ndim - 1}). Got {len(self.size)}')

        if len(imgs) % 2 == 1:
            raise ValueError(f'The number of the LR images should be the same as the HR images')

        lr_imgs, hr_imgs = imgs[:len(imgs) // 2], imgs[len(imgs) // 2:]
        if not all(j // i == self.ratio for lr_img, hr_img in zip(lr_imgs, hr_imgs) for i, j in zip(lr_img.shape[:-1], hr_img.shape[:-1])):
            raise ValueError(f'The ratio between the HR images and the LR images should be {self.ratio}.')

        if ndim == 3:
            lr_h0, lr_hn, lr_w0, lr_wn = self._get_coordinates(lr_imgs[0], self.size)
            hr_h0, hr_hn, hr_w0, hr_wn = lr_h0 * self.ratio, lr_hn * self.ratio, \
                                         lr_w0 * self.ratio, lr_wn * self.ratio
            imgs = tuple([lr_img[lr_h0:lr_hn, lr_w0:lr_wn] for lr_img in lr_imgs] + \
                         [hr_img[hr_h0:hr_hn, hr_w0:hr_wn] for hr_img in hr_imgs])
        elif ndim == 4:
            lr_h0, lr_hn, lr_w0, lr_wn, lr_d0, lr_dn = self._get_coordinates(lr_imgs[0], self.size)
            hr_h0, hr_hn, hr_w0, hr_wn, hr_d0, hr_dn = lr_h0 * self.ratio, lr_hn * self.ratio, \
                                                       lr_w0 * self.ratio, lr_wn * self.ratio, \
                                                       lr_d0 * self.ratio, lr_dn * self.ratio
            imgs = tuple([lr_img[lr_h0:lr_hn, lr_w0:lr_wn, lr_d0:lr_dn] for lr_img in lr_imgs] + \
                         [hr_img[hr_h0:hr_hn, hr_w0:hr_wn, hr_d0:hr_dn] for hr_img in hr_imgs])
        return imgs

    @staticmethod
    def _get_coordinates(img, size):
        """Compute the coordinates of the cropped image.
        Args:
            img (numpy.ndarray): The image to be cropped.
            size (list): The desired output size of the cropped image.

        Returns:
            coordinates (tuple): The coordinates of the cropped image.
        """
        if any(i - j < 0 for i, j in zip(img.shape, size)):
            raise ValueError(f'The image ({img.shape}) is smaller than the cropped size ({size}). Please use a smaller cropped size.')

        if img.ndim == 3:
            h, w = img.shape[:-1]
            ht, wt = size
            h0, w0 = random.randint(0, h - ht), random.randint(0, w - wt)
            return h0, h0 + ht, w0, w0 + wt
        elif img.ndim == 4:
            h, w, d = img.shape[:-1]
            ht, wt, dt = size
            h0, w0, d0 = random.randint(0, h - ht), random.randint(0, w - wt), random.randint(0, d - dt)
            return h0, h0 + ht, w0, w0 + wt, d0, d0 + dt
