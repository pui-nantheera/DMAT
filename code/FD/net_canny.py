import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, k_gaussian=3, k_sobel=3, mu=0, sigma=3, requires_grad=True):
        super(Net, self).__init__()
        # Gaussian filter
        gaussian_2D = self.get_gaussian_kernel(k_gaussian, mu, sigma)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_gaussian,
            padding=k_gaussian // 2,
            padding_mode="replicate",
            bias=False,
        )
        gaussian_2D = torch.from_numpy(gaussian_2D)
        gaussian_2D.requires_grad = requires_grad
        with torch.no_grad():
            self.conv1.weight[:] = gaussian_2D

        # Sobel filter x direction
        sobel_2D = self.get_sobel_kernel(k_sobel)
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_sobel,
            padding=k_sobel // 2,
            padding_mode="replicate",
            bias=False,
        )
        sobel_2D = torch.from_numpy(sobel_2D)
        sobel_2D.requires_grad = requires_grad
        with torch.no_grad():
            self.conv2.weight[:] = sobel_2D

        # Sobel filter y direction
        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_sobel,
            padding=k_sobel // 2,
            padding_mode="replicate",
            bias=False,
        )
        with torch.no_grad():
            self.conv3.weight[:] = sobel_2D.T

        # Hysteresis custom kernel
        self.conv4 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            bias=False,
        ).cuda()
        hyst_kernel = np.ones((3, 3)) + 0.25
        hyst_kernel = torch.from_numpy(hyst_kernel).unsqueeze(0).unsqueeze(0)
        hyst_kernel.requires_grad = False
        with torch.no_grad():
            self.conv4.weight = nn.Parameter(hyst_kernel)

        # Threshold parameters

        self.lowThreshold = torch.nn.Parameter(
            torch.tensor(0.10), requires_grad=requires_grad
        )
        self.highThreshold = torch.nn.Parameter(
            torch.tensor(0.20), requires_grad=requires_grad
        )

    def get_gaussian_kernel(self, k=3, mu=0, sigma=1, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x**2 + y**2) ** 0.5
        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-((distance - mu) ** 2) / (2 * sigma**2))
        gaussian_2D = gaussian_2D / (2 * np.pi * sigma**2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        return gaussian_2D

    def get_sobel_kernel(self, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = x**2 + y**2
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def set_local_maxima(
        self, magnitude, pts, w_num, w_denum, row_slices, col_slices, out
    ):
        """Get the magnitudes shifted left to make a matrix of the points to
        the right of pts. Similarly, shift left and down to get the points
        to the top right of pts."""
        pts = pts.cuda()
        out = out.cuda()
        r_0, r_1, r_2, r_3 = row_slices
        c_0, c_1, c_2, c_3 = col_slices
        c1 = magnitude[:, 0, r_0, c_0][pts[:, 0, r_1, c_1]]
        c2 = magnitude[:, 0, r_2, c_2][pts[:, 0, r_3, c_3]]
        m = magnitude[pts]
        w = w_num[pts] / w_denum[pts]
        c_plus = c2 * w + c1 * (1 - w) <= m
        c_plus = c_plus.cuda()
        c1 = magnitude[:, 0, r_1, c_1][pts[:, 0, r_0, c_0]]
        c2 = magnitude[:, 0, r_3, c_3][pts[:, 0, r_2, c_2]]
        c_minus = c2 * w + c1 * (1 - w) <= m
        c_minus = c_minus.cuda()
        out[pts] = c_plus & c_minus

        return out

    def get_local_maxima(self, isobel, jsobel, magnitude, eroded_mask):
        """Edge thinning by non-maximum suppression."""

        abs_isobel = torch.abs(jsobel)
        abs_jsobel = torch.abs(isobel)

        eroded_mask = eroded_mask & (magnitude > 0)

        # Normals' orientations
        is_horizontal = eroded_mask & (abs_isobel >= abs_jsobel)
        is_vertical = eroded_mask & (abs_isobel <= abs_jsobel)
        is_up = isobel >= 0
        is_down = isobel <= 0
        is_right = jsobel >= 0
        is_left = jsobel <= 0
        #
        # --------- Find local maxima --------------
        #
        # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
        # 90-135 degrees and 135-180 degrees.
        #
        local_maxima = torch.zeros(magnitude.shape, dtype=bool)
        # ----- 0 to 45 degrees ------
        # Mix diagonal and horizontal
        pts_plus = is_up & is_right
        pts_minus = is_down & is_left
        pts = (pts_plus | pts_minus) & is_horizontal
        # Get the magnitudes shifted left to make a matrix of the points to the
        # right of pts. Similarly, shift left and down to get the points to the
        # top right of pts.
        local_maxima = self.set_local_maxima(
            magnitude,
            pts,
            abs_jsobel,
            abs_isobel,
            [slice(1, None), slice(-1), slice(1, None), slice(-1)],
            [slice(None), slice(None), slice(1, None), slice(-1)],
            local_maxima,
        )
        # ----- 45 to 90 degrees ------
        # Mix diagonal and vertical
        #
        pts = (pts_plus | pts_minus) & is_vertical
        local_maxima = self.set_local_maxima(
            magnitude,
            pts,
            abs_isobel,
            abs_jsobel,
            [slice(None), slice(None), slice(1, None), slice(-1)],
            [slice(1, None), slice(-1), slice(1, None), slice(-1)],
            local_maxima,
        )
        # ----- 90 to 135 degrees ------
        # Mix anti-diagonal and vertical
        #
        pts_plus = is_down & is_right
        pts_minus = is_up & is_left
        pts = (pts_plus | pts_minus) & is_vertical
        local_maxima = self.set_local_maxima(
            magnitude,
            pts,
            abs_isobel,
            abs_jsobel,
            [slice(None), slice(None), slice(-1), slice(1, None)],
            [slice(1, None), slice(-1), slice(1, None), slice(-1)],
            local_maxima,
        )
        # ----- 135 to 180 degrees ------
        # Mix anti-diagonal and anti-horizontal
        #
        pts = (pts_plus | pts_minus) & is_horizontal
        local_maxima = self.set_local_maxima(
            magnitude,
            pts,
            abs_jsobel,
            abs_isobel,
            [slice(-1), slice(1, None), slice(-1), slice(1, None)],
            [slice(None), slice(None), slice(1, None), slice(-1)],
            local_maxima,
        )

        return local_maxima

    def threshold(self, img):
        """Thresholds for defining weak and strong edge pixels"""

        alpha = 100000
        weak = 0.5
        strong = 1
        res_strong = strong * (alpha * (img - self.highThreshold)).sigmoid()
        res_weak_1 = weak * (alpha * (self.highThreshold - img)).sigmoid()
        res_weak_2 = weak * (alpha * (self.lowThreshold - img)).sigmoid()
        res_weak = res_weak_1 - res_weak_2
        res = res_weak + res_strong

        return res

    def hysteresis(self, img):

        # Create image that has strong pixels remain at one, weak pixels become zero
        img_strong = img.clone()
        img_strong[img == 0.5] = 0

        # Create masked image that turns all weak pixel into ones, rest to zeros
        masked_img = img.clone()
        masked_img[torch.logical_not(img == 0.5)] = 0
        masked_img[img == 0.5] = 1

        # Calculate weak edges that are changed to strong edges
        changed_edges = img.clone()
        changed_edges[((self.conv4(img_strong) > 1) & (masked_img == 1))] = 1

        # Add changed edges to already good edges
        changed_edges[changed_edges != 1] = 0

        # Add changed edges to already good edges
        return changed_edges

    def forward(self, x):

        # Gaussian filter
        x = self.conv1(x)

        # Sobel filter
        sobel_x = self.conv2(x)
        sobel_y = self.conv3(x)

        # Magnitude and angles
        eps = 1e-10
        self.grad_magnitude = torch.hypot(sobel_x + eps, sobel_y + eps)

        # Non-max-suppression
        eroded_mask = torch.ones(x.shape, dtype=bool).cuda()
        eroded_mask[:, 0, :1, :] = 0
        eroded_mask[:, 0, -1:, :] = 0
        eroded_mask[:, 0, :, :1] = 0
        eroded_mask[:, 0, :, -1:] = 0
        thin_edges = self.get_local_maxima(
            sobel_x, sobel_y, self.grad_magnitude, eroded_mask
        )
        thin_edges = self.grad_magnitude * (thin_edges * 1)

        # Double threshold
        thin_edges = thin_edges / torch.max(thin_edges)
        thresh = self.threshold(thin_edges)

        # Hysteresis (Currently not working)
        # result = self.hysteresis(thresh)

        # return result, self.grad_magnitude, thin_edges, thresh
        return self.grad_magnitude, thin_edges, thresh
