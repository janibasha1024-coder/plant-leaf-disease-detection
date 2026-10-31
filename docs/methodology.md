# Methodology

The application uses classical image processing. It does not train a classifier and it does not name a disease.

Color shares are counted on the resized original photograph. Noise reduction is used to find the leaf outline. Contrast adjustment is shown as its own step and is not used in the percentage, so the enhancement does not change the reported area.

## 1. File check

The file extension must be `.jpg`, `.jpeg`, or `.png`. The file must be 8 MB or smaller, and both sides must be at least 64 pixels. The bytes are decoded and must be a JPEG or PNG. Other types, empty files, and files that cannot be decoded are rejected.

## 2. Resize

If the longest side is above 900 pixels, the photograph is scaled down with area interpolation. Smaller photographs are left at their original size. Later steps use this resized copy.

## 3. Noise reduction

A bilateral filter (diameter 9, color sigma 75, space sigma 75) smooths speckle and keeps stronger edges. The leaf outline is taken from this denoised image.

## 4. Contrast adjustment

Contrast-limited adaptive histogram equalization is applied to the lightness channel in CIE Lab (clip limit 2.0, 8×8 tiles). This image is displayed so the enhancement can be inspected. It is not the image used for the area count.

## 5. Leaf segmentation

Two masks are combined:

- Excess green, `2G − R − B`, thresholded with Otsu’s method
- An HSV range for green (hue 25–95) and a narrower range for yellow-green tissue (hue 12–35)

The combined mask is closed and then opened with a 7×7 elliptical element. The largest outer contour with an area of at least 800 pixels is filled, which includes spots that sit inside the outline. The region is kept only when it covers at least 5% of the image. If several regions pass the area test, the largest one is measured and the result says so.

## 6. Color groups

Inside the leaf, each pixel is placed in one group. OpenCV’s CIE Lab encoding is used for the color tests. In that encoding the a\* and b\* channels are shifted by 128.

- Dark spots: the lightness is at least 32 levels below a local average (Gaussian sigma 9). Pixels within 4 pixels of the outline are ignored, and a spot must contain at least 12 pixels. This is aimed at compact marks such as specks. A broad, even patch can be missed when its center is no darker than the tissue around it.
- Yellow: b\* at least 158, a\* from 118 up to but not including 150, and lightness at least 55
- Brown: a\* at least 145, and not already labeled dark or yellow
- Green reference: remaining pixels with a\* at most 136
- Other: any leaf pixel left over

The affected area is the yellow, brown, dark, and other pixels divided by the leaf pixel count. Hue variation is the standard deviation of the OpenCV hue inside the leaf. Edge density is the share of those pixels that a Canny edge detector marks.

When the green region covers more than 90% of the photograph, the result says so. That usually means the picture is filled with foliage rather than one leaf on a plain background, and shade can be counted with the spots.

## 7. Sharpness and lighting

Sharpness is the variance of the Laplacian on the resized original. Values below 60 add a warning that the photograph is soft. Lighting is the mean brightness of the leaf. A mean below 55 is described as dark, and a mean above 215 as very bright.

## What the percentage is not

The percentage is a color comparison. It is not a lesion count, not a species identification, and not a disease diagnosis. Veins, shadows, holes, and naturally red or purple leaves can raise it. Faint spots can be missed. No accuracy figure is reported because this recreation does not include a labeled test set or a classifier from the 2022 project.
