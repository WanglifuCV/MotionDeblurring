# MotionDeblurring

Editor: Lifu Wang

This repository is devoted to restore Blind Motion Deblurring code implemented by Matlab in 2014~2015.

---

## Introduction
As we know, a motion blur is a common artifact that produces blurry images with information loss. If a motion blur is shift-invariant(which means it is a global motion), it can be modeled as tht convolution of a latent image with a motion blur kernel. Therefore, removing a motion blur is to recover the latent image from a blurry version using the kernel.

In Blind Motion Deblurring, the kernel is unknown, and therefore, in order to recovery the latent shape image, the kernel should be estimated.

My code is used to solve the single-image blind deconvlution problem, where both blur kernel and latent sharp image are estimated from just one blurred image.

I present a short introduction and some results in several slices.[Introduction & Results](./Introduction.pdf)
