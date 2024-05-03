#![cfg(feature = "image")]

use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use crate::Convolution;
use crate::iter::ImageIterator;
use crate::kernel::SeparableKernel;

impl Convolution for ImageBuffer<Luma<f32>, Vec<f32>> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();
        
        for (x, y) in ImageIterator::new(self.width(), self.height()) {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x as usize,
                    self.width() as usize
                );

                pixel_sum += self.get_pixel(pixel_index, y).0[0] * value;
            }

            self.put_pixel(x, y, Luma([pixel_sum]));   
        }

        for (x, y) in ImageIterator::new(self.width(), self.height()) {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y as usize,
                    self.height() as usize
                );

                pixel_sum += self.get_pixel(x, pixel_index).0[0] * value;
            }

            self.put_pixel(x, y, Luma([pixel_sum]));
        }
    }
}

impl Convolution for ImageBuffer<Rgb<f32>, Vec<f32>> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();

        for (x, y) in ImageIterator::new(self.width(), self.height()) {
            let mut pixel_sum = [0., 0., 0.];

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x as usize,
                    self.width() as usize
                );

                let mut computed_values = self.get_pixel(pixel_index, y).0;
                computed_values = [computed_values[0] * value, computed_values[1] * value, computed_values[2] * value];

                pixel_sum = [pixel_sum[0] + computed_values[0], pixel_sum[1] + computed_values[1], pixel_sum[2] + computed_values[2]];
            }

            let [r, g, b] = pixel_sum;
            self.put_pixel(x, y, Rgb([r, g, b]));   
        }

        for (x, y) in ImageIterator::new(self.width(), self.height()) {
            let mut pixel_sum = [0., 0., 0.];

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y as usize,
                    self.height() as usize
                );

                let mut computed_values = self.get_pixel(x, pixel_index).0;
                computed_values = [computed_values[0] * value, computed_values[1] * value, computed_values[2] * value];

                pixel_sum = [pixel_sum[0] + computed_values[0], pixel_sum[1] + computed_values[1], pixel_sum[2] + computed_values[2]];
            }

            let [r, g, b] = pixel_sum;
            self.put_pixel(x, y, Rgb([r, g, b]));   
        }
    }
}

impl Convolution for DynamicImage {
    fn convolve<const KERNEL_SIZE: usize>(
        &mut self,
        kernel: SeparableKernel<KERNEL_SIZE>,
        stride: usize,
    ) {
        match self {
            DynamicImage::ImageLuma8(_) |
            DynamicImage::ImageLumaA8(_) |
            DynamicImage::ImageLuma16(_) |
            DynamicImage::ImageLumaA16(_) => {
                let mut image = self.to_luma32f();
                image.convolve(kernel, stride);

                let mut result_img: ImageBuffer<Luma<u16>, Vec<u16>> =
                    ImageBuffer::new(self.width(), self.height());

                for (x, y, pixel) in result_img.enumerate_pixels_mut() {
                    *pixel =
                        Luma([(image.get_pixel(x, y).0[0] * u16::MAX as f32) as u16]);
                }

                *self = DynamicImage::ImageLuma16(result_img);
            }
            DynamicImage::ImageRgb8(_) |
            DynamicImage::ImageRgba8(_) |
            DynamicImage::ImageRgb16(_) |
            DynamicImage::ImageRgba16(_) |
            DynamicImage::ImageRgb32F(_) |
            DynamicImage::ImageRgba32F(_) => {
                let mut image = self.to_rgb32f();
                image.convolve(kernel, stride);

                *self = DynamicImage::ImageRgb32F(image);
            }
            _ => unimplemented!("Not implemented")
        }
    }
}