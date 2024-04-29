#![cfg(feature = "ndarray")]

use ndarray::{Array2, Array3};
use crate::Convolution;
use crate::kernel::SeparableKernel;
use crate::dimensions::DimensionIterator;

impl Convolution for Array2<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();

        let (height, width) = self.dim();

        for y in 0..height {
            for x in 0..width {
                let mut pixel_sum = 0.;

                for (kernel_index, value) in linear_kernel.iter().enumerate() {
                    let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                    let pixel_index = Self::compute_pixel_index(
                        stride,
                        KERNEL_SIZE,
                        relative_kernel_index,
                        x,
                        width
                    );

                    pixel_sum += self[[y, pixel_index as usize]] * value;
                }

                self[[y, x]] = pixel_sum;
            }
        }

        for x in 0..width {
            for y in 0..height {
                let mut pixel_sum = 0.;

                for (kernel_index, value) in linear_kernel.iter().enumerate() {
                    let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                    let pixel_index = Self::compute_pixel_index(
                        stride,
                        KERNEL_SIZE,
                        relative_kernel_index,
                        y,
                        width
                    );

                    pixel_sum += self[[pixel_index as usize, x]] * value;
                }

                self[[y, x]] = pixel_sum;
            }
        }
    }
}

impl Convolution for Array3<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();
        let (height, width, _) = self.dim();
        let dimensions = self.raw_dim();

        for (y, x, channel) in dimensions.into_iter() {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x,
                    width
                );

                pixel_sum += self[[pixel_index as usize, x, channel]] * *value;
            }

            self[[y, x, channel]] = pixel_sum;   
        }

        for (y, x, channel) in dimensions.into_iter() {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y,
                    height
                );

                pixel_sum += self[[y, pixel_index as usize, channel]] * *value;
            }

            self[[y, x, channel]] = pixel_sum;
        }
    }
}