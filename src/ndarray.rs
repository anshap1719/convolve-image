#![cfg(feature = "ndarray")]

use ndarray::{Array1, Array2, Array3};
use crate::Convolution;
use crate::kernel::{Kernel, NonSeparableKernel, SeparableKernel};
use crate::dimensions::DimensionIterator;

impl Convolution for Array1<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: impl Kernel<KERNEL_SIZE, Self>, stride: usize) {
        let linear_kernel = kernel.values();

        let sample_length = self.len();
        
        for index in 0..sample_length {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    index,
                    sample_length
                );

                pixel_sum += self[pixel_index as usize] * *value;
            }

            self[index] = pixel_sum;
        }
    }
}

impl Convolution for Array2<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: impl Kernel<KERNEL_SIZE, Self>, stride: usize) {
        let linear_kernel = kernel.values();

        let (height, width) = self.dim();
        let dimensions = self.raw_dim();

        for (y, x) in dimensions.into_iter() {
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

                pixel_sum += self[[y, pixel_index as usize]] * *value;
            }

            self[[y, x]] = pixel_sum;
        }

        for (y, x) in dimensions.into_iter() {
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

                pixel_sum += self[[pixel_index as usize, x]] * *value;
            }

            self[[y, x]] = pixel_sum;
        }
    }
}

impl Convolution for Array3<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: impl Kernel<KERNEL_SIZE, Self>, stride: usize) {
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

                pixel_sum += self[[y, pixel_index as usize, channel]] * *value;
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

                pixel_sum += self[[pixel_index as usize, x, channel]] * *value;
            }

            self[[y, x, channel]] = pixel_sum;
        }
    }
}

pub(crate) trait Aggregate {
    fn min(&self) -> f32;
    fn max(&self) -> f32;
}

impl Aggregate for Array2<f32> {
    fn min(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current < previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }

    fn max(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current > previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }
}

impl Aggregate for Array3<f32> {
    fn min(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current < previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }

    fn max(&self) -> f32 {
        *self
            .iter()
            .reduce(|current, previous| {
                if current > previous {
                    current
                } else {
                    previous
                }
            })
            .unwrap()
    }
}

impl<const SIZE: usize> Kernel<SIZE, Array2<f32>> for NonSeparableKernel<SIZE> {
    fn apply(&self, values: &mut Array2<f32>, stride: usize) {
        let stride = stride as isize;

        // The kernel's input positions relative to the current pixel.
        let taps: &[(isize, isize)] = &[
            (-1 * stride, -1 * stride),
            (0, -1 * stride),
            (1 * stride, -1 * stride),
            (-1 * stride, 0),
            (0, 0),
            (1 * stride, 0),
            (-1 * stride, 1 * stride),
            (0, 1 * stride),
            (1 * stride, 1 * stride),
        ];

        let (height, width) = values.dim();

        let sum = match self.values().iter().flatten().fold(0.0, |sum, &item| sum + item) {
            x if x == 0.0 => 1.0,
            sum => sum,
        };

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut total = 0.0;

                for (&kernel, &(x_index, y_index)) in self.values().iter().flatten().zip(taps.iter()) {
                    let x0 = x as isize + x_index;
                    let y0 = y as isize + y_index;

                    let pixel = values[[y0 as usize, x0 as usize]];

                    total += pixel * kernel;
                }

                let total = total / sum;

                values[[y, x]] = total;
            }
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().iter().flatten().copied().collect()
    }
}

impl<const SIZE: usize> Kernel<SIZE, Array3<f32>> for NonSeparableKernel<SIZE> {
    fn apply(&self, values: &mut Array3<f32>, stride: usize) {
        let stride = stride as isize;

        // The kernel's input positions relative to the current pixel.
        let taps: &[(isize, isize)] = &[
            (-1 * stride, -1 * stride),
            (0, -1 * stride),
            (1 * stride, -1 * stride),
            (-1 * stride, 0),
            (0, 0),
            (1 * stride, 0),
            (-1 * stride, 1 * stride),
            (0, 1 * stride),
            (1 * stride, 1 * stride),
        ];

        let (height, width, _) = values.dim();

        let sum = match self.values().iter().flatten().fold(0.0, |s, &item| s + item) {
            0.0 => 1.0,
            sum => sum,
        };

        let sum = (sum, sum, sum);

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut total = (0.0, 0.0, 0.0);

                for (&kernal, &(x_index, y_index)) in self.values().iter().flatten().zip(taps.iter()) {
                    let kernal = (kernal, kernal, kernal);
                    let x0 = x as isize + x_index;
                    let y0 = y as isize + y_index;

                    let pixel = (values[[y0 as usize, x0 as usize, 0]], values[[y0 as usize, x0 as usize, 1]], values[[y0 as usize, x0 as usize, 2]]);

                    #[allow(deprecated)]
                    let (r, g, b) = pixel;

                    total.0 += r * kernal.0;
                    total.1 += g * kernal.1;
                    total.2 += b * kernal.2;
                }

                values[[y, x, 0]] = total.0 / sum.0;
                values[[y, x, 1]] = total.1 / sum.1;
                values[[y, x, 2]] = total.2 / sum.2;
            }
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().iter().flatten().copied().collect()
    }
}

impl<const KERNEL_SIZE: usize> Kernel<KERNEL_SIZE, Array2<f32>> for SeparableKernel<KERNEL_SIZE> {
    fn apply(&self, values: &mut Array2<f32>, stride: usize) {
        let linear_kernel = self.values();

        let (height, width) = values.dim();
        let dimensions = values.raw_dim();

        for (y, x) in dimensions.into_iter() {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Array2::<f32>::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x,
                    width
                );

                pixel_sum += values[[y, pixel_index as usize]] * *value;
            }

            values[[y, x]] = pixel_sum;
        }

        for (y, x) in dimensions.into_iter() {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Array2::<f32>::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y,
                    height
                );

                pixel_sum += values[[pixel_index as usize, x]] * *value;
            }

            values[[y, x]] = pixel_sum;
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().to_vec()
    }
}

impl<const KERNEL_SIZE: usize> Kernel<KERNEL_SIZE, Array3<f32>> for SeparableKernel<KERNEL_SIZE> {
    fn apply(&self, values: &mut Array3<f32>, stride: usize) {
        let linear_kernel = self.values();
        let (height, width, _) = values.dim();
        let dimensions = values.raw_dim();

        for (y, x, channel) in dimensions.into_iter() {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Array3::<f32>::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x,
                    width
                );

                pixel_sum += values[[y, pixel_index as usize, channel]] * *value;
            }

            values[[y, x, channel]] = pixel_sum;
        }

        for (y, x, channel) in dimensions.into_iter() {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Array3::<f32>::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y,
                    height
                );

                pixel_sum += values[[pixel_index as usize, x, channel]] * *value;
            }

            values[[y, x, channel]] = pixel_sum;
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().to_vec()
    }
}