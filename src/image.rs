#![cfg(feature = "image")]

use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use crate::Convolution;
use crate::iter::ImageIterator;
use crate::kernel::{Kernel, NonSeparableKernel, SeparableKernel};

impl Convolution for ImageBuffer<Luma<f32>, Vec<f32>> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: impl Kernel<KERNEL_SIZE, Self>, stride: usize) {
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
    fn convolve<const KERNEL_SIZE: usize>(&mut self, kernel: impl Kernel<KERNEL_SIZE, Self>, stride: usize) {
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
        kernel: impl Kernel<KERNEL_SIZE, Self>,
        stride: usize,
    ) {
        match self {
            DynamicImage::ImageLuma8(_) |
            DynamicImage::ImageLumaA8(_) |
            DynamicImage::ImageLuma16(_) |
            DynamicImage::ImageLumaA16(_) => {
                kernel.apply(self, stride);
            }
            DynamicImage::ImageRgb8(_) |
            DynamicImage::ImageRgba8(_) |
            DynamicImage::ImageRgb16(_) |
            DynamicImage::ImageRgba16(_) |
            DynamicImage::ImageRgb32F(_) |
            DynamicImage::ImageRgba32F(_) => {
                kernel.apply(self, stride);
            }
            _ => unimplemented!("Not implemented")
        }
    }
}

impl<const SIZE: usize> Kernel<SIZE, ImageBuffer<Luma<f32>, Vec<f32>>> for NonSeparableKernel<SIZE> {
    fn apply(&self, values: &mut ImageBuffer<Luma<f32>, Vec<f32>>, stride: usize) {
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

        let (width, height) = values.dimensions();

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

                    let [pixel] = values.get_pixel(x0 as u32, y0 as u32).0;

                    total += pixel * kernel;
                }

                let total = total / sum;

                values.put_pixel(x, y, Luma([total]));
            }
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().iter().flatten().copied().collect()
    }
}

impl<const SIZE: usize> Kernel<SIZE, ImageBuffer<Rgb<f32>, Vec<f32>>> for NonSeparableKernel<SIZE> {
    fn apply(&self, values: &mut ImageBuffer<Rgb<f32>, Vec<f32>>, stride: usize) {
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

        let (width, height) = values.dimensions();

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

                    let pixel = values.get_pixel(x0 as u32, y0 as u32);

                    #[allow(deprecated)]
                    let [r, g, b] = pixel.0;

                    total.0 += r * kernal.0;
                    total.1 += g * kernal.1;
                    total.2 += b * kernal.2;
                }

                values.put_pixel(x, y, Rgb([total.0 / sum.0, total.1 / sum.1, total.2 / sum.2]));
            }
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().iter().flatten().copied().collect()
    }
}

impl<const SIZE: usize> Kernel<SIZE, ImageBuffer<Luma<f32>, Vec<f32>>> for SeparableKernel<SIZE> {
    fn apply(&self, values: &mut ImageBuffer<Luma<f32>, Vec<f32>>, stride: usize) {
        let linear_kernel = self.values();

        for (x, y) in ImageIterator::new(values.width(), values.height()) {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (SIZE as isize / 2);
                let pixel_index = ImageBuffer::<Luma<f32>, Vec<f32>>::compute_pixel_index(
                    stride,
                    SIZE,
                    relative_kernel_index,
                    x as usize,
                    values.width() as usize
                );

                pixel_sum += values.get_pixel(pixel_index, y).0[0] * value;
            }

            values.put_pixel(x, y, Luma([pixel_sum]));
        }

        for (x, y) in ImageIterator::new(values.width(), values.height()) {
            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (SIZE as isize / 2);
                let pixel_index = ImageBuffer::<Luma<f32>, Vec<f32>>::compute_pixel_index(
                    stride,
                    SIZE,
                    relative_kernel_index,
                    y as usize,
                    values.height() as usize
                );

                pixel_sum += values.get_pixel(x, pixel_index).0[0] * value;
            }

            values.put_pixel(x, y, Luma([pixel_sum]));
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().to_vec()
    }
}

impl<const SIZE: usize> Kernel<SIZE, ImageBuffer<Rgb<f32>, Vec<f32>>> for SeparableKernel<SIZE> {
    fn apply(&self, values: &mut ImageBuffer<Rgb<f32>, Vec<f32>>, stride: usize) {
        let linear_kernel = self.values();

        for (x, y) in ImageIterator::new(values.width(), values.height()) {
            let mut pixel_sum = [0., 0., 0.];

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (SIZE as isize / 2);
                let pixel_index = ImageBuffer::<Rgb<f32>, Vec<f32>>::compute_pixel_index(
                    stride,
                    SIZE,
                    relative_kernel_index,
                    x as usize,
                    values.width() as usize
                );

                let mut computed_values = values.get_pixel(pixel_index, y).0;
                computed_values = [computed_values[0] * value, computed_values[1] * value, computed_values[2] * value];

                pixel_sum = [pixel_sum[0] + computed_values[0], pixel_sum[1] + computed_values[1], pixel_sum[2] + computed_values[2]];
            }

            let [r, g, b] = pixel_sum;
            values.put_pixel(x, y, Rgb([r, g, b]));
        }

        for (x, y) in ImageIterator::new(values.width(), values.height()) {
            let mut pixel_sum = [0., 0., 0.];

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (SIZE as isize / 2);
                let pixel_index = ImageBuffer::<Rgb<f32>, Vec<f32>>::compute_pixel_index(
                    stride,
                    SIZE,
                    relative_kernel_index,
                    y as usize,
                    values.height() as usize
                );

                let mut computed_values = values.get_pixel(x, pixel_index).0;
                computed_values = [computed_values[0] * value, computed_values[1] * value, computed_values[2] * value];

                pixel_sum = [pixel_sum[0] + computed_values[0], pixel_sum[1] + computed_values[1], pixel_sum[2] + computed_values[2]];
            }

            let [r, g, b] = pixel_sum;
            values.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().to_vec()
    }
}

impl<const SIZE: usize> Kernel<SIZE, DynamicImage> for NonSeparableKernel<SIZE> {
    fn apply(&self, values: &mut DynamicImage, stride: usize) {
        match values {
            DynamicImage::ImageLuma8(_)
            | DynamicImage::ImageLumaA8(_)
            | DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA16(_) => {
                let mut buffer = values.to_luma32f();
                self.apply(&mut buffer, stride);

                let (width, height) = buffer.dimensions();

                *values = DynamicImage::ImageLuma16(ImageBuffer::from_vec(width, height, buffer.into_iter().map(|item| (*item * u16::MAX as f32) as u16).collect::<Vec<_>>()).unwrap());
            }
            DynamicImage::ImageRgb8(_)
            | DynamicImage::ImageRgba8(_)
            | DynamicImage::ImageRgb16(_)
            | DynamicImage::ImageRgba16(_)
            | DynamicImage::ImageRgb32F(_)
            | DynamicImage::ImageRgba32F(_) => {
                let mut buffer = values.to_rgb32f();
                self.apply(&mut buffer, stride);

                *values = DynamicImage::ImageRgb32F(buffer);
            },
            &mut _ => unimplemented!()
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().iter().flatten().copied().collect()
    }
}

impl<const SIZE: usize> Kernel<SIZE, DynamicImage> for SeparableKernel<SIZE> {
    fn apply(&self, values: &mut DynamicImage, stride: usize) {
        match values {
            DynamicImage::ImageLuma8(_)
            | DynamicImage::ImageLumaA8(_)
            | DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA16(_) => {
                let mut buffer = values.to_luma32f();
                self.apply(&mut buffer, stride);

                let (width, height) = buffer.dimensions();

                *values = DynamicImage::ImageLuma16(ImageBuffer::from_vec(width, height, buffer.into_iter().map(|item| (*item * u16::MAX as f32) as u16).collect::<Vec<_>>()).unwrap());
            }
            DynamicImage::ImageRgb8(_)
            | DynamicImage::ImageRgba8(_)
            | DynamicImage::ImageRgb16(_)
            | DynamicImage::ImageRgba16(_)
            | DynamicImage::ImageRgb32F(_)
            | DynamicImage::ImageRgba32F(_) => {
                let mut buffer = values.to_rgb32f();
                self.apply(&mut buffer, stride);

                *values = DynamicImage::ImageRgb32F(buffer);
            },
            &mut _ => unimplemented!()
        }
    }

    fn values(&self) -> Vec<f32> {
        self.values().to_vec()
    }
}