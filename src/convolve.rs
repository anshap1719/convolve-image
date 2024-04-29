use crate::kernel::{SeparableKernel};

pub trait Convolution {
    fn compute_pixel_index(
        stride: usize,
        kernel_size: usize,
        kernel_index: isize,
        pixel_index: usize,
        max: usize,
    ) -> u32 {
        let kernel_size = kernel_size as isize;
        let kernel_padding = kernel_size / 2;

        let distance = kernel_index * stride as isize;

        let mut index = pixel_index as isize + distance;

        if index < 0 {
            index = -index;
        } else if index > max as isize - kernel_padding {
            let overshot_distance = index - max as isize + kernel_padding;
            index = max as isize - overshot_distance;
        }

        index as u32
    }

    fn convolve<const KERNEL_SIZE: usize>(
        &mut self,
        kernel: SeparableKernel<KERNEL_SIZE>,
        stride: usize,
    );
}