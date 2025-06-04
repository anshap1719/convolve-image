use crate::dimensions::DimensionIterator;
#[cfg(feature = "ndarray")]
use ndarray::{Array2};
use ndarray::Array3;
use crate::Convolution;

#[derive(Copy, Clone)]
pub struct SeparableKernel<const SIZE: usize> {
    values: [f32; SIZE]
}

impl<const SIZE: usize> SeparableKernel<SIZE> {
    pub fn new(values: [f32; SIZE]) -> Self {
        Self {
            values
        }    
    }
    
    pub fn values(&self) -> [f32; SIZE] {
        self.values
    }
}

#[derive(Copy, Clone)]
pub struct NonSeparableKernel<const SIZE: usize> {
    values: [[f32; SIZE]; SIZE]
}

impl<const SIZE: usize> NonSeparableKernel<SIZE> {
    pub fn new(values:  [[f32; SIZE]; SIZE]) -> Self {
        Self {
            values
        }
    }

    pub fn values(&self) ->  [[f32; SIZE]; SIZE] {
        self.values
    }
}

pub trait Kernel<const KERNEL_SIZE: usize, T: Sized> {
    fn apply(&self, values: &mut T, stride: usize);
    
    fn values(&self) -> Vec<f32>;
}