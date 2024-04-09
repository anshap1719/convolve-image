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