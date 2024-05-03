#[cfg(feature = "image")]
use std::cmp::Ordering;
#[cfg(feature = "image")]
use image::{DynamicImage, Rgb};

#[cfg(feature = "ndarray")]
use ndarray::{Array2, Array3};
#[cfg(feature = "ndarray")]
use crate::ndarray::Aggregate;

#[derive(Copy, Clone)]
pub enum RescaleRange {
    Custom(f32, f32),
    Max,
}

impl RescaleRange {
    fn min(self) -> f32 {
        match self {
            RescaleRange::Custom(min, _) => min,
            RescaleRange::Max => 0.
        }
    }

    fn max(self) -> f32 {
        match self {
            RescaleRange::Custom(_, max) => max,
            RescaleRange::Max => 1.
        }
    }
}

pub trait Rescale {
    fn min(&self) -> f32;
    fn max(&self) -> f32;
    fn rescale(&mut self, range: RescaleRange);
    fn channel_wise_rescale(&mut self, range: RescaleRange);
    fn rescale_value(min: f32, max: f32, value: f32, range: RescaleRange) -> f32 {
        let (new_min, new_max) = (range.min(), range.max());
        let new_range = new_max - new_min;

        new_min + ((value - min) * new_range / (max - min))
    }
}

#[cfg(feature = "image")]
impl Rescale for DynamicImage {
    fn min(&self) -> f32 {
        self
            .to_rgb32f()
            .iter()
            .min_by(|x, y| {
                if x > y {
                    Ordering::Greater
                } else if y > x {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .copied()
            .unwrap_or(0.)
            .min(0.)
    }

    fn max(&self) -> f32 {
        self
            .to_rgb32f()
            .iter()
            .max_by(|x, y| {
                if x > y {
                    Ordering::Greater
                } else if y > x {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .copied()
            .unwrap_or(1.)
            .max(1.)
    }

    fn rescale(&mut self, range: RescaleRange) {
        let min = self.min();
        let max = self.max();

        for pixel in self.to_rgb32f().pixels_mut() {
            let [r, g, b] = pixel.0;
            *pixel = Rgb([Self::rescale_value(min, max, r, range), Self::rescale_value(min, max, g, range), Self::rescale_value(min, max, b, range)]);
        }
    }

    fn channel_wise_rescale(&mut self, range: RescaleRange) {
        let mut max_r: f32 = 1.;
        let mut max_g: f32 = 1.;
        let mut max_b: f32 = 1.;

        let mut min_r: f32 = 0.;
        let mut min_g: f32 = 0.;
        let mut min_b: f32 = 0.;

        let mut source = self.to_rgb32f();

        for pixel in source.pixels() {
            let [r, g, b] = pixel.0;
            max_r = max_r.max(r);
            max_g = max_g.max(g);
            max_b = max_b.max(b);

            min_r = min_r.min(r);
            min_g = min_g.min(g);
            min_b = min_b.min(b);
        }

        for pixel in source.pixels_mut() {
            let [r, g, b] = pixel.0;
            *pixel = Rgb([
                Self::rescale_value(min_r, max_r, r, range),
                Self::rescale_value(min_g, max_g, g, range),
                Self::rescale_value(min_b, max_b, b, range),
            ]);
        }
    }
}

#[cfg(feature = "ndarray")]
impl Rescale for Array2<f32> {
    fn min(&self) -> f32 {
        Aggregate::min(self)
    }

    fn max(&self) -> f32 {
        Aggregate::max(self)
    }

    fn rescale(&mut self, range: RescaleRange) {
        let min = Rescale::min(self);
        let max = Rescale::max(self);

        for item in self.iter_mut() {
            *item = Self::rescale_value(min, max, *item, range);
        }
    }

    fn channel_wise_rescale(&mut self, range: RescaleRange) {
        self.rescale(range)
    }
}

#[cfg(feature = "ndarray")]
impl Rescale for Array3<f32> {
    fn min(&self) -> f32 {
        Aggregate::min(self)
    }

    fn max(&self) -> f32 {
        Aggregate::max(self)
    }

    fn rescale(&mut self, range: RescaleRange) {
        let min = Rescale::min(self);
        let max = Rescale::max(self);

        for item in self.iter_mut() {
            *item = Self::rescale_value(min, max, *item, range);
        }
    }

    fn channel_wise_rescale(&mut self, range: RescaleRange) {
        let mut max_r: f32 = 1.;
        let mut max_g: f32 = 1.;
        let mut max_b: f32 = 1.;

        let mut min_r: f32 = 0.;
        let mut min_g: f32 = 0.;
        let mut min_b: f32 = 0.;

        for ((_, _, channel), item) in self.indexed_iter() {
            match channel {
                0 => {
                    min_r = min_r.min(*item);
                    max_r = max_r.max(*item);
                }
                1 => {
                    min_g = min_g.min(*item);
                    max_g = max_g.max(*item);
                }
                2 => {
                    min_b = min_b.min(*item);
                    max_b = max_b.max(*item);
                }
                _ => panic!("Unexpected number of channels")
            };
        }

        for ((_, _, channel), item) in self.indexed_iter_mut() {
            match channel {
                0 => {
                    *item = Self::rescale_value(min_r, max_r, *item, range);
                }
                1 => {
                    *item = Self::rescale_value(min_g, max_g, *item, range);
                }
                2 => {
                    *item = Self::rescale_value(min_b, max_b, *item, range);
                }
                _ => panic!("Unexpected number of channels")
            };
        }
    }
}