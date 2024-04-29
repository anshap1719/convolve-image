#![cfg(feature = "ndarray")]

use ndarray::{Dimension, Ix2, Ix3};

pub trait DimensionIterator {
    type Item: Copy;
    
    fn into_iter(self) -> impl IntoIterator<Item = Self::Item>;
}

impl DimensionIterator for Ix2 {
    type Item = (usize, usize);

    fn into_iter(self) -> impl IntoIterator<Item=Self::Item> {
        let [y, x] = self.slice() else {
            panic!("Expected 2d slice")
        };
        
        let y = *y;
        let x = *x;

        (0..y).flat_map(move |y| (0..x).map(move |x| (y, x)))
    }
}

impl DimensionIterator for Ix3 {
    type Item = (usize, usize, usize);

    fn into_iter(self) -> impl IntoIterator<Item=Self::Item> {
        let [y, x, z] = self.slice() else {
            panic!("Expected 3d slice")
        };

        let y = *y;
        let x = *x;
        let z = *z;

        (0..y).flat_map(move |y| (0..x).flat_map(move |x| (0..z).map(move |z| (y, x, z))))
    }
}