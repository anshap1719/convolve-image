pub mod kernel;
pub mod convolve;

#[cfg(feature = "image")]
pub mod image;

#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "ndarray")]
pub(crate) mod dimensions;

pub use convolve::Convolution;