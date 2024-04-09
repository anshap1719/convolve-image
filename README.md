# Image Convolution Using Separable Kernel

This project provides an implementation convolution on images using separable kernels of arbitrary size. The library is aimed to provide a fast implementation with 0 dependencies other than `image-rs`.

## Usage

```rust
use convolve_image::convolve::Convolution;
use convolve_image::kernel::SeparableKernel;

fn convolve_image() {
    let image = image::open("./sample.jpg").unwrap();
    let image = image.to_rgb32f();
    image.convolve(SeparableKernel::new([1. / 4., 1. / 2., 1. / 4.]), 1);
}
```

## Installation

To use this library in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
convolve-image = "0.1.0"