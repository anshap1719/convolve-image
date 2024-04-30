pub struct ImageIterator {
    width: u32,
    height: u32,
    index: u32,
}

impl ImageIterator {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            index: 0_u32,
        }
    }
}

impl Iterator for ImageIterator {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.index;
        self.index += 1;
        if n < (self.width * self.height) {
            Some((n / self.height, n % self.height))
        } else {
            None
        }
    }
}
