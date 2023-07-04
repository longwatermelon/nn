use crate::matrix::{Matrix, Shape3, Shape4, Shape};

/// images[example][pixel]
pub fn images_flat(images: Vec<Vec<f32>>) -> Matrix {
    Matrix::from(images).transpose()
}

/// images[example].get(channel).at(row, col)
pub fn images_channels(images: Vec<Shape3>) -> Shape4 {
    let shape: (usize, usize, usize) = images[0].shape();
    let mut res: Shape4 = Shape4::new(images.len(), shape.0, shape.1, shape.2);
    for e in 0..images.len() {
        *res.at_mut(e) = images[e].clone();
    }

    res
}

/// labels[example][output unit]
pub fn labels(labels: Vec<Vec<f32>>) -> Matrix {
    Matrix::from(labels).transpose()
}

