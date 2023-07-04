use crate::matrix::{Matrix, Shape3, Shape4, Shape};

pub mod img {
    use super::*;

    /// Vec of images to matrix (pixel index, example)
    pub mod flat {
        use super::*;

        /// images[example][pixel]
        pub fn one_dim(images: Vec<Vec<f32>>) -> Matrix {
            Matrix::from(images).transpose()
        }

        /// images[example].at(row, col)
        pub fn two_dim(images: Vec<Matrix>) -> Matrix {
            one_dim(images.iter().map(|x| x.flatten()).collect())
        }
    }

    /// Vec of images to 4D (example, channel, row, col)
    pub mod shaped {
        use super::*;

        /// images[example].get(channel).at(row, col)
        pub fn three_dim(images: Vec<Shape3>) -> Shape4 {
            let shape: (usize, usize, usize) = images[0].shape();
            let mut res: Shape4 = Shape4::new(images.len(), shape.0, shape.1, shape.2);
            for (e, img) in images.iter().enumerate() {
                *res.at_mut(e) = img.clone();
            }

            res
        }
    }
}

/// labels[example][output unit]
pub fn labels(labels: Vec<Vec<f32>>) -> Matrix {
    Matrix::from(labels).transpose()
}

