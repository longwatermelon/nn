use std::ops;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    data: Vec<Vec<f32>>
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![0.; cols]; rows]
        }
    }

    pub fn from(data: Vec<Vec<f32>>) -> Self {
        Self { data }
    }

    pub fn random_init(&mut self, lower: f32, upper: f32) {
        *self = self.foreach(
            |_, _| rand::thread_rng().gen_range(0..1000) as f32 / 1000. *
                   (upper - lower) + lower
        );
    }

    pub fn atref(&mut self, row: usize, col: usize) -> &mut f32 {
        self.check_valid(row, col);
        &mut self.data[row][col]
    }

    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.check_valid(row, col);
        self.data[row][col]
    }

    pub fn transpose(&self) -> Matrix {
        let mut res: Matrix = Matrix::new(self.cols(), self.rows());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                *res.atref(c, r) = self.at(r, c);
            }
        }

        res
    }

    pub fn foreach(&self, f: impl Fn(usize, usize) -> f32) -> Matrix {
        let mut res: Matrix = Matrix::new(self.rows(), self.cols());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                *res.atref(r, c) = f(r, c);
            }
        }

        res
    }

    pub fn extract_row(&self, row: usize) -> Vec<f32> {
        let mut res: Vec<f32> = Vec::new();
        for c in 0..self.cols() {
            res.push(self.at(row, c));
        }

        res
    }

    pub fn extract_col(&self, col: usize) -> Vec<f32> {
        let mut res: Vec<f32> = Vec::new();
        for r in 0..self.rows() {
            res.push(self.at(r, col));
        }

        res
    }

    pub fn element_wise_mul(&self, other: Matrix) -> Matrix {
        self.foreach(|row, col| self.at(row, col) * other.at(row, col))
    }

    pub fn check_valid(&self, row: usize, col: usize) {
        if row >= self.rows() || col >= self.cols() {
            panic!(
                "Error: {}x{} matrix indexed with ({},{})",
                self.rows(), self.cols(), row, col
            );
        }
    }

    pub fn rows(&self) -> usize {
        self.data.len()
    }

    pub fn cols(&self) -> usize {
        self.data[0].len()
    }

    pub fn data(&self) -> &Vec<Vec<f32>> {
        &self.data
    }
}

impl ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.rows(), rhs.cols());

        for r in 0..self.rows() {
            for c in 0..rhs.cols() {
                for i in 0..self.cols() {
                    *res.atref(r, c) += self.at(r, i) * rhs.at(i, c);
                }
            }
        }

        res
    }
}

impl ops::Mul<f32> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Matrix {
        let mut res: Matrix = self.clone();

        for r in 0..res.rows() {
            for c in 0..res.cols() {
                *res.atref(r, c) *= rhs;
            }
        }

        res
    }
}

impl ops::Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.rows(), self.cols());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                *res.atref(r, c) = self.at(r, c) + rhs.at(r, c);
            }
        }

        res
    }
}

impl ops::Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.rows(), self.cols());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                *res.atref(r, c) = self.at(r, c) - rhs.at(r, c);
            }
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose() {
        let m: Matrix = Matrix::from(vec![
            vec![1., 2., 3.],
            vec![4., 5., 6.],
            vec![7., 8., 9.]
        ]);
        let transpose: Matrix = m.transpose();

        assert_eq!(transpose.rows(), m.cols());
        assert_eq!(transpose.cols(), m.rows());

        for r in 0..m.rows() {
            for c in 0..m.cols() {
                assert_eq!(m.at(r, c), transpose.at(c, r));
            }
        }
    }

    #[test]
    fn foreach() {
        let mut m: Matrix = Matrix::from(vec![
            vec![1., 1., 1.],
            vec![1., 1., 1.],
            vec![1., 1., 1.]
        ]);
        m = m.foreach(|r, c| m.at(r, c) + 1.);

        for r in 0..m.rows() {
            for c in 0..m.cols() {
                assert_eq!(m.at(r, c), 2.);
            }
        }
    }

    #[test]
    fn extract() {
        let m: Matrix = Matrix::from(vec![
            vec![1., 2., 3.],
            vec![4., 5., 6.]
        ]);

        let row: Vec<f32> = m.extract_row(0);
        let col: Vec<f32> = m.extract_col(1);

        assert_eq!(row, vec![1., 2., 3.]);
        assert_eq!(col, vec![2., 5.]);
    }

    #[test]
    fn elementwise() {
        let a: Matrix = Matrix::from(vec![
            vec![2., 2., 2.],
            vec![2., 2., 2.]
        ]);

        let b: Matrix = Matrix::from(vec![
            vec![1., 2., 3.],
            vec![2., 3., 4.]
        ]);

        let mul: Matrix = a.element_wise_mul(b);
        assert_eq!(mul,
            Matrix::from(vec![
                vec![2., 4., 6.],
                vec![4., 6., 8.]
            ])
        );
    }

    #[test]
    fn multiplication() {
        let a: Matrix = Matrix::from(vec![
            vec![2., 1., 4.],
            vec![0., 1., 1.]
        ]);

        let b: Matrix = Matrix::from(vec![
            vec![6., 3., -1., 0.],
            vec![1., 1., 0., 4.],
            vec![-2., 5., 0., 2.]
        ]);

        assert_eq!(a * b,
            Matrix::from(vec![
                vec![5., 27., -2., 12.],
                vec![-1., 6., 0., 6.]
            ])
        );
    }

    #[test]
    fn addition() {
        let a: Matrix = Matrix::from(vec![
            vec![1., 1.],
            vec![1., 1.]
        ]);

        let b: Matrix = a.clone();
        assert_eq!(a + b,
            Matrix::from(vec![
                vec![2., 2.],
                vec![2., 2.]
            ])
        );
    }
}

