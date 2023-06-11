use std::ops;
use rand::Rng;

use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    data: Vec<Vec<f32>>
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Shape3 {
    data: Vec<Matrix>
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Shape4 {
    data: Vec<Shape3>
}

pub trait Shape {
    type T;
    type Enclosed;
    fn flatten(&self) -> Vec<f32>;
    fn at(&self, index: usize) -> &Self::Enclosed;
    fn at_mut(&mut self, index: usize) -> &mut Self::Enclosed;
    fn data(&self) -> &Vec<Self::Enclosed>;
    fn data_mut(&mut self) -> &mut Vec<Self::Enclosed>;
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

    pub fn from_1d(data: &[f32], rows: usize, cols: usize) -> Self {
        let mut index: usize = 0;
        let mut res: Matrix = Matrix::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                *res.atref(r, c) = data[index];
                index += 1;
            }
        }

        res
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

    pub fn convolve(&self, filter: &Matrix) -> Matrix {
        let nh: usize = self.rows() - filter.rows() + 1;
        let nw: usize = self.cols() - filter.cols() + 1;
        let mut res: Matrix = Matrix::new(nh, nw);

        for r in 0..res.rows() {
            for c in 0..res.cols() {
                *res.atref(r, c) = filter.foreach(
                        |fr, fc| self.at(r + fr, c + fc) * filter.at(fr, fc))
                    .flatten()
                    .iter()
                    .sum();
            }
        }

        res
    }

    pub fn reshape_to4(&self, dims: (usize, usize, usize, usize)) -> Shape4 {
        let flat: Vec<f32> = self.flatten();
        let mut res: Shape4 = Shape4::new(dims.0, dims.1, dims.2, dims.3);
        let mut index: usize = 0;

        for i in 0..dims.0 {
            for j in 0..dims.1 {
                for k in 0..dims.2 {
                    for l in 0..dims.3 {
                        *res.at_mut(i).at_mut(j).atref(k, l) = flat[index];
                        index += 1;
                    }
                }
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

    pub fn flatten(&self) -> Vec<f32> {
        self.data.clone().into_iter().flatten().collect()
    }

    pub fn sum(&self) -> f32 {
        self.data.iter()
            .flatten()
            .fold(0., |acc, &x| acc + x)
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

    pub fn dims(&self) -> String {
        format!("{}x{}", self.rows(), self.cols())
    }

    pub fn data(&self) -> &Vec<Vec<f32>> {
        &self.data
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl ops::Mul<&Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
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
        let mut res: Matrix = self;

        for r in 0..res.rows() {
            for c in 0..res.cols() {
                *res.atref(r, c) *= rhs;
            }
        }

        res
    }
}

impl ops::Add<&Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.rows(), self.cols());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                *res.atref(r, c) = self.at(r, c) + rhs.at(r, c);
            }
        }

        res
    }
}

impl ops::Add<f32> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: f32) -> Matrix {
        let mut res: Matrix = Matrix::new(self.rows(), self.cols());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                *res.atref(r, c) = self.at(r, c) + rhs;
            }
        }

        res
    }
}

impl ops::Sub<&Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.rows(), self.cols());

        for r in 0..self.rows() {
            for c in 0..self.cols() {
                *res.atref(r, c) = self.at(r, c) - rhs.at(r, c);
            }
        }

        res
    }
}

impl Shape3 {
    pub fn new(channels: usize, rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Matrix::new(rows, cols); channels]
        }
    }
}

impl Shape for Shape3 {
    type T = Shape3;
    type Enclosed = Matrix;
    fn flatten(&self) -> Vec<f32> {
        self.data.iter()
                 .flat_map(|x| x.flatten())
                 .collect()
    }

    fn at(&self, index: usize) -> &Matrix {
        &self.data[index]
    }

    fn at_mut(&mut self, index: usize) -> &mut Matrix {
        &mut self.data[index]
    }

    fn data(&self) -> &Vec<Matrix> {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Vec<Matrix> {
        &mut self.data
    }
}

impl From<Vec<Matrix>> for Shape3 {
    fn from(data: Vec<Matrix>) -> Shape3 {
        Shape3 { data }
    }
}

impl Shape4 {
    pub fn new(blocks: usize, channels: usize, rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Shape3::new(channels, rows, cols); blocks]
        }
    }

    pub fn from_1d(data: &[f32], shape: (usize, usize, usize, usize)) -> Self {
        let mut index: usize = 0;
        let mut res: Shape4 = Shape4::new(shape.0, shape.1, shape.2, shape.3);
        for b in 0..shape.0 {
            for c in 0..shape.1 {
                for i in 0..shape.2 {
                    for j in 0..shape.3 {
                        *res.at_mut(b).at_mut(c).atref(i, j) = data[index];
                        index += 1;
                    }
                }
            }
        }

        res
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.data.len(), self.data[0].data().len(),
        self.data[0].data()[0].rows(), self.data[0].data()[0].cols())
    }

    pub fn zero(&self) -> Self {
        let mut res: Shape4 = self.clone();

        for i in 0..self.shape().0 {
            for j in 0..self.shape().1 {
                *res.data[i].at_mut(j) = self.data[i].at(j).foreach(|_, _| 0.);
            }
        }

        res
    }

    pub fn foreach(&self, f: impl Fn((usize, usize), &Matrix) -> Matrix) -> Self {
        let mut res: Shape4 = self.clone();
        for block in 0..self.shape().0 {
            for channel in 0..self.shape().1 {
                *res.at_mut(block).at_mut(channel) = f((block, channel), res.at(block).at(channel));
            }
        }

        res
    }

    pub fn sum(&self) -> f32 {
        let mut sum: f32 = 0.;
        for block in 0..self.shape().0 {
            for channel in 0..self.shape().1 {
                sum += self.at(block).at(channel).sum();
            }
        }

        sum
    }
}

impl Shape for Shape4 {
    type T = Shape4;
    type Enclosed = Shape3;
    fn flatten(&self) -> Vec<f32> {
        self.data.iter()
                 .flat_map(|x| x.flatten())
                 .collect()
    }

    fn at(&self, index: usize) -> &Shape3 {
        &self.data[index]
    }

    fn at_mut(&mut self, index: usize) -> &mut Shape3 {
        &mut self.data[index]
    }

    fn data(&self) -> &Vec<Shape3> {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Vec<Shape3> {
        &mut self.data
    }
}

impl From<Vec<Shape3>> for Shape4 {
    fn from(data: Vec<Shape3>) -> Shape4 {
        Shape4 { data }
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

        assert_eq!(a * &b,
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
        assert_eq!(a + &b,
            Matrix::from(vec![
                vec![2., 2.],
                vec![2., 2.]
            ])
        );
    }

    #[test]
    fn mat_flatten() {
        let a: Matrix = Matrix::from(vec![
            vec![1., 2.],
            vec![3., 4.]
        ]);

        let v: Vec<f32> = a.flatten();
        assert_eq!(v, [1., 2., 3., 4.]);
    }

    #[test]
    fn shape4_flatten() {
        let a: Shape4 = Shape4::from(vec![
            Shape3::from(vec![
                Matrix::from(vec![
                    vec![1., 2.],
                    vec![3., 4.]
                ]),
                Matrix::from(vec![
                    vec![5., 6.],
                    vec![7., 8.]
                ]),
            ]),
            Shape3::from(vec![
                Matrix::from(vec![
                    vec![9., 10.],
                    vec![11., 12.]
                ]),
                Matrix::from(vec![
                    vec![13., 14.],
                    vec![15., 16.]
                ]),
            ])
        ]);

        let v: Vec<f32> = a.flatten();
        assert_eq!(v, (1..=16).map(|x| x as f32).collect::<Vec<f32>>());
    }

    #[test]
    fn convolution() {
        let input: Matrix = Matrix::from(vec![
            vec![1., 1., 1., 0., 0., 0.],
            vec![1., 1., 1., 0., 0., 0.],
            vec![1., 1., 1., 0., 0., 0.],
            vec![1., 1., 1., 0., 0., 0.],
            vec![1., 1., 1., 0., 0., 0.],
            vec![1., 1., 1., 0., 0., 0.]
        ]);

        let filter: Matrix = Matrix::from(vec![
            vec![1., 0., -1.],
            vec![1., 0., -1.],
            vec![1., 0., -1.]
        ]);

        let output: Matrix = input.convolve(&filter);
        assert_eq!(output, Matrix::from(vec![
            vec![0., 3., 3., 0.],
            vec![0., 3., 3., 0.],
            vec![0., 3., 3., 0.],
            vec![0., 3., 3., 0.]
        ]));
    }

    #[test]
    fn reshape_to4() {
        let input: Matrix = Matrix::from(vec![
            vec![1., 2., 3.],
            vec![4., 5., 6.],
            vec![7., 8., 9.],
            vec![10., 11., 12.],
        ]);

        let shape = (2, 2, 3, 1);
        let reshaped: Shape4 = input.reshape_to4(shape);
        assert_eq!(reshaped.flatten(), (1..=12).map(|x| x as f32).collect::<Vec<f32>>());
    }

    #[test]
    fn matsum() {
        let m: Matrix = Matrix::from(vec![
            vec![1., 1., 1.],
            vec![2., 2., 2.]
        ]);

        assert_eq!(m.sum(), 9.);
    }

    #[test]
    fn shape4sum() {
        let s: Shape4 = Shape4::from(
            vec![
                Shape3::from(vec![
                    Matrix::from(
                        vec![
                            vec![1., 2., 3.],
                            vec![4., 5., 6.]
                        ]
                    )
                ]),
                Shape3::from(vec![
                    Matrix::from(
                        vec![
                            vec![1., 2., 3.],
                            vec![4., 5., 6.]
                        ]
                    )
                ])
            ]
        );

        assert_eq!(s.sum(), 42.);
    }
}

