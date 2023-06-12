use crate::matrix::Matrix;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PoolType {
    Max
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pooling {
    ptype: PoolType,
    pub(crate) w: usize,
    pub(crate) h: usize
}

impl Pooling {
    pub fn new(ptype: PoolType, w: usize, h: usize) -> Self {
        Self { ptype, w, h }
    }
    pub fn pool(&self, input: &Matrix) -> (Matrix, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        match self.ptype {
            PoolType::Max => {
                let mut res: Matrix = Matrix::new(input.rows() / self.h, input.cols() / self.w);
                let mut row_maxes: Vec<Vec<usize>> = vec![vec![0; res.cols()]; res.rows()];
                let mut col_maxes: Vec<Vec<usize>> = row_maxes.clone();

                for r in 0..res.rows() {
                    for c in 0..res.cols() {
                        self.pool_one(input, &mut res, &mut row_maxes, &mut col_maxes, r, c);
                    }
                }

                (res, row_maxes, col_maxes)
            }
        }
    }

    fn pool_one(&self, input: &Matrix, res: &mut Matrix,
                row_maxes: &mut [Vec<usize>], col_maxes: &mut [Vec<usize>],
                row: usize, col: usize) {
        match self.ptype {
            PoolType::Max => {
                let mut largest: f32 = f32::NEG_INFINITY;
                let mut index: (usize, usize) = (0, 0);

                for dr in 0..self.h {
                    for dc in 0..self.w {
                        let i: (usize, usize) = (
                            row * self.h + dr,
                            col * self.w + dc
                        );
                        let a: f32 = input.at(i.0, i.1);

                        if a > largest {
                            largest = a;
                            index = i;
                        }
                    }
                }

                *res.atref(row, col) = largest;
                row_maxes[row][col] = index.0;
                col_maxes[row][col] = index.1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool() {
        let m: Matrix = Matrix::from(vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 1., 2., 3.],
            vec![4., 5., 6., 7.]
        ]);

        let pooling: Pooling = Pooling::new(PoolType::Max, 2, 2);
        let (p, rm, cm) = pooling.pool(&m);

        assert_eq!(p, Matrix::from(vec![vec![6., 8.], vec![9., 7.]]));
        assert_eq!(rm, vec![vec![1, 1], vec![2, 3]]);
        assert_eq!(cm, vec![vec![1, 3], vec![0, 3]]);
    }
}

