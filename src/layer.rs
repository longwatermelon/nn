use crate::matrix::{Matrix, Shape3, Shape4, Shape};

use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Linear,
    Relu,
    Tanh
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer {
    Dense(Dense),
    Conv(Conv)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Pooling {
    Max
}

#[derive(Debug)]
pub enum Delta {
    Dense {
        dw: Matrix,
        db: Vec<f32>
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dense {
    pub n: usize,
    w: Matrix,
    b: Vec<f32>,
    afn: Activation,

    pub a: Matrix,
    z: Matrix,

    dz: Matrix
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conv {
    nc: usize,
    nh: usize,
    nw: usize,
    fh: usize,
    fw: usize,
    ph: usize,
    pw: usize,
    afn: Activation,
    pooling: Pooling,

    w: Shape4,
    b: Vec<f32>,

    a: Shape4,
    z: Shape4,
    p: Shape4,
    row_maxes: Vec<Vec<Vec<Vec<usize>>>>,
    col_maxes: Vec<Vec<Vec<Vec<usize>>>>,

    dz: Shape4
}

pub trait Prop {
    fn forward_prop(&mut self, back: &Layer, x: &Matrix);
    fn back_prop(&mut self, back: &Layer, front: Option<&Layer>, y: &Matrix) -> Delta;
}

impl Activation {
    pub fn getfn(&self) -> impl Fn(f32) -> f32 {
        match self {
            Activation::Linear =>  |z: f32| z,
            Activation::Sigmoid => |z: f32| 1. / (1. + f32::exp(-z)),
            Activation::Relu =>    |z: f32| f32::max(0., z),
            Activation::Tanh =>    |z: f32| (f32::exp(z) - f32::exp(-z)) / (f32::exp(z) + f32::exp(-z))
        }
    }

    pub fn getfn_derivative(&self) -> impl Fn(f32) -> f32 {
        match self {
            Activation::Linear =>  |_| 1.,
            Activation::Sigmoid => |z: f32| {
                let sigmoid = |z: f32| 1. / (1. + f32::exp(-z));
                sigmoid(z) * (1. - sigmoid(z))
            },
            Activation::Relu =>    |z: f32| if z > 0. { z } else { 0. },
            Activation::Tanh =>    |z: f32| {
                let tanh = |z: f32| (f32::exp(z) - f32::exp(-z)) / (f32::exp(z) + f32::exp(-z));
                1. - tanh(z) * tanh(z)
            }
        }
    }
}

impl Layer {
    pub fn dense(n: usize, afn: Activation) -> Self {
        Layer::Dense(Dense::new(n, afn))
    }

    pub fn conv(filters: usize, fshape: (usize, usize),
                afn: Activation, pooling: Pooling, pshape: (usize, usize)) -> Self {
        Layer::Conv(Conv::new(filters, fshape, afn, pooling, pshape))
    }

    pub fn to_dense(&self) -> &Dense {
        match self {
            Layer::Dense(d) => d,
            Layer::Conv(_c) => todo!()
        }
    }

    pub fn to_conv(&self) -> &Conv {
        match self {
            Layer::Dense(_d) => todo!(),
            Layer::Conv(c) => c
        }
    }

    pub fn apply_delta(&mut self, delta: &Delta, a: f32) {
        match delta {
            Delta::Dense { dw, db } => {
                if let Layer::Dense(d) = self {
                    d.w = d.w.clone() - &(dw.clone() * a);
                    d.b.iter_mut().zip(db.iter()).for_each(|(b, db): (&mut f32, &f32)| *b *= db);
                } else {
                    panic!("Mismatch in delta and layer type: delta ({:?}) layer ({:?})",
                           delta, self);
                }
            }
        }
    }
}

impl Dense {
    pub fn new(n: usize, afn: Activation) -> Self {
        Self {
            n,
            w: Matrix::new(1, 1),
            b: Vec::new(),
            afn,
            a: Matrix::new(1, 1),
            z: Matrix::new(1, 1),
            dz: Matrix::new(1, 1)
        }
    }

    pub fn adjust_dims(&mut self, back_n: usize, m: usize) {
        self.w = Matrix::new(self.n, back_n);
        self.w.random_init(-0.2, 0.2);
        self.b.resize(self.n, 0.);

        self.a = Matrix::new(self.n, m);
        self.z = Matrix::new(self.n, m);
        self.dz = Matrix::new(self.n, m);
    }
}

impl Conv {
    pub fn new(filters: usize, fshape: (usize, usize),
               afn: Activation, pooling: Pooling, pshape: (usize, usize)) -> Self {
        Self {
            nc: filters,
            nh: 0,
            nw: 0,
            fh: fshape.0,
            fw: fshape.1,
            ph: pshape.0,
            pw: pshape.1,
            afn,
            pooling,
            w: Shape4::default(),
            b: Vec::new(),
            a: Shape4::default(),
            z: Shape4::default(),
            p: Shape4::default(),
            row_maxes: Vec::new(),
            col_maxes: Vec::new(),
            dz: Shape4::default()
        }
    }

    pub fn adjust_dims(&mut self, back_nc: usize, back_nw: usize, back_nh: usize, m: usize) {
        self.nh = back_nh - self.fh + 1;
        self.nw = back_nw - self.fw + 1;

        self.w = Shape4::new(self.nc, back_nc, self.fh, self.fw);
        self.b.resize(self.nc, 0.);

        self.a = Shape4::new(m, self.nc, self.nh, self.nw);
        self.z = self.a.clone();
        self.dz = self.z.clone();

        self.p = Shape4::new(
            self.a.shape().0,
            self.a.shape().1,
            self.a.shape().2 / self.ph,
            self.a.shape().3 / self.pw
        );

        self.row_maxes = vec![vec![vec![vec![0; self.p.shape().3];
                        self.p.shape().2]; self.p.shape().1];
                        self.p.shape().0];
        self.col_maxes = self.row_maxes.clone();
    }

    fn pool(&mut self) {
        match self.pooling {
            Pooling::Max => {
                for e in 0..self.p.shape().0 {
                    for c in 0..self.p.shape().1 {
                        for row in 0..self.p.shape().2 {
                            for col in 0..self.p.shape().3 {
                                self.pool_one(row, col, e, c);
                            }
                        }
                    }
                }
            }
        }
    }

    fn pool_one(&mut self, row: usize, col: usize, e: usize, c: usize) {
        match self.pooling {
            Pooling::Max => {
                let mut largest: f32 = f32::NEG_INFINITY;
                let mut index: (usize, usize) = (0, 0);

                for dr in 0..self.ph {
                    for dc in 0..self.pw {
                        let i: (usize, usize) = (
                            row * self.ph + dr,
                            col * self.pw + dc
                        );
                        let a: f32 = self.a.at(e).at(c).at(i.0, i.1);

                        if a > largest {
                            largest = a;
                            index = i;
                        }
                    }
                }

                *self.p.at_mut(e).at_mut(c).atref(row, col) = largest;
                self.row_maxes[e][c][row][col] = index.0;
                self.col_maxes[e][c][row][col] = index.1;
            }
        }
    }
}

impl Prop for Dense {
    fn forward_prop(&mut self, back: &Layer, x: &Matrix) {
        let bl: &Dense = back.to_dense();

        self.a = Matrix::new(self.n, x.cols());
        let afn = self.afn.getfn();

        self.z = self.w.clone() * &bl.a;
        self.z = self.z.foreach(|r, c| self.z.at(r, c) + self.b[r]);
        self.a = self.a.foreach(|r, c| self.a.at(r, c) + afn(self.z.at(r, c)));
    }

    fn back_prop(&mut self, back: &Layer, front: Option<&Layer>, y: &Matrix) -> Delta {
        let bl: &Dense = back.to_dense();

        if let Some(front) = front {
            let fl: &Dense = front.to_dense();

            let afn = self.afn.getfn_derivative();
            let left: Matrix = fl.w.transpose() * &fl.dz;
            let right: Matrix = self.z.foreach(|r, c| afn(self.z.at(r, c)));

            self.dz = left.element_wise_mul(right);
        } else {
            self.dz = self.a.clone() - y;
        }

        let dw: Matrix = self.dz.clone() * &bl.a.transpose() * (1. / y.cols() as f32);
        let mut db: Vec<f32> = Vec::with_capacity(self.dz.rows());
        for r in self.dz.data() {
            db.push(r.iter().sum());
        }

        Delta::Dense {
            dw, db
        }
    }
}

impl Prop for Conv {
    fn forward_prop(&mut self, back: &Layer, x: &Matrix) {
        let m: usize = x.cols();
        let bl: &Conv = back.to_conv();

        for e in 0..m {
            for n in 0..self.nc {
                let z: Matrix = convolve(&bl.a, &self.w, e, n) +
                                &Matrix::from(
                                    vec![self.b.clone()]
                                ).transpose();
                *self.z.at_mut(e).at_mut(n) = z;
            }
        }

        let afn = self.afn.getfn();
        for (block, zblock) in self.a.data_mut()
                                     .iter_mut()
                                     .zip(self.z.data().iter()) {
            for (channel, zchannel) in block.data_mut()
                                            .iter_mut()
                                            .zip(zblock.data().iter()) {
                *channel = channel.foreach(|r, c| afn(zchannel.at(r, c)));
            }
        }
    }

    fn back_prop(&mut self, back: &Layer, front: Option<&Layer>, y: &Matrix) -> Delta {
        match front.unwrap() {
            // conv -> dense
            Layer::Dense(fl) => {
                let dlf: Matrix = fl.w.transpose() * &fl.dz;
                let dlp: Shape4 = dlf.reshape_to4(self.p.shape());
            },
            // conv -> conv
            Layer::Conv(fl) => {
            }
        }

        todo!()
    }
}

fn convolve(input: &Shape4, filter: &Shape4, e: usize, n: usize) -> Matrix {
    let layers: Vec<Matrix> = input.at(e).data().iter()
        .zip(filter.data().iter())
        .map(|(a, w)| a.convolve(w.at(n)))
        .collect();
    layers.iter()
        .fold(Matrix::new(layers[0].rows(), layers[0].cols()),
            |sum, val| sum + val
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjust_dims_dense() {
        let mut dense: Dense = Dense::new(10, Activation::Linear);
        dense.adjust_dims(5, 2);

        assert_eq!(dense.w.rows(), 10);
        assert_eq!(dense.w.cols(), 5);
        assert_eq!(dense.a.rows(), 10);
        assert_eq!(dense.a.cols(), 2);
        assert_eq!(dense.z.rows(), 10);
        assert_eq!(dense.z.cols(), 2);
        assert_eq!(dense.dz.rows(), 10);
        assert_eq!(dense.dz.cols(), 2);
    }

    #[test]
    fn adjust_dims_conv() {
        let mut conv: Conv = Conv::new(6, (5, 5), Activation::Linear,
                                       Pooling::Max, (2, 2));
        conv.adjust_dims(3, 32, 32, 1);

        assert_eq!(conv.a.shape(), (1, 6, 28, 28));
        assert_eq!(conv.p.shape(), (1, 6, 14, 14));
        assert_eq!(conv.w.shape(), (6, 3, 5, 5));
    }

    #[test]
    fn conv_maxpool() {
        let mut conv: Conv = Conv::new(6, (5, 5), Activation::Linear,
                                       Pooling::Max, (2, 2));
        conv.a = Shape4::from(
            vec![
                Shape3::from(
                    vec![
                        Matrix::from(
                            vec![
                                vec![1., 2., 3., 4.],
                                vec![5., 6., 7., 8.],
                                vec![9., 1., 2., 3.],
                                vec![4., 5., 6., 7.]
                            ]
                        )
                    ]
                )
            ]
        );

        conv.p = Shape4::new(
            conv.a.shape().0,
            conv.a.shape().1,
            conv.a.shape().2 / conv.ph,
            conv.a.shape().3 / conv.pw
        );

        conv.row_maxes = vec![vec![vec![vec![0; conv.p.shape().3];
                        conv.p.shape().2]; conv.p.shape().1];
                        conv.p.shape().0];
        conv.col_maxes = conv.row_maxes.clone();
        conv.pool();

        assert_eq!(conv.row_maxes, vec![
            vec![
                vec![
                    vec![1, 1],
                    vec![2, 3]
                ]
            ]
        ]);

        assert_eq!(conv.p, Shape4::from(
            vec![
                Shape3::from(
                    vec![
                        Matrix::from(
                            vec![
                                vec![6., 8.],
                                vec![9., 7.]
                            ]
                        )
                    ]
                )
            ]
        ));
    }
}

