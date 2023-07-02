use super::{
    conv::Conv,
    dense::Dense,
    pool::{PoolType, Pooling},
};
use crate::matrix::{Matrix, Shape, Shape4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Linear,
    Relu,
    Tanh,
}

#[derive(Debug)]
pub enum Delta {
    Dense { dw: Matrix, db: Vec<f32> },
    Conv { dw: Shape4, db: Vec<f32> },
}

#[derive(Clone)]
pub enum Input {
    Dense(Matrix),
    Conv(Shape4),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer {
    Dense(Dense),
    Conv(Conv),
}

pub trait Prop {
    fn forward_prop(&mut self, back: &Layer, x: &Input);
    fn back_prop(&mut self, back: &Layer, front: Option<&Layer>, y: &Matrix) -> Delta;
    fn apply_delta(&mut self, delta: &Delta, a: f32);
}

impl Activation {
    pub fn getfn(&self) -> impl Fn(f32) -> f32 {
        match self {
            Activation::Linear => |z: f32| z,
            Activation::Sigmoid => |z: f32| 1. / (1. + f32::exp(-z)),
            Activation::Relu => |z: f32| f32::max(0., z),
            Activation::Tanh => {
                |z: f32| (f32::exp(z) - f32::exp(-z)) / (f32::exp(z) + f32::exp(-z))
            }
        }
    }

    pub fn getfn_derivative(&self) -> impl Fn(f32) -> f32 {
        match self {
            Activation::Linear => |_| 1.,
            Activation::Sigmoid => |z: f32| {
                let sigmoid = |z: f32| 1. / (1. + f32::exp(-z));
                sigmoid(z) * (1. - sigmoid(z))
            },
            Activation::Relu => |z: f32| if z > 0. { z } else { 0. },
            Activation::Tanh => |z: f32| {
                let tanh = |z: f32| (f32::exp(z) - f32::exp(-z)) / (f32::exp(z) + f32::exp(-z));
                1. - tanh(z) * tanh(z)
            },
        }
    }
}

impl Input {
    pub fn to_dense(&self) -> Matrix {
        match self {
            Input::Dense(a) => a.clone(),
            Input::Conv(a) => {
                let f: Vec<f32> = a.flatten();
                let mut m: Matrix = Matrix::new(f.len() / a.shape().0, a.shape().0);
                let mut index: usize = 0;
                for c in 0..m.cols() {
                    for r in 0..m.rows() {
                        *m.atref(r, c) = f[index];
                        index += 1;
                    }
                }

                m
            }
        }
    }

    /// Panics if self is not conv.
    pub fn to_conv(&self) -> Shape4 {
        if let Input::Conv(c) = self {
            c.clone()
        } else {
            panic!("Dense -> Conv is unsupported.")
        }
    }
}

impl Layer {
    pub fn dense(n: usize, afn: Activation) -> Self {
        Layer::Dense(Dense::new(n, afn))
    }

    pub fn conv(filters: usize, fshape: (usize, usize), afn: Activation, pooling: Pooling) -> Self {
        Layer::Conv(Conv::new(filters, fshape, afn, pooling))
    }

    pub fn input(x: &Input) -> Self {
        match x {
            Input::Dense(a) => Layer::dense(a.rows(), Activation::Linear),
            Input::Conv(a) => Layer::conv(
                a.shape().1,
                (1, 1),
                Activation::Linear,
                Pooling::new(PoolType::Max, 1, 1),
            ),
        }
    }

    pub fn to_dense(&self) -> Dense {
        match self {
            Layer::Dense(d) => d.clone(),
            Layer::Conv(c) => {
                let a: Matrix = Input::Conv(c.p.clone()).to_dense();
                let mut res: Dense = Dense::new(a.rows(), Activation::Linear);
                res.a = a;

                res
            }
        }
    }

    pub fn to_conv(&self) -> Conv {
        match self {
            Layer::Dense(_) => panic!("Dense to conv not supported."),
            Layer::Conv(c) => c.clone(),
        }
    }

    pub fn apply_delta(&mut self, delta: &Delta, a: f32) {
        match self {
            Layer::Dense(l) => l.apply_delta(delta, a),
            Layer::Conv(l) => l.apply_delta(delta, a),
        }
    }
}
