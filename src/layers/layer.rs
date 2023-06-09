use super::dense::Dense;
use super::conv::{Conv, Pooling};
use crate::matrix::{Matrix, Shape4};

use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Linear,
    Relu,
    Tanh
}

#[derive(Debug)]
pub enum Delta {
    Dense {
        dw: Matrix,
        db: Vec<f32>
    },
    Conv {
        dw: Shape4,
        db: Vec<f32>
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer {
    Dense(Dense),
    Conv(Conv)
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
        match self {
            Layer::Dense(l) => l.apply_delta(delta, a),
            Layer::Conv(l) => l.apply_delta(delta, a)
        }
    }
}

