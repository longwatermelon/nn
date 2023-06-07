use crate::matrix::{Matrix, Shape4};

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

pub enum Delta {
    Dense {
        dw: Matrix,
        db: Vec<f32>
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dense {
    pub n: usize,
    pub w: Matrix,
    pub b: Vec<f32>,
    pub afn: Activation,

    pub a: Matrix,
    pub z: Matrix,

    pub dz: Matrix
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conv {
    pub nc: usize,
    pub fh: usize,
    pub fw: usize,
    pub s: usize,
    pub afn: Activation,

    pub w: Shape4,
    pub b: Vec<f32>,

    pub a: Shape4,
    pub z: Shape4,

    pub dz: Shape4
}

pub trait Prop {
    type T;
    fn forward_prop(&mut self, back: &Self::T, x: &Matrix);
    fn back_prop(&mut self, back: &Self::T, front: Option<&Self::T>, y: &Matrix) -> Delta;
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

    pub fn conv(filters: usize, fshape: (usize, usize), afn: Activation) -> Self {
        Layer::Conv(Conv::new(filters, fshape, afn))
    }

    pub fn to_dense(&self) -> &Dense {
        match self {
            Layer::Dense(d) => d,
            Layer::Conv(_c) => todo!()
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
    pub fn new(filters: usize, fshape: (usize, usize), afn: Activation) -> Self {
        Self {
            nc: filters,
            fh: fshape.0,
            fw: fshape.1,
            s: 1,
            afn,
            w: Shape4::default(),
            b: Vec::new(),
            a: Shape4::default(),
            z: Shape4::default(),
            dz: Shape4::default()
        }
    }
}

impl Prop for Dense {
    type T = Dense;
    fn forward_prop(&mut self, back: &Self::T, x: &Matrix) {
        self.a = Matrix::new(self.n, x.cols());
        let afn = self.afn.getfn();

        self.z = self.w.clone() * &back.a;
        self.z = self.z.foreach(|r, c| self.z.at(r, c) + self.b[r]);
        self.a = self.a.foreach(|r, c| self.a.at(r, c) + afn(self.z.at(r, c)));
    }

    fn back_prop(&mut self, back: &Self::T, front: Option<&Self::T>, y: &Matrix) -> Delta {
        if let Some(front) = front {
            let afn = self.afn.getfn_derivative();
            let left: Matrix = front.w.transpose() * &front.dz;
            let right: Matrix = self.z.foreach(|r, c| afn(self.z.at(r, c)));

            self.dz = left.element_wise_mul(right);
        } else {
            self.dz = self.a.clone() - y;
        }

        let dw: Matrix = self.dz.clone() * &back.a.transpose() * (1. / y.cols() as f32);
        let mut db: Vec<f32> = Vec::with_capacity(self.dz.rows());
        for r in self.dz.data() {
            db.push(r.iter().sum());
        }

        Delta::Dense {
            dw, db
        }
    }
}

