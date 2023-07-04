use crate::layers::{Layer, Delta, Input, Prop};
use crate::matrix::Matrix;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rnn {
    na: usize,
    nx: usize,
    wax: Matrix,
    waa: Matrix,
    ba: Vec<f32>,
    by: Vec<f32>,
    pub(crate) a: Matrix,
    x: Matrix,
}

impl Rnn {
    pub fn new(n: usize) -> Self {
        Self {
            na: n,
            nx: 0,
            wax: Matrix::default(),
            waa: Matrix::default(),
            ba: Vec::new(),
            by: Vec::new(),
            a: Matrix::default(),
            x: Matrix::default(),
        }
    }

    pub fn adjust_dims(&mut self, bl: &Layer) {
        todo!()
    }
}

impl Prop for Rnn {
    fn forward_prop(&mut self, back: &Layer, x: &Input) {
        todo!()
    }

    fn back_prop(&mut self, back: &Layer, front: Option<&Layer>, y: &Matrix) -> Delta {
        todo!()
    }

    fn apply_delta(&mut self, delta: &Delta, a: f32) {
        todo!()
    }
}

