use super::layer::*;
use crate::matrix::Matrix;
use crate::model::Input;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dense {
    pub n: usize,
    pub(crate) w: Matrix,
    b: Vec<f32>,
    afn: Activation,

    pub a: Matrix,
    z: Matrix,

    pub(crate) dz: Matrix
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

    pub fn adjust_dims(&mut self, bl: &Layer, m: usize) {
        let back_n: usize = match bl {
            Layer::Dense(d) => d.n,
            Layer::Conv(_) => bl.to_dense().a.rows()
        };

        self.w = Matrix::new(self.n, back_n);
        self.w.random_init(-0.2, 0.2);
        self.b.resize(self.n, 0.);

        self.adjust_nonparameter_dims(m);
    }

    pub fn adjust_nonparameter_dims(&mut self, m: usize) {
        self.a = Matrix::new(self.n, m);
        self.z = Matrix::new(self.n, m);
        self.dz = Matrix::new(self.n, m);
    }

    pub fn apply_delta(&mut self, delta: &Delta, a: f32) {
        match delta {
            Delta::Dense { dw, db } => {
                self.w = self.w.clone() - &(dw.clone() * a);
                self.b.iter_mut().zip(db.iter()).for_each(|(b, db): (&mut f32, &f32)| *b -= db * a);
            },
            _ => panic!("Delta type mismatch: Dense layer | {:?} delta", delta)
        }
    }
}

impl Prop for Dense {
    fn forward_prop(&mut self, back: &Layer, x: &Input) {
        let bl: Dense = back.to_dense();

//         for i in 0..864 {
//             if bl.a.at(i, 0) != bl.a.at(i, 1) {
//                 println!("Not equal");
//             }
//         }

        self.a = Matrix::new(self.n, x.to_dense().cols());
        let afn = self.afn.getfn();

        self.z = self.w.clone() * &bl.a;
        self.z = self.z.foreach(|r, c| self.z.at(r, c) + self.b[r]);
        self.a = self.a.foreach(|r, c| self.a.at(r, c) + afn(self.z.at(r, c)));
    }

    fn back_prop(&mut self, back: &Layer, front: Option<&Layer>, y: &Matrix) -> Delta {
        let bl: Dense = back.to_dense();

        if let Some(front) = front {
            let fl: Dense = front.to_dense();

            let afn = self.afn.getfn_derivative();
            let left: Matrix = fl.w.transpose() * &fl.dz;
            let right: Matrix = self.z.foreach(|r, c| afn(self.z.at(r, c)));

            self.dz = left.element_wise_mul(right);
        } else {
            self.dz = self.a.clone() - y;
        }

        let dw: Matrix = self.dz.clone() * &bl.a.transpose() * (1. / y.cols() as f32);
        let mut db: Vec<f32> = Vec::with_capacity(self.dz.cols());
        for i in 0..self.dz.cols() {
            db.push(self.dz.extract_col(i).iter().sum::<f32>() / y.cols() as f32);
        }

        Delta::Dense {
            dw, db
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjust_dims_dense() {
        let mut dense: Dense = Dense::new(10, Activation::Linear);
        let bl: Layer = Layer::dense(5, Activation::Linear);
        dense.adjust_dims(&bl, 2);

        assert_eq!(dense.w.rows(), 10);
        assert_eq!(dense.w.cols(), 5);
        assert_eq!(dense.a.rows(), 10);
        assert_eq!(dense.a.cols(), 2);
        assert_eq!(dense.z.rows(), 10);
        assert_eq!(dense.z.cols(), 2);
        assert_eq!(dense.dz.rows(), 10);
        assert_eq!(dense.dz.cols(), 2);
    }
}

