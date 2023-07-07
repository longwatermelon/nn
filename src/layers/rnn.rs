// Temporary
#![allow(unused)]

use crate::layers::{Layer, Delta, Input, Prop};
use crate::matrix::{Matrix, Shape3, Shape};
use crate::util;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rnn {
    na: usize,
    nx: usize,
    ny: usize,
    wax: Matrix,
    waa: Matrix,
    wya: Matrix,
    ba: Vec<f32>,
    by: Vec<f32>,
    pub(crate) a: Shape3,
    x: Shape3,
    y: Shape3,
}

impl Rnn {
    pub fn new(n: usize) -> Self {
        Self {
            na: n,
            nx: 0,
            ny: 0,
            wax: Matrix::default(),
            waa: Matrix::default(),
            wya: Matrix::default(),
            ba: Vec::new(),
            by: Vec::new(),
            a: Shape3::default(),
            x: Shape3::default(),
            y: Shape3::default(),
        }
    }

    pub fn adjust_dims(&mut self, ny: usize) {
        self.wax = Matrix::new(self.na, self.nx);
        self.waa = Matrix::new(self.na, self.na);
        self.wya = Matrix::new(ny, self.na);
        self.wax.random_init(-1., 1.);
        self.waa.random_init(-1., 1.);
        self.wya.random_init(-1., 1.);

        self.ba = vec![0.; self.na];
        self.by = vec![0.; ny];
    }

    pub fn prepare_nonparam(&mut self, nx: usize, ny: usize, m: usize, tx: usize) {
        self.nx = nx;
        self.ny = ny;
        self.x = Shape3::new(nx, m, tx);
        self.y = Shape3::new(ny, m, tx);
        self.a = Shape3::new(self.na, m, tx);
    }

    fn cell_forward(&mut self, x: Shape3, prev_a: Matrix, t: usize) {
        // Constant is x.shape().2 or t
        let mut xt: Matrix = Matrix::new(x.shape().0, x.shape().1);
        for n in 0..xt.rows() {
            for e in 0..xt.cols() {
                *xt.atref(n, e) = x.at(n).at(e, t);
            }
        }

        for n in 0..self.x.shape().0 {
            for e in 0..self.x.shape().1 {
                *self.x.at_mut(n).atref(e, t) = xt.at(n, e);
            }
        }

        // a = waa * a<l-1> + wax * x<t>
        // a dims = n_a x m

        let prod: Matrix = self.waa.clone() * prev_a + self.wax.clone() * xt;
        for n in 0..self.a.shape().0 {
            for e in 0..self.a.shape().1 {
                *self.a.at_mut(n).atref(e, t) = prod.at(n, e);
            }
        }
        // self.a = ;

        // a = a + b
        // 0 to m
        for e in 0..self.a.shape().1 {
            for n in 0..self.a.shape().0 {
                *self.a.at_mut(n).atref(e, t) += self.ba[n];
            }
        }
        // for c in 0..self.a.cols() {
        //     // 0 to n_a
        //     for r in 0..self.a.rows() {
        //         *self.a.atref(r, c) += self.ba[r];
        //     }
        // }

        // a = tanh(a)
        for n in 0..self.a.shape().0 {
            for e in 0..self.a.shape().1 {
                *self.a.at_mut(n).atref(e, t) = f32::tanh(self.a.at(n).at(e, t));
            }
        }
        // self.a = self.a.foreach(|r, c| f32::tanh(self.a.at(r, c)));

        let mut at: Matrix = Matrix::new(self.a.shape().0, self.a.shape().1);
        for n in 0..at.rows() {
            for e in 0..at.cols() {
                *at.atref(n, e) = self.a.at(n).at(e, t);
            }
        }

        // y<t> = wya * a<t>
        let prod: Matrix = self.wya.clone() * at;
        for n in 0..self.y.shape().0 {
            for e in 0..self.y.shape().1 {
                *self.y.at_mut(n).atref(e, t) = prod.at(n, e);
            }
        }

        // Iter over examples
        for e in 0..self.y.shape().1 {
            // Add by[n] where n = 0..ny for each ex
            for n in 0..self.y.shape().0 {
                *self.y.at_mut(n).atref(e, t) += self.by[n];
            }
        }

        // y<t> = softmax(y<t>)
        for e in 0..self.y.shape().1 {
            let mut y: Vec<f32> = vec![0.; self.ny];
            for n in 0..self.ny {
                y[n] = self.y.at(n).at(e, t);
            }

            let softmax_y: Vec<f32> = util::softmax(&y);

            for n in 0..self.y.shape().0 {
                *self.y.at_mut(n).atref(e, t) = softmax_y[n];
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_cell() {
        let mut l: Rnn = Rnn::new(5);
        l.waa = Matrix::from(
            vec![
                vec![-0.22232814, -0.20075807,  0.18656139,  0.41005165,  0.19829972],
                vec![ 0.11900865, -0.67066229,  0.37756379,  0.12182127,  1.12948391],
                vec![ 1.19891788,  0.18515642, -0.37528495, -0.63873041,  0.42349435],
                vec![ 0.07734007, -0.34385368,  0.04359686, -0.62000084,  0.69803203],
                vec![-0.44712856,  1.2245077 ,  0.40349164,  0.59357852, -1.09491185],
            ]
        );

        l.wax = Matrix::from(
            vec![
                vec![ 0.16938243,  0.74055645, -0.9537006 ],
                vec![-0.26621851,  0.03261455, -1.37311732],
                vec![ 0.31515939,  0.84616065, -0.85951594],
                vec![ 0.35054598, -1.31228341, -0.03869551],
                vec![-1.61577235,  1.12141771,  0.40890054],
            ]
        );

        l.wya = Matrix::from(
            vec![
                vec![-0.02461696, -0.77516162,  1.27375593,  1.96710175, -1.85798186],
                vec![ 1.23616403,  1.62765075,  0.3380117 , -1.19926803,  0.86334532],
            ]
        );

        l.ba = vec![
            -0.1809203,
            -0.60392063,
            -1.23005814,
            0.5505375,
            0.79280687,
        ];

        l.by = vec![
            -0.62353073,
            0.52057634
        ];

        // let xt: Matrix = Matrix::from(
        //     vec![
        //         vec![1.6243453636632417, -0.6117564136500754, -0.5281717522634557, -1.0729686221561705, 0.8654076293246785, -2.3015386968802827, 1.74481176421648, -0.7612069008951028, 0.31903909605709857, -0.2493703754774101],
        //         vec![1.462107937044974, -2.060140709497654, -0.3224172040135075, -0.38405435466841564, 1.1337694423354374, -1.0998912673140309, -0.17242820755043575, -0.8778584179213718, 0.04221374671559283, 0.5828152137158222],
        //         vec![-1.1006191772129212, 1.1447237098396141, 0.9015907205927955, 0.5024943389018682, 0.9008559492644118, -0.6837278591743331, -0.12289022551864817, -0.9357694342590688, -0.2678880796260159, 0.530355466738186],
        //     ]
        // );

        // x is (nx, m, tx) = (3, 10, 1)
        let x: Shape3 = Shape3::from(
            vec![
                Matrix::from(vec![vec![1.6243453636632417], vec![ -0.6117564136500754], vec![ -0.5281717522634557], vec![ -1.0729686221561705], vec![ 0.8654076293246785], vec![ -2.3015386968802827], vec![ 1.74481176421648], vec![ -0.7612069008951028], vec![ 0.31903909605709857], vec![ -0.2493703754774101]]),
                Matrix::from(vec![vec![1.462107937044974], vec![ -2.060140709497654], vec![ -0.3224172040135075], vec![ -0.38405435466841564], vec![ 1.1337694423354374], vec![ -1.0998912673140309], vec![ -0.17242820755043575], vec![ -0.8778584179213718], vec![ 0.04221374671559283], vec![ 0.5828152137158222]]),
                Matrix::from(vec![vec![-1.1006191772129212], vec![ 1.1447237098396141], vec![ 0.9015907205927955], vec![ 0.5024943389018682], vec![ 0.9008559492644118], vec![ -0.6837278591743331], vec![ -0.12289022551864817], vec![ -0.9357694342590688], vec![ -0.2678880796260159], vec![ 0.530355466738186]]),
            ]
        );

        let prev_a: Matrix = Matrix::from(
            vec![
                vec![-0.691660751725309, -0.39675352685597737, -0.6871727001195994, -0.8452056414987196, -0.671246130836819, -0.01266459891890136, -1.1173103486352778, 0.23441569781709215, 1.6598021771098705, 0.7420441605773356],
                vec![-0.19183555236161492, -0.8876289640848363, -0.7471582937508376, 1.6924546010277466, 0.05080775477602897, -0.6369956465693534, 0.19091548466746602, 2.100255136478842, 0.12015895248162915, 0.6172031097074192],
                vec![0.3001703199558275, -0.35224984649351865, -1.1425181980221402, -0.3493427224128775, -0.2088942333747781, 0.5866231911821976, 0.8389834138745049, 0.9311020813035573, 0.2855873252542588, 0.8851411642707281],
                vec![-0.7543979409966528, 1.2528681552332879, 0.5129298204180088, -0.29809283510271567, 0.48851814653749703, -0.07557171302105573, 1.131629387451427, 1.5198168164221988, 2.1855754065331614, -1.3964963354881377],
                vec![-1.4441138054295894, -0.5044658629464512, 0.16003706944783047, 0.8761689211162249, 0.31563494724160523, -2.022201215824003, -0.3062040126283718, 0.8279746426072462, 0.2300947353643834, 0.7620111803120247],
            ]
        );

        // println!("waa = {}", l.waa.dims());
        // println!("wya = {}", l.wya.dims());
        // println!("wax = {}", l.wax.dims());
        // println!("ba = {}", l.ba.len());
        // println!("by = {}", l.by.len());
        // println!("xt = {}", xt.dims());
        // println!("prev_a = {}", prev_a.dims());

        l.prepare_nonparam(3, 2, 10, 1);
        l.cell_forward(x, prev_a, 0);

        let mut at: Matrix = Matrix::new(l.a.shape().0, l.a.shape().1);
        for n in 0..l.a.shape().0 {
            for e in 0..l.a.shape().1 {
                *at.atref(n, e) = l.a.at(n).at(e, 0);
            }
        }
        assert_eq!(at.extract_row(4), vec![0.59584534, 0.18141817, 0.61311865, 0.99808216, 0.850162, 0.9998098, -0.1888717, 0.99815553, 0.65311515, 0.8287204]);

        let mut yt: Matrix = Matrix::new(l.y.shape().0, l.y.shape().1);
        for n in 0..l.y.shape().0 {
            for e in 0..l.y.shape().1 {
                *yt.atref(n, e) = l.y.at(n).at(e, 0);
            }
        }
        assert_eq!(yt.extract_row(1), vec![0.988816, 0.016820231, 0.21140899, 0.36817473, 0.98988384, 0.88945216, 0.36920208, 0.9966312, 0.99825585, 0.17746533]);
    }
}

