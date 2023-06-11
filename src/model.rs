use crate::layers::{Layer, Prop, Delta, dense::Dense};
use crate::matrix::{Matrix, Shape4, Shape};

use serde::{Serialize, Deserialize};

use std::fmt;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Debug)]
pub struct Error {
    msg: String
}

impl Error {
    pub fn new(msg: &str) -> Self {
        Self { msg: msg.to_string() }
    }
}

pub enum Input {
    Dense(Matrix),
    Conv(Shape4)
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    layers: Vec<Layer>
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl Input {
    pub fn to_dense(&self) -> Matrix {
        match self {
            Input::Dense(a) => a.clone(),
            Input::Conv(a) => {
                let f: Vec<f32> = a.flatten();
                Matrix::from_1d(f.as_slice(), f.len() / a.shape().0, a.shape().0)
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

impl Model {
    pub fn new() -> Self {
        Self {
            layers: Vec::new()
        }
    }

    pub fn from(path: &str) -> Self {
        let mut file = File::open(path).unwrap_or_else(|_| panic!("File {} doesn't exist.", path));
        let mut data: String = String::new();
        file.read_to_string(&mut data).unwrap();

        serde_json::from_str(data.as_str()).unwrap()
    }

    pub fn train(&mut self, x: &Input, y: &Matrix, epochs: usize, a: f32) {
        self.adjust_layer_dims(x);

        for i in 0..epochs {
            self.forward_prop(x);

            if (i + 1) % 10 == 0 {
                println!("Iteration {} | Cost {}", i + 1, self.cost(y));
            }

            self.back_prop(y, a);
        }
    }

    pub fn predict(&mut self, x: &Input) -> Result<Vec<f32>, Error> {
        self.forward_prop(x);
        if let Some(last) = self.layers.last() {
            Ok(last.to_dense().a.extract_col(0))
        } else {
            Err(Error::new("no layers detected"))
        }
    }

    pub fn save(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        file.write_all(serde_json::to_string(self).unwrap().as_bytes()).unwrap();
    }

    fn adjust_layer_dims(&mut self, x: &Input) {
        self.prepare_layer0(x);

        for i in 1..self.layers.len() {
            let [bl, l, ..] = self.layers[(i - 1)..].as_mut() else { panic!() };

            match l {
                Layer::Dense(d) => d.adjust_dims(bl, x.to_dense().cols()),
                Layer::Conv(c) => c.adjust_dims(bl, x.to_conv().shape().0)
            }
        }
    }

    fn forward_prop(&mut self, x: &Input) {
        self.prepare_layer0(x);

        for i in 1..self.layers.len() {
            let [back, l, ..] = self.layers[(i - 1)..].as_mut() else { panic!() };

            match l {
                Layer::Dense(d) => d.forward_prop(back, x),
                Layer::Conv(c) => c.forward_prop(back, x)
            };
        }
    }

    fn back_prop(&mut self, y: &Matrix, a: f32) {
        let mut deltas: Vec<Delta> = Vec::new();

        let mut is_last_layer: bool = true;
        for i in (1..self.layers.len()).rev() {
            let (back, rest) = self.layers[(i - 1)..].split_first_mut().unwrap();
            let (l, rest) = rest.split_first_mut().unwrap();

            let f: Option<&Layer>;
            if is_last_layer {
                f = None;
                is_last_layer = false;
            } else {
                f = Some(&rest[0]);
            }

            match l {
                Layer::Dense(d) => deltas.insert(0,
                    d.back_prop(back, f, y)
                ),
                Layer::Conv(c) => deltas.insert(0,
                    c.back_prop(back, f, y)
                )
            };
        }

        for i in 1..self.layers.len() {
            self.layers[i].apply_delta(&deltas[i - 1], a);
        }
    }

    fn prepare_layer0(&mut self, x: &Input) {
        match &mut self.layers[0] {
            Layer::Dense(d) => d.a = x.to_dense(),
            Layer::Conv(c) => {
                c.a = x.to_conv();
                c.p = x.to_conv();
            }
        }
    }

    pub fn add(&mut self, l: Layer) {
        self.layers.push(l);
    }

    pub fn cost(&mut self, y: &Matrix) -> f32 {
        let mut sum: f32 = 0.;

        if let Some(last) = self.layers.last() {
            let d: Dense = last.to_dense();
            for r in 0..y.rows() {
                for c in 0..y.cols() {
                    sum += y.at(r, c) * if d.a.at(r, c) == 0. {
                        0.
                    } else {
                        f32::log10(d.a.at(r, c))
                    };
                }
            }
        }

        -sum
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Activation;

    fn get_x() -> Input {
        Input::Dense(Matrix::from(vec![
            vec![0., 0., 1., 1.],
            vec![1., 1., 0., 0.],
        ]))
    }

    fn get_y() -> Matrix {
        Matrix::from(vec![
            vec![1., 1., 0., 0.]
        ])
    }

    fn get_nf() -> usize {
        get_x().to_dense().rows()
    }

    // fn get_m() -> usize {
    //     get_x().cols()
    // }

    fn get_model() -> Model {
        let mut model: Model = Model::new();
        model.add(Layer::dense(get_nf(), Activation::Linear));
        model.add(Layer::dense(4, Activation::Sigmoid));
        model.add(Layer::dense(1, Activation::Sigmoid));
        model
    }

//     #[test]
//     fn dims() {
//         let mut model: Model = get_model();
//         model.adjust_layer_dims(&get_x());

//         if let Layer::Dense(l_input) = &model.layers[0] {
//             if let Layer::Dense(l_hidden) = &model.layers[1] {
//                 if let Layer::Dense(l_output) = &model.layers[2] {
//                     let nf = get_nf();
//                     let m = get_m();
//                     let x = get_x();
//                     let y = get_y();

//                     assert_eq!(l_input.n, nf);
//                     assert_eq!(l_input.a.rows(), x.rows());
//                     assert_eq!(l_input.a.cols(), x.cols());

//                     assert_eq!(l_hidden.w.rows(), 4);
//                     assert_eq!(l_hidden.w.cols(), nf);
//                     assert_eq!(l_hidden.a.rows(), 4);
//                     assert_eq!(l_hidden.a.cols(), m);
//                     assert_eq!(l_hidden.z.rows(), 4);
//                     assert_eq!(l_hidden.z.cols(), m);
//                     assert_eq!(l_hidden.dz.rows(), 4);
//                     assert_eq!(l_hidden.dz.cols(), m);

//                     assert_eq!(l_output.a.rows(), y.rows());
//                     assert_eq!(l_output.a.cols(), y.cols());
//                 }
//             }
//         }
//     }

    #[test]
    fn train_and_predict() {
        let mut model: Model = get_model();
        model.train(&get_x(), &get_y(), 1000, 0.5);

        let prediction: f32 = model.predict(
            &Input::Dense(Matrix::from(vec![
                vec![0.],
                vec![1.]
            ]))
        ).unwrap()[0];

        assert!(prediction > 0.5);
    }
}

