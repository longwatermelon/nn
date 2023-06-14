#![allow(clippy::needless_range_loop)]

use super::layer::*;
use super::pool::Pooling;
use crate::matrix::{Matrix, Shape4, Shape};
use crate::model::Input;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conv {
    nc: usize,
    nh: usize,
    nw: usize,
    fh: usize,
    fw: usize,
    afn: Activation,
    pooling: Pooling,

    w: Shape4,
    b: Vec<f32>,

    pub(crate) a: Shape4,
    z: Shape4,
    pub(crate) p: Shape4,
    row_maxes: Vec<Vec<Vec<Vec<usize>>>>,
    col_maxes: Vec<Vec<Vec<Vec<usize>>>>,

    dz: Shape4
}

impl Conv {
    pub fn new(filters: usize, fshape: (usize, usize), afn: Activation, pooling: Pooling) -> Self {
        Self {
            nc: filters,
            nh: 0,
            nw: 0,
            fh: fshape.0,
            fw: fshape.1,
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

    pub fn adjust_dims(&mut self, bl: &Layer, m: usize) {
        let (back_nc, back_nh, back_nw) = match bl {
            Layer::Dense(_) => panic!("Dense -> Conv is unsupported."),
            Layer::Conv(c) => (c.nc, c.a.shape().2, c.a.shape().3)
        };

        self.nh = back_nh - self.fh + 1;
        self.nw = back_nw - self.fw + 1;

        self.w = Shape4::new(self.nc, back_nc, self.fh, self.fw);
        self.w.random_init();
        self.b = vec![0.; self.nc];

        self.adjust_nonparameter_dims(m);
    }

    pub fn adjust_nonparameter_dims(&mut self, m: usize) {
        self.a = Shape4::new(m, self.nc, self.nh, self.nw);
        self.z = self.a.clone();
        self.dz = self.z.clone();

        self.p = Shape4::new(
            self.a.shape().0,
            self.a.shape().1,
            self.a.shape().2 / self.pooling.h,
            self.a.shape().3 / self.pooling.w
        );

        self.row_maxes = vec![vec![vec![vec![0; self.p.shape().3];
                        self.p.shape().2]; self.p.shape().1];
                        self.p.shape().0];
        self.col_maxes = self.row_maxes.clone();
    }

    fn pool(&mut self) {
        for e in 0..self.p.shape().0 {
            for n in 0..self.p.shape().1 {
                let (p, rm, cm) = self.pooling.pool(self.a.at(e).at(n));
                *self.p.at_mut(e).at_mut(n) = p;
                self.row_maxes[e][n] = rm;
                self.col_maxes[e][n] = cm;
            }
        }
    }

    fn db(&mut self, front: &Layer) -> Vec<f32> {
        match front {
            Layer::Dense(fl) => {
                let dlf: Matrix = fl.w.transpose() * &fl.dz;
                let dlp: Shape4 = dlf.reshape_to4(self.p.shape());

                let mut dla: Shape4 = self.a.clone().zero();
                for e in 0..self.p.shape().0 {
                    for n in 0..self.p.shape().1 {
                        for y in 0..self.p.shape().2 {
                            for x in 0..self.p.shape().3 {
                                let (umax, vmax) = (
                                    self.row_maxes[e][n][y][x],
                                    self.col_maxes[e][n][y][x]
                                );

                                *dla.at_mut(e).at_mut(n).atref(umax, vmax)
                                    = dlp.at(e).at(n).at(y, x);
                            }
                        }
                    }
                }

                let gprime = self.afn.getfn_derivative();
                for e in 0..self.dz.shape().0 {
                    for n in 0..self.dz.shape().1 {
                        for u in 0..self.dz.shape().2 {
                            for v in 0..self.dz.shape().3 {
                                *self.dz.at_mut(e).at_mut(n).atref(u, v) =
                                    dla.at(e).at(n).at(u, v) * gprime(self.z.at(e).at(n).at(u, v));
                            }
                        }
                    }
                }

                // Len n_c^l
                let mut dlb: Vec<f32> = vec![0.; self.dz.shape().1];
                for n in 0..dlb.len() {
                    // dlb_n
                    for e in 0..self.dz.shape().0 {
                        for u in 0..self.nh {
                            for v in 0..self.nw {
                                dlb[n] += self.dz.at(e).at(n).at(u, v);
                            }
                        }
                    }
                }

                dlb
            },
            Layer::Conv(fl) => {
                let gprime = self.afn.getfn_derivative();
                let daz: Shape4 = self.z.foreach(|_, m|
                    m.foreach(|i, j| gprime(m.at(i, j)))
                );

                let dzp: Shape4 = Shape4::new(fl.nc, self.nc, fl.fh, fl.fw)
                    .foreach(|(n, c), m|
                        m.foreach(|r, s| fl.w.at(n).at(c).at(r, s))
                    );
                let mut dlp: Shape4 = self.p.clone().zero();
                for e in 0..dlp.shape().0 {
                    for c in 0..dlp.shape().1 {
                        for r in 0..dlp.shape().2 {
                            for s in 0..dlp.shape().3 {
                                // dLP_ecrs
                                for n in 0..fl.nc {
                                    for u in 0..fl.nh {
                                        for v in 0..fl.nw {
                                            if r < dzp.shape().2 && s < dzp.shape().3 {
                                                *dlp.at_mut(e).at_mut(c).atref(r, s) +=
                                                    fl.dz.at(e).at(n).at(u, v) * dzp.at(n).at(c).at(r, s);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let mut dla: Shape4 = self.a.clone().zero();
                for e in 0..self.p.shape().0 {
                    for c in 0..self.p.shape().1 {
                        for r in 0..self.p.shape().2 {
                            for s in 0..self.p.shape().3 {
                                *dla.at_mut(e).at_mut(c).atref(
                                    self.row_maxes[e][c][r][s],
                                    self.col_maxes[e][c][r][s]
                                ) = dlp.at(e).at(c).at(r, s);
                            }
                        }
                    }
                }

                let mut dlb: Vec<f32> = vec![0.; self.nc];
                for e in 0..dla.shape().0 {
                    for c in 0..dlb.len() {
                        for i in 0..self.nh {
                            for j in 0..self.nw {
                                dlb[c] = dla.at(e).at(c).at(i, j) * daz.at(e).at(c).at(i, j);
                            }
                        }
                    }
                }

                dlb
            }
        }
    }

    fn dw(&self, bl: &Conv, front: &Layer, m: usize) -> Shape4 {
        let dzw_assign_to: &Shape4 = match front {
            Layer::Dense(_) => &bl.p,
            Layer::Conv(_) => &bl.a
        };

        // dzw[p][q].at(e).at(c).at(u, v)
        let mut dzw: Vec<Vec<Shape4>> = vec![
            vec![
                Shape4::new(m, bl.nc, self.nh, self.nw); self.fw
            ]; self.fh
        ];
        for p in 0..dzw.len() {
            for q in 0..dzw[0].len() {
                for e in 0..m {
                    for c in 0..bl.nc {
                        for u in 0..self.nh {
                            for v in 0..self.nw {
                                *dzw[p][q].at_mut(e).at_mut(c).atref(u, v) =
                                    dzw_assign_to.at(e).at(c).at(p + u, q + v);
                            }
                        }
                    }
                }
            }
        }

        let mut dlw: Shape4 = Shape4::new(self.nc, bl.nc, self.fh, self.fw);
        for n in 0..self.nc {
            for c in 0..bl.nc {
                for p in 0..self.fh {
                    for q in 0..self.fw {
                        // dLW_ncpq
                        for e in 0..m {
                            *dlw.at_mut(n).at_mut(c).atref(p, q) +=
                                (0..self.nh).zip(0..self.nw)
                                .map(|(u, v)| self.dz.at(e).at(n).at(u, v) * dzw[p][q].at(e).at(c).at(u, v))
                                .sum::<f32>();
                        }
                    }
                }
            }
        }

        dlw
    }

    pub fn apply_delta(&mut self, delta: &Delta, a: f32) {
        match delta {
            Delta::Conv { dw, db } => {
                for n in 0..self.w.shape().0 {
                    for c in 0..self.w.shape().1 {
                        *self.w.at_mut(n).at_mut(c) = self.w.at(n).at(c).clone() - &(dw.at(n).at(c).clone() * a);
                    }
                }

                self.b.iter_mut().zip(db.iter()).for_each(|(b, db)| *b -= db * a);
            },
            _ => panic!("Delta type mismatch: Conv layer | {:?} delta", delta)
        }
    }
}

impl Prop for Conv {
    fn forward_prop(&mut self, back: &Layer, x: &Input) {
        let m: usize = x.to_conv().shape().0;
        let bl: Conv = back.to_conv();

        for e in 0..m {
            for n in 0..self.nc {
                let convolved: Matrix = convolve(&bl.p, &self.w, e, n);
                let z: Matrix = convolved.clone() + self.b[n];

                *self.z.at_mut(e).at_mut(n) = z;
            }
        }

        let afn = self.afn.getfn();
        for e in 0..m {
            for n in 0..self.nc {
                *self.a.at_mut(e).at_mut(n) =
                    self.z.at(e).at(n).foreach(|r, c| afn(self.z.at(e).at(n).at(r, c)));
            }
        }

        self.pool();
    }

    fn back_prop(&mut self, back: &Layer, front: Option<&Layer>, y: &Matrix) -> Delta {
        // db must be calculated before dw, because db sets self.dz
        let db: Vec<f32> = self.db(front.unwrap());
        let dw: Shape4 = self.dw(&back.to_conv(), front.unwrap(), y.cols());

        Delta::Conv { dw, db }
    }
}

fn convolve(input: &Shape4, filter: &Shape4, e: usize, n: usize) -> Matrix {
    let mut sum: Matrix = Matrix::new(
        input.shape().2 - filter.shape().2 + 1,
        input.shape().3 - filter.shape().3 + 1
    );

    for c in 0..filter.shape().1 {
        let convolved = input.at(e).at(c).convolve(filter.at(n).at(c));
        sum = sum + &convolved;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Shape3;
    use super::super::pool::PoolType;

    #[test]
    fn adjust_dims_conv() {
        let mut conv: Conv = Conv::new(6, (5, 5), Activation::Linear,
                                       Pooling::new(PoolType::Max, 2, 2));
        let mut bl: Layer = Layer::conv(3, (1, 1), Activation::Linear, Pooling::new(PoolType::Max, 2, 2));
        if let Layer::Conv(bl) = &mut bl {
            bl.a = Shape4::new(1, 3, 32, 32);
        }

        conv.adjust_dims(&bl, 1);

        assert_eq!(conv.a.shape(), (1, 6, 28, 28));
        assert_eq!(conv.p.shape(), (1, 6, 14, 14));
        assert_eq!(conv.w.shape(), (6, 3, 5, 5));
    }

    #[test]
    fn conv_maxpool() {
        let mut conv: Conv = Conv::new(6, (5, 5), Activation::Linear,
                                        Pooling::new(PoolType::Max, 2, 2));
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
            conv.a.shape().2 / conv.pooling.h,
            conv.a.shape().3 / conv.pooling.w
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

    #[test]
    fn forward_prop() {
        let mut l: Layer = Layer::conv(1, (3, 3), Activation::Linear, Pooling::new(PoolType::Max, 2, 2));
        let mut prev_l: Layer = Layer::conv(1, (1, 1), Activation::Linear, Pooling::new(PoolType::Max, 2, 2));
        let x: Input = Input::Conv(Shape4::from(
            vec![
                Shape3::from(
                    vec![
                        Matrix::from(
                            vec![
                                vec![1., 1., 1., 0., 0., 0.],
                                vec![1., 1., 1., 0., 0., 0.],
                                vec![1., 1., 1., 0., 0., 0.],
                                vec![1., 1., 1., 0., 0., 0.],
                                vec![1., 1., 1., 0., 0., 0.],
                                vec![1., 1., 1., 0., 0., 0.]
                            ]
                        )
                    ]
                ),
                Shape3::from(
                    vec![
                        Matrix::from(
                            vec![
                                vec![0., 0., 0., 1., 1., 1.],
                                vec![0., 0., 0., 1., 1., 1.],
                                vec![0., 0., 0., 1., 1., 1.],
                                vec![0., 0., 0., 1., 1., 1.],
                                vec![0., 0., 0., 1., 1., 1.],
                                vec![0., 0., 0., 1., 1., 1.]
                            ]
                        )
                    ]
                )
            ]
        ));

        if let Layer::Conv(c) = &mut prev_l {
            c.a = x.to_conv();
            c.p = x.to_conv();
        }

        if let Layer::Conv(c) = &mut l {
            c.adjust_dims(&prev_l, 2);
            c.b.iter_mut().for_each(|b| *b = 1.);
            c.w = Shape4::from(vec![
                Shape3::from(
                    vec![
                        Matrix::from(vec![
                            vec![1., 0., -1.],
                            vec![1., 0., -1.],
                            vec![1., 0., -1.]
                        ])
                        // Matrix::from(vec![
                        //     vec![1., 0., -1.],
                        //     vec![1., 0., -1.],
                        //     vec![1., 0., -1.]
                        // ])
                    ]
                )
            ]);

            c.forward_prop(&prev_l, &x);

            assert_eq!(c.a, Shape4::from(
                vec![
                    Shape3::from(
                        vec![
                            Matrix::from(
                                vec![
                                    vec![1., 4., 4., 1.],
                                    vec![1., 4., 4., 1.],
                                    vec![1., 4., 4., 1.],
                                    vec![1., 4., 4., 1.]
                                ]
                            ),
                        ]
                    ),
                    Shape3::from(
                        vec![
                            Matrix::from(
                                vec![
                                    vec![1., -2., -2., 1.],
                                    vec![1., -2., -2., 1.],
                                    vec![1., -2., -2., 1.],
                                    vec![1., -2., -2., 1.]
                                ]
                            ),
                        ]
                    )
                ]
            ));
        }
    }

    #[test]
    fn delta() {
        let mut layer: Conv = Conv::new(1, (1, 1), Activation::Linear, Pooling::new(PoolType::Max, 2, 2));
        layer.w = Shape4::from(
            vec![
                Shape3::from(
                    vec![
                        Matrix::from(
                            vec![
                                vec![0., 0.],
                                vec![0., 0.]
                            ]
                        )
                    ]
                )
            ]
        );

        layer.b = vec![0.; 10];

        let dw: Shape4 = Shape4::from(
            vec![
                Shape3::from(
                    vec![
                        Matrix::from(
                            vec![
                                vec![2., 2.],
                                vec![2., 2.]
                            ]
                        )
                    ]
                )
            ]
        );

        let db: Vec<f32> = vec![1.; 10];

        let delta: Delta = Delta::Conv { dw: dw.clone(), db: db.clone() };
        layer.apply_delta(&delta, 1.);

        assert_eq!(layer.w, Shape4::from(
            vec![
                Shape3::from(
                    vec![
                        Matrix::from(
                            vec![
                                vec![-2., -2.],
                                vec![-2., -2.]
                            ]
                        )
                    ]
                )
            ]
        ));
        assert_eq!(layer.b, vec![-1.; 10]);
    }
}

