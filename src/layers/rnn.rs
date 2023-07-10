use crate::layers::{Layer, Delta, Input, Prop};
use crate::matrix::{Matrix, Shape3, Shape};
use serde::{Serialize, Deserialize};

/// Uses tanh activation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rnn {
    pub na: usize,
    nx: usize,
    wax: Matrix,
    waa: Matrix,
    ba: Vec<f32>,
    pub(crate) a: Shape3,
    x: Shape3,
    a0: Matrix,
}

impl Rnn {
    pub fn new(n: usize) -> Self {
        Self {
            na: n,
            nx: 0,
            wax: Matrix::default(),
            waa: Matrix::default(),
            ba: Vec::new(),
            a: Shape3::default(),
            x: Shape3::default(),
            a0: Matrix::default(),
        }
    }

    pub fn adjust_dims(&mut self) {
        self.wax = Matrix::new(self.na, self.nx);
        self.waa = Matrix::new(self.na, self.na);
        self.wax.random_init(-1., 1.);
        self.waa.random_init(-1., 1.);

        self.ba = vec![0.; self.na];
    }

    pub fn prepare_nonparam(&mut self, nx: usize, m: usize, tx: usize) {
        self.nx = nx;
        self.x = Shape3::new(nx, m, tx);
        self.a = Shape3::new(self.na, m, tx);
        self.a0 = Matrix::new(self.na, m);
    }

    pub fn result(&self) -> Matrix {
        self.a.index_last(self.a.shape().2 - 1)
    }

    fn cell_forward(&mut self, x: &Shape3, prev_a: &Matrix, t: usize) {
        let xt: Matrix = x.index_last(t);
        self.x.assign_last(t, &xt);

        // a = waa * a<l-1> + wax * x<t>
        // a dims = n_a x m
        let prod: Matrix = self.waa.clone() * prev_a.clone() + self.wax.clone() * xt;
        self.a.assign_last(t, &prod);

        // a = a + b
        // 0 to m
        for e in 0..self.a.shape().1 {
            for n in 0..self.a.shape().0 {
                *self.a.at_mut(n).atref(e, t) += self.ba[n];
            }
        }

        // a<t> = tanh(a<t>)
        self.a.foreach_t(t, f32::tanh);
    }

    /// Returns da_front for the next cell_back call
    fn cell_back(&mut self, da_front: Matrix, t: usize, delta: &mut Delta) -> Matrix {
        let a_prev: Matrix = if t == 0 { self.a0.clone() } else { self.a.index_last(t - 1) };
        let xt: Matrix = self.x.index_last(t);

        let mut prod: Matrix = self.wax.clone() * xt.clone() +
                           self.waa.clone() * a_prev.clone();
        for e in 0..prod.cols() {
            for n in 0..prod.rows() {
                *prod.atref(n, e) += self.ba[n];
            }
        }

        let tanh: Matrix = prod.foreach(|r, c| 1. - f32::tanh(prod.at(r, c)).powi(2));
        let dtanh: Matrix = da_front.element_wise_mul(tanh);

        let dwax: Matrix = dtanh.clone() * xt.transpose();
        let dwaa: Matrix = dtanh.clone() * a_prev.transpose();

        let mut dba: Vec<f32> = vec![0.; dtanh.rows()];
        for i in 0..dba.len() {
            dba[i] = dtanh.extract_row(i).iter().sum();
        }

        let Delta::Rnn { dwax: ddwax, dwaa: ddwaa, dba: ddba } = delta else { unreachable!() };
        *ddwax = ddwax.clone() + dwax;
        *ddwaa = ddwaa.clone() + dwaa;
        *ddba = ddba.iter().zip(dba.iter()).map(|(a, b)| a + b).collect();

        self.waa.transpose() * dtanh
    }
}

impl Prop for Rnn {
    fn forward_prop(&mut self, _back: &Layer, x: &Input) {
        let x: Shape3 = x.to_rnn();
        // let mut prev_a: Matrix = Matrix::new(self.na, self.a.shape().1);
        let mut prev_a: Matrix = self.a0.clone();
        for t in 0..x.shape().2 {
            self.cell_forward(&x, &prev_a, t);

            for n in 0..self.a.shape().0 {
                for e in 0..self.a.shape().1 {
                    *prev_a.atref(n, e) = self.a.at(n).at(e, t);
                }
            }
        }
    }

    fn back_prop(&mut self, _back: &Layer, _front: Option<&Layer>, _y: &Matrix) -> Delta {
        let dwax: Matrix = Matrix::new(self.na, self.nx);
        let dwaa: Matrix = Matrix::new(self.na, self.na);
        let dba: Vec<f32> = vec![0.; self.na];
        let mut delta: Delta = Delta::Rnn { dwax, dwaa, dba };

        let mut da_front: Matrix = self.a.index_last(0).foreach(|_, _| 0.);

        for t in (0..self.x.shape().2).rev() {
            da_front = self.cell_back(da_front, t, &mut delta);
        }

        delta
    }

    fn apply_delta(&mut self, delta: &Delta, a: f32) {
        let Delta::Rnn { dwax, dwaa, dba } = &delta else { unreachable!() };
        self.wax = self.wax.clone() + dwax.clone() * a;
        self.waa = self.waa.clone() + dwaa.clone() * a;
        self.ba = self.ba.iter().zip(dba.iter())
                         .map(|(a, b)| a + b)
                         .collect();
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

        l.ba = vec![
            -0.1809203,
            -0.60392063,
            -1.23005814,
            0.5505375,
            0.79280687,
        ];

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

        l.prepare_nonparam(3, 10, 1);
        l.cell_forward(&x, &prev_a, 0);

        let mut at: Matrix = Matrix::new(l.a.shape().0, l.a.shape().1);
        for n in 0..l.a.shape().0 {
            for e in 0..l.a.shape().1 {
                *at.atref(n, e) = l.a.at(n).at(e, 0);
            }
        }
        assert_eq!(at.extract_row(4), vec![0.59584534, 0.18141817, 0.61311865, 0.99808216, 0.850162, 0.9998098, -0.1888717, 0.99815553, 0.65311515, 0.8287204]);
    }

    #[test]
    fn forward_prop() {
        let nx: usize = 3;
        let na: usize = 5;
        let m: usize = 10;

        let mut l: Rnn = Rnn::new(na);

        l.waa = Matrix::from(
            vec![
                vec![-0.64691669,  0.90148689,  2.52832571, -0.24863478,  0.04366899],
                vec![-0.22631424,  1.33145711, -0.28730786,  0.68006984, -0.3198016 ],
                vec![-1.27255876,  0.31354772,  0.50318481,  1.29322588, -0.11044703],
                vec![-0.61736206,  0.5627611 ,  0.24073709,  0.28066508, -0.0731127 ],
                vec![ 1.16033857,  0.36949272,  1.90465871,  1.1110567 ,  0.6590498 ],
            ]
        );

        l.wax = Matrix::from(
            vec![
                vec![-1.62743834,  0.60231928,  0.4202822 ],
                vec![ 0.81095167,  1.04444209, -0.40087819],
                vec![ 0.82400562, -0.56230543,  1.95487808],
                vec![-1.33195167, -1.76068856, -1.65072127],
                vec![-0.89055558, -1.1191154 ,  1.9560789 ],
            ]
        );

        l.ba = vec![ 1.6169496, 0.50274088, 1.55880554, 0.1094027 , -1.2197444 ];

        let x: Shape3 = Shape3::from(
            vec![
                Matrix::from(vec![
                    vec![1.6243453636632417,
                   -0.6117564136500754,
                   -0.5281717522634557,
                   -1.0729686221561705],
                  vec![0.8654076293246785,
                   -2.3015386968802827,
                   1.74481176421648,
                   -0.7612069008951028],
                  vec![0.31903909605709857,
                   -0.2493703754774101,
                   1.462107937044974,
                   -2.060140709497654],
                  vec![-0.3224172040135075,
                   -0.38405435466841564,
                   1.1337694423354374,
                   -1.0998912673140309],
                  vec![-0.17242820755043575,
                   -0.8778584179213718,
                   0.04221374671559283,
                   0.5828152137158222],
                  vec![-1.1006191772129212,
                   1.1447237098396141,
                   0.9015907205927955,
                   0.5024943389018682],
                  vec![0.9008559492644118,
                   -0.6837278591743331,
                   -0.12289022551864817,
                   -0.9357694342590688],
                  vec![-0.2678880796260159,
                   0.530355466738186,
                   -0.691660751725309,
                   -0.39675352685597737],
                  vec![-0.6871727001195994,
                   -0.8452056414987196,
                   -0.671246130836819,
                   -0.01266459891890136],
                  vec![-1.1173103486352778,
                   0.23441569781709215,
                   1.6598021771098705,
                   0.7420441605773356]
                ]),
                Matrix::from(vec![
                    vec![-0.19183555236161492,
                   -0.8876289640848363,
                   -0.7471582937508376,
                   1.6924546010277466],
                  vec![0.05080775477602897,
                   -0.6369956465693534,
                   0.19091548466746602,
                   2.100255136478842],
                  vec![0.12015895248162915,
                   0.6172031097074192,
                   0.3001703199558275,
                   -0.35224984649351865],
                  vec![-1.1425181980221402,
                   -0.3493427224128775,
                   -0.2088942333747781,
                   0.5866231911821976],
                  vec![0.8389834138745049,
                   0.9311020813035573,
                   0.2855873252542588,
                   0.8851411642707281],
                  vec![-0.7543979409966528,
                   1.2528681552332879,
                   0.5129298204180088,
                   -0.29809283510271567],
                  vec![0.48851814653749703,
                   -0.07557171302105573,
                   1.131629387451427,
                   1.5198168164221988],
                  vec![2.1855754065331614,
                   -1.3964963354881377,
                   -1.4441138054295894,
                   -0.5044658629464512],
                  vec![0.16003706944783047,
                   0.8761689211162249,
                   0.31563494724160523,
                   -2.022201215824003],
                  vec![-0.3062040126283718,
                   0.8279746426072462,
                   0.2300947353643834,
                   0.7620111803120247]
                ]),
                Matrix::from(vec![
                    vec![-0.22232814261035927,
                   -0.20075806892999745,
                   0.1865613909882843,
                   0.4100516472082563],
                  vec![0.19829972012676975,
                   0.11900864580745882,
                   -0.6706622862890306,
                   0.3775637863209194],
                  vec![0.12182127099143693,
                   1.1294839079119197,
                   1.198917879901507,
                   0.18515641748394385],
                  vec![-0.3752849500901142,
                   -0.6387304074542224,
                   0.4234943540641129,
                   0.07734006834855942],
                  vec![-0.3438536755710756,
                   0.04359685683424694,
                   -0.6200008439481293,
                   0.6980320340722189],
                  vec![-0.4471285647859982, 1.2245077048054989, 0.4034916417908, 0.593578523237067],
                  vec![-1.0949118457410418,
                   0.1693824330586681,
                   0.7405564510962748,
                   -0.9537006018079346],
                  vec![-0.26621850600362207,
                   0.03261454669335856,
                   -1.3731173202467557,
                   0.31515939204229176],
                  vec![0.8461606475850334,
                   -0.8595159408319863,
                   0.35054597866410736,
                   -1.3122834112374318],
                  vec![-0.038695509266051115,
                   -1.6157723547032947,
                   1.121417708235664,
                   0.4089005379368278]
                ])
            ]
        );

        l.a0 = Matrix::from(
            vec![
                vec![-0.024616955875778355, -0.7751616191691596, 1.2737559301587766, 1.9671017492547347, -1.857981864446752, 1.2361640304528203, 1.6276507531489064, 0.3380116965744758, -1.199268032335186, 0.8633453175440216],
                vec![-0.18092030207815046, -0.6039206277932573, -1.2300581356669618, 0.5505374959762154, 0.7928068659193477, -0.6235307296797916, 0.5205763370733708, -1.1443413896231427, 0.8018610318713447, 0.04656729842414554],
                vec![-0.18656977190734877, -0.10174587252914521, 0.8688861570058679, 0.7504116398650081, 0.5294653243527092, 0.13770120999738608, 0.07782112791270591, 0.6183802619985245, 0.23249455917873788, 0.6825514068644851],
                vec![-0.31011677351806, -2.434837764107139, 1.038824601859414, 2.1869796469742577, 0.44136444356858207, -0.10015523328349978, -0.13644474389603303, -0.11905418777480989, 0.0174094083000046, -1.1220187287468883],
                vec![-0.5170944579202279, -0.997026827650263, 0.2487991613877705, -0.29664115237086275, 0.4952113239779604, -0.17470315974250095, 0.9863351878212421, 0.2135339013354418, 2.1906997289697334, -1.8963609228910925],
            ]
        );

        let tmp: Rnn = Rnn::new(1);
        // l.adjust_dims(ny);
        l.prepare_nonparam(nx, m, 4);
        l.forward_prop(&Layer::Rnn(tmp), &Input::Rnn(x));

        let mut ane: Vec<f32> = vec![0.; 4];
        for t in 0..4 {
            ane[t] = l.a.at(4).at(1, t);
        }

        assert_eq!(ane, vec![-0.99999375,  0.77911205, -0.99861469, -0.99833267]);
    }

    #[test]
    fn back_cell() {
        let nx: usize = 3;
        let na: usize = 5;
        let m: usize = 10;
        let tx: usize = 1;

        let mut l: Rnn = Rnn::new(na);

        l.wax = Matrix::from(
            vec![
                vec![-0.22232814, -0.20075807,  0.18656139],
                vec![ 0.41005165,  0.19829972,  0.11900865],
                vec![-0.67066229,  0.37756379,  0.12182127],
                vec![ 1.12948391,  1.19891788,  0.18515642],
                vec![-0.37528495, -0.63873041,  0.42349435],
            ]
        );

        l.waa = Matrix::from(
            vec![
                vec![ 0.07734007, -0.34385368,  0.04359686, -0.62000084,  0.69803203],
                vec![-0.44712856,  1.2245077 ,  0.40349164,  0.59357852, -1.09491185],
                vec![ 0.16938243,  0.74055645, -0.9537006 , -0.26621851,  0.03261455],
                vec![-1.37311732,  0.31515939,  0.84616065, -0.85951594,  0.35054598],
                vec![-1.31228341, -0.03869551, -1.61577235,  1.12141771,  0.40890054],
            ]
        );

        l.ba = vec![-0.1809203, -0.60392063, -1.23005814, 0.5505375, 0.79280687];

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

        l.prepare_nonparam(nx, m, tx);
        l.cell_forward(&x, &prev_a, 0);

        let da_next: Matrix = Matrix::from(
            vec![
                vec![-1.1443413896231427, 0.8018610318713447, 0.04656729842414554, -0.18656977190734877, -0.10174587252914521, 0.8688861570058679, 0.7504116398650081, 0.5294653243527092, 0.13770120999738608, 0.07782112791270591],
                vec![0.6183802619985245, 0.23249455917873788, 0.6825514068644851, -0.31011677351806, -2.434837764107139, 1.038824601859414, 2.1869796469742577, 0.44136444356858207, -0.10015523328349978, -0.13644474389603303],
                vec![-0.11905418777480989, 0.0174094083000046, -1.1220187287468883, -0.5170944579202279, -0.997026827650263, 0.2487991613877705, -0.29664115237086275, 0.4952113239779604, -0.17470315974250095, 0.9863351878212421],
                vec![0.2135339013354418, 2.1906997289697334, -1.8963609228910925, -0.646916688254908, 0.901486891648711, 2.528325706806398, -0.24863477771546005, 0.043668993178389105, -0.22631424251360518, 1.3314571125875918],
                vec![-0.2873078634760189, 0.6800698398781045, -0.3198015988986712, -1.2725587552459943, 0.31354772046343216, 0.5031848134353261, 1.2932258825322618, -0.11044702641731631, -0.6173620637123609, 0.5627610966190263],
            ]
        );

        let mut delta: Delta = Delta::Rnn { dwax: Matrix::new(na, nx), dwaa: Matrix::new(na, na), dba: vec![0.; na] };
        l.a0 = prev_a.clone();
        l.cell_back(da_next, 0, &mut delta);

        let Delta::Rnn { dwax, dwaa, dba } = &delta else { unreachable!() };
        assert_eq!(dba[4], 0.2002348);
        assert_eq!(dwax.at(3, 1), 0.41077277);
        assert_eq!(dwaa.at(1, 2), 1.1503452);
    }
}

