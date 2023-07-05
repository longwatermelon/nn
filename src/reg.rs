pub struct DataPoint {
    pub features: Vec<f32>,
    pub y: f32,
}

impl DataPoint {
    pub fn new(n: usize) -> Self {
        Self { features: vec![0.; n], y: 0. }
    }

    pub fn from(features: &[f32], y: f32) -> Self {
        Self { features: Vec::from(features), y }
    }
}

/// Gradient descent
pub fn gdescent(w: &mut [f32], b: &mut f32, a: f32, x: &[DataPoint],
                //         w       x       b       y
                f: impl Fn(&[f32], &[f32], f32) -> f32)
{
    // New b
    let db_j: f32 = x.iter()
                     .map(|dp| f(w, &dp.features, *b) - dp.y)
                     .sum();
    let b1: f32 = *b - a * db_j;

    // New w
    let mut w1: Vec<f32> = vec![0.; w.len()];
    for j in 0..w1.len() {
        let mut dw_j: f32 = x.iter()
                     .map(|xi| (f(w, &xi.features, *b) - xi.y) * xi.features[j])
                     .sum();
        dw_j /= x.len() as f32;

        w1[j] = w[j] - a * dw_j;
    }

    *b = b1;
    w.clone_from_slice(&w1);
}

pub fn cost(x: &[DataPoint], err: impl Fn(&DataPoint) -> f32) -> f32 {
    x.iter().map(|x| err(x)).sum::<f32>() / (2. * x.len() as f32)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

