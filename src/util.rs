pub fn softmax(x: &[f32]) -> Vec<f32> {
    let x: Vec<f32> = x.iter().map(|&x| f32::exp(x)).collect();
    let sum: f32 = x.iter().sum();
    x.iter().map(|x| x / sum).collect()
}

pub fn sigmoid(z: f32) -> f32 {
    1. / (1. + f32::exp(-z))
}

pub fn tanh(z: f32) -> f32 {
    (f32::exp(z) - f32::exp(-z)) /
    (f32::exp(z) + f32::exp(-z))
}

