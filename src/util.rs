pub fn softmax(x: &[f32]) -> Vec<f32> {
    let x: Vec<f32> = x.iter().map(|&x| f32::exp(x)).collect();
    let sum: f32 = x.iter().sum();
    x.iter().map(|x| x / sum).collect()
}

