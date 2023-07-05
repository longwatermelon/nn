use nn::reg::{self, DataPoint};

fn main() {
    let x: Vec<DataPoint> = vec![
        DataPoint::from(&[0., 1.], 1.),
        DataPoint::from(&[1., 0.], 0.),
    ];

    let mut w: Vec<f32> = vec![0.; 2];
    let mut b: f32 = 0.;

    for i in 0..100 {
        reg::gdescent(&mut w, &mut b, 0.1, &x,
            |w: &[f32], x: &[f32], b: f32| -> f32 {
                w.iter().zip(x.iter()).map(|(w, x)| w * x).sum::<f32>() + b
            }
        );

        if (i + 1) % 10 == 0 {
            println!("Iteration {}: cost {:.10}",
                i + 1, reg::cost(&x, |dp| {
                    let ypred: f32 = x.iter().map(|x| reg::dot(&w, &x.features)).sum::<f32>() + b;
                    f32::powi(ypred - dp.y, 2)
                })
            );
        }
    }

    println!("0 prediction: {}", reg::dot(&w, &[1., 0.]) + b);
    println!("1 prediction: {}", reg::dot(&w, &[0., 1.]) + b);
}

