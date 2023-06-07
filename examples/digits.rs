use nn::model::Model;
use nn::layer::{Layer, Activation};
use nn::matrix::Matrix;

use image::{DynamicImage, GenericImageView};

fn process_image(image: DynamicImage) -> Vec<f32> {
    let (w, h) = image.dimensions();
    let mut res: Vec<f32> = Vec::with_capacity((w * h) as usize);

    for y in 0..h {
        for x in 0..w {
            let pixel = image.get_pixel(x, y);
            res.push(pixel[0] as f32 / 255.);
        }
    }

    res
}

fn main() {
    let mut y: Matrix = Matrix::new(1, 20);

    let mut images: Vec<Vec<f32>> = Vec::new();
    for i in 0..10 {
        images.push(
            process_image(
                image::open(format!("data/digits/{}0.png", i)
            ).unwrap())
        );

        *y.atref(0, i) = 0.;
    }

    for i in 0..10 {
        images.push(
            process_image(
                image::open(format!("data/digits/{}1.png", i)
            ).unwrap())
        );

        *y.atref(0, i + 10) = 1.;
    }

    let x: Matrix = Matrix::from(images).transpose();

    let mut model: Model = Model::new();
    model.add(Layer::dense(x.rows(), Activation::Linear));
    model.add(Layer::dense(25, Activation::Sigmoid));
    model.add(Layer::dense(15, Activation::Sigmoid));
    model.add(Layer::dense(1, Activation::Sigmoid));
    model.train(&x, &y, 1000, 1.);

    let example0: Vec<f32> = process_image(
        image::open("data/digits/test0.png").unwrap()
    );

    let example1: Vec<f32> = process_image(
        image::open("data/digits/test1.png").unwrap()
    );

    let pred0: f32 = model.predict(
        &Matrix::from(vec![example0]).transpose()
    ).unwrap()[0];

    let pred1: f32 = model.predict(
        &Matrix::from(vec![example1]).transpose()
    ).unwrap()[0];

    println!("test0 prediction: {:.2}% accuracy", (1. - pred0) * 100.);
    println!("test1 prediction: {:.2}% accuracy", pred1 * 100.);
}

