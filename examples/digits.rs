use nn::prelude::*;
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
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        let mut model: Model = Model::from("examples/model/params");
        let example0: Vec<f32> =
            process_image(image::open("examples/data/digits/test0.png").unwrap());

        let example1: Vec<f32> =
            process_image(image::open("examples/data/digits/test1.png").unwrap());

        let pred0: f32 = model
            .predict(&Input::Dense(data::img::flat::one_dim(vec![example0])))
            .unwrap().at(0, 0);

        let pred1: f32 = model
            .predict(&Input::Dense(data::img::flat::one_dim(vec![example1])))
            .unwrap().at(0, 0);

        println!("test0 prediction: {:.2}% accuracy", (1. - pred0) * 100.);
        println!("test1 prediction: {:.2}% accuracy", pred1 * 100.);
    } else if args[0] == "train" {
        let mut y: Vec<Vec<f32>> = Vec::new();

        let mut images: Vec<Vec<f32>> = Vec::new();
        for i in 0..10 {
            images.push(process_image(
                image::open(format!("examples/data/digits/{}0.png", i)).unwrap(),
            ));

            y.push(vec![0.]);
        }

        for i in 0..10 {
            images.push(process_image(
                image::open(format!("examples/data/digits/{}1.png", i)).unwrap(),
            ));

            y.push(vec![1.]);
        }

        let x: Input = Input::Dense(data::img::flat::one_dim(images));
        let y: Matrix = data::labels(y);

        let mut model: Model = Model::new();
        model.push(Layer::input(&x));
        model.push(Layer::dense(25, Activation::Sigmoid));
        model.push(Layer::dense(15, Activation::Sigmoid));
        model.push(Layer::dense(1, Activation::Sigmoid));
        model.train(&x, &y, Target::Cost(0.001), 1., Some(100));
        model.save("examples/model/params");
    } else {
        println!("Error: unrecognized subcommand '{}'", args[0]);
        std::process::exit(1);
    }
}

