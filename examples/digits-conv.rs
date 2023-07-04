use nn::layers::{
    pool::{PoolType, Pooling},
    Activation, Layer,
};
use nn::matrix::{Matrix, Shape, Shape3};
use nn::model::{Input, Model, Target};
use nn::data;

use image::{DynamicImage, GenericImageView};

fn process_image(image: DynamicImage) -> Shape3 {
    let (w, h) = image.dimensions();
    let mut res: Shape3 = Shape3::new(1, h as usize, w as usize);

    for y in 0..h {
        for x in 0..w {
            let pixel = image.get_pixel(x, y);
            *res.at_mut(0).atref(y as usize, x as usize) = pixel[0] as f32 / 255.;
        }
    }

    res
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        let mut model: Model = Model::from("examples/model/params");
        let example0: Shape3 =
            process_image(image::open("examples/data/digits/test0.png").unwrap());

        let example1: Shape3 =
            process_image(image::open("examples/data/digits/test1.png").unwrap());

        let pred0: f32 = model
            .predict(&Input::Conv(data::img::shaped::three_dim(vec![example0])))
            .unwrap()[0];

        let pred1: f32 = model
            .predict(&Input::Conv(data::img::shaped::three_dim(vec![example1])))
            .unwrap()[0];

        println!("test0 prediction: {:.2}% accuracy", (1. - pred0) * 100.);
        println!("test1 prediction: {:.2}% accuracy", pred1 * 100.);
    } else if args[0] == "train" {
        let mut y: Vec<Vec<f32>> = Vec::new();
        let mut images: Vec<Shape3> = Vec::new();

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

        let x: Input = Input::Conv(data::img::shaped::three_dim(images));
        let y: Matrix = data::labels(y);

        let mut model: Model = Model::new();
        model.push(Layer::input(&x));
        model.push(Layer::conv(
            6,
            (5, 5),
            Activation::Relu,
            Pooling::new(PoolType::Max, 2, 2),
        ));
        model.push(Layer::conv(
            16,
            (5, 5),
            Activation::Relu,
            Pooling::new(PoolType::Max, 2, 2),
        ));
        model.push(Layer::dense(1, Activation::Sigmoid));
        model.train(&x, &y, Target::Cost(0.0001), 0.05, Some(10));
        model.save("examples/model/params");
    } else {
        println!("Error: unrecognized subcommand '{}'", args[0]);
        std::process::exit(1);
    }
}
