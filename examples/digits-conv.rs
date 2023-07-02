use nn::layers::{
    pool::{PoolType, Pooling},
    Activation, Layer,
};
use nn::matrix::{Matrix, Shape, Shape3, Shape4};
use nn::model::{Input, Model};

use image::{DynamicImage, GenericImageView};

fn process_image(image: DynamicImage) -> Matrix {
    let (w, h) = image.dimensions();
    let mut res: Matrix = Matrix::new(h as usize, w as usize);

    for y in 0..h {
        for x in 0..w {
            let pixel = image.get_pixel(x, y);
            *res.atref(y as usize, x as usize) = pixel[0] as f32 / 255.;
        }
    }

    res
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        let mut model: Model = Model::from("examples/model/params");
        let example0: Matrix =
            process_image(image::open("examples/data/digits/test0.png").unwrap());

        let example1: Matrix =
            process_image(image::open("examples/data/digits/test1.png").unwrap());

        let pred0: f32 = model
            .predict(&Input::Conv(Shape4::from(vec![Shape3::from(vec![
                example0,
            ])])))
            .unwrap()[0];

        let pred1: f32 = model
            .predict(&Input::Conv(Shape4::from(vec![Shape3::from(vec![
                example1,
            ])])))
            .unwrap()[0];

        println!("test0 prediction: {:.2}% accuracy", (1. - pred0) * 100.);
        println!("test1 prediction: {:.2}% accuracy", pred1 * 100.);
    } else if args[0] == "train" {
        let mut y: Matrix = Matrix::new(1, 20);
        let mut images: Vec<Matrix> = Vec::new();

        for i in 0..10 {
            images.push(process_image(
                image::open(format!("examples/data/digits/{}0.png", i)).unwrap(),
            ));

            *y.atref(0, i) = 0.;
        }

        for i in 0..10 {
            images.push(process_image(
                image::open(format!("examples/data/digits/{}1.png", i)).unwrap(),
            ));

            *y.atref(0, i + 10) = 1.;
        }

        let mut data: Shape4 = Shape4::new(images.len(), 1, 28, 28);
        for e in 0..data.shape().0 {
            *data.at_mut(e).at_mut(0) = images[e].clone();
        }
        let x: Input = Input::Conv(data);

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
        model.train(&x, &y, 500, 0.03, true);
        model.save("examples/model/params");

        // let mut x_copy: Shape4 = x.to_conv().clone();
        // x_copy.data_mut().remove(0);
        model.predict(&x).unwrap();
    } else {
        println!("Error: unrecognized subcommand '{}'", args[0]);
        std::process::exit(1);
    }
}
