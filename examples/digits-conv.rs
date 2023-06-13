use nn::model::{Model, Input};
use nn::layers::{Layer, Activation, pool::{Pooling, PoolType}};
use nn::matrix::{Matrix, Shape3, Shape4, Shape};

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
        let mut model: Model = Model::from("params");
        // let example0: Matrix = process_image(
        //     image::open("data/digits/test0.png").unwrap()
        // );

        // let example1: Matrix = process_image(
        //     image::open("data/digits/test1.png").unwrap()
        // );

        let example1: Matrix = Matrix::from(vec![
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.]
        ]);

        let example0: Matrix = Matrix::from(vec![
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.]
        ]);

        let pred1: f32 = model.predict(
            &Input::Conv(
                Shape4::from(
                    vec![
                        Shape3::from(vec![example1; 1]); 1
                    ]
                )
            )
        ).unwrap()[0];

        let pred0: f32 = model.predict(
            &Input::Conv(
                Shape4::from(
                    vec![
                        Shape3::from(vec![example0; 1]); 1
                    ]
                )
            )
        ).unwrap()[0];

        println!("test0 prediction: {:.2}% accuracy", (1. - pred0) * 100.);
        println!("test1 prediction: {:.2}% accuracy", pred1 * 100.);
    } else if args[0] == "train" {
        // let mut y: Matrix = Matrix::new(1, 20);
        let mut y: Matrix = Matrix::new(1, 2);

        let mut images: Vec<Matrix> = Vec::new();
        images.push(Matrix::from(vec![
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.],
            vec![1., 1., 1., 1., 1., 1., 1., 1.]
        ]));

        images.push(Matrix::from(vec![
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 0., 0., 0., 0.]
        ]));

        *y.atref(0, 0) = 1.;
        *y.atref(0, 1) = 0.;
        // for i in 0..10 {
        //     images.push(
        //         process_image(
        //             image::open(format!("data/digits/{}0.png", i)
        //         ).unwrap())
        //     );

        //     *y.atref(0, i) = 0.;
        // }

        // for i in 0..10 {
        //     images.push(
        //         process_image(
        //             image::open(format!("data/digits/{}1.png", i)
        //         ).unwrap())
        //     );

        //     *y.atref(0, i + 10) = 1.;
        // }

        // let mut data: Shape4 = Shape4::new(images.len(), 1, 28, 28);
        let mut data: Shape4 = Shape4::new(images.len(), 1, 8, 8);
        for e in 0..data.shape().0 {
            *data.at_mut(e).at_mut(0) = images[e].clone();
        }
        let x: Input = Input::Conv(data);

        let mut model: Model = Model::new();
        model.add(Layer::input(&x));
        model.add(Layer::conv(4, (3, 3), Activation::Relu, Pooling::new(PoolType::Max, 2, 2)));
        model.add(Layer::dense(1, Activation::Sigmoid));
        model.train(&x, &y, 1000, 1., true);
        model.save("params");
    } else {
        println!("Error: unrecognized subcommand '{}'", args[0]);
        std::process::exit(1);
    }
}

