use nn::prelude::*;
use nn::layers::rnn::SeqResult;
use nn::matrix::Matrix;

fn main() {
    // Determine if vector is ascending (1) or descending (0)
    let x: Vec<Vec<f32>> = vec![
        vec![0., 1., 2.],
        vec![2., 1., 0.]
    ];
    let x: Input = Input::Rnn(data::seq::one_dim(x));

    let y: Vec<Vec<f32>> = vec![vec![1.], vec![0.]];
    let y: Matrix = data::labels(y);

    let mut model: Model = Model::new();
    model.push(Layer::input(&x));
    model.push(Layer::rnn(5, SeqResult::Last));
    model.push(Layer::dense(1, Activation::Sigmoid));

    model.train(&x, &y, Target::Epochs(10000), 0.1, Some(10));
    model.save("examples/model/params");

    let xtest: Input = Input::Rnn(data::seq::one_dim(vec![vec![2., 1., 0.]]));
    println!("{:?}", model.predict(&xtest));
}

