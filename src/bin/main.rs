
use hello_nn::loss_impls::{CrossEntropy};
use hello_nn::util::{shuffle};
use hello_nn::{Mat, MatView, NeuralNetworkModel};
use mnist_data_loader::{parse_imgs_from_reader, parse_labels_from_reader};


const BATCH_SIZE: usize = 16;
const INPUT_SIZE: usize = 784;

fn main() {
    let mut model = NeuralNetworkModel::new();
    model.push_dense_relu_layer(INPUT_SIZE, 256);
    //model.push_dense_sigmod_layer(16, 16);
    model.push_dense_softmax_layer(256, 10);
    model.minimize(CrossEntropy::new());
    //model.minimize(MSE::new());

    let (mut data, mut labels) = load_train_data().unwrap();
    let (test_data, test_labels) = load_test_data().unwrap();
    let mut epoch = 0;
    let _cnt = 20;
    let mut loss = 999.;
    loop {
        let mut data_it = data.chunks(BATCH_SIZE);
        let mut label_it = labels.chunks(BATCH_SIZE);
        while let Some(data) = data_it.next() {
            let label = label_it.next().unwrap();
            loss = model.fit(data, label, 0.1);
        }
        epoch += 1;
        //cnt -= 1;
        println!("epoch: {}, loss: {}", epoch, loss);
        print_rate(&mut model, &test_data, &test_labels);
        (data, labels) = shuffle(data, labels);
    }
}

pub fn print_rate(model: &mut NeuralNetworkModel, datas: &[Mat], labels: &[u8]) {
    let mut accept = 0;
    let mut wrong = 0;

    for (i, data) in datas.iter().enumerate() {
        let r = model.predict(&data.view());

        let got = judge(&r.view());
        let want = labels[i];
        //println!("r: {}: {}", want, r.t());
        if got == want {
            accept += 1;
        } else {
            wrong += 1;
            // println!("want: {} got: {}, sum:{}", want, got, r.sum());
        }
    }
    let rate = accept as f32 / (accept + wrong) as f32 * 100.;
    println!("accept: {}, wrong:{}, rate: {:.2}%", accept, wrong, rate);
}

pub fn load_train_data() -> anyhow::Result<(Vec<Mat>, Vec<Mat>)> {
    let mut train_imgs = std::fs::File::open("data/train-images.idx3-ubyte").unwrap();
    let mut train_labels = std::fs::File::open("data/train-labels.idx1-ubyte").unwrap();

    let (row, col, imgs) = parse_imgs_from_reader(&mut train_imgs).unwrap();
    let labels = parse_labels_from_reader(&mut train_labels).unwrap();

    let mut r_datas = vec![];
    let mut r_labels = vec![];

    let num_px = row * col;

    for data in imgs {
        // 灰度值转换为 0 - 1 的小数
        let data = data
            .into_iter()
            .map(|v| v as f32 / u8::MAX as f32)
            .collect();
        let d = Mat::from_shape_vec((num_px as _, 1), data)?;
        r_datas.push(d);
    }

    for label in labels {
        let mut l = vec![];
        for i in 0..10 {
            if i == label as i32 {
                l.push(1.);
            } else {
                l.push(0.);
            }
        }
        let l = Mat::from_shape_vec((10, 1), l)?;
        r_labels.push(l);
    }

    Ok((r_datas, r_labels))
}

pub fn load_test_data() -> anyhow::Result<(Vec<Mat>, Vec<u8>)> {
    let mut test_imgs = std::fs::File::open("data/t10k-images.idx3-ubyte").unwrap();
    let mut test_labels = std::fs::File::open("data/t10k-labels.idx1-ubyte").unwrap();

    let (row, col, imgs) = parse_imgs_from_reader(&mut test_imgs).unwrap();
    let labels = parse_labels_from_reader(&mut test_labels).unwrap();

    let mut r_datas = vec![];
    let mut r_labels = vec![];

    let num_px = row * col;

    for data in imgs {
        // 灰度值转换为 0 - 1 的小数
        let data = data
            .into_iter()
            .map(|v| v as f32 / u8::MAX as f32)
            .collect();
        let d = Mat::from_shape_vec((num_px as _, 1), data)?;
        r_datas.push(d);
    }

    for label in labels {
        r_labels.push(label);
    }

    Ok((r_datas, r_labels))
}

pub fn judge(result: &MatView) -> u8 {
    let mut max: f32 = -999999.;
    let mut idx = 0;
    for (i, v) in result.iter().enumerate() {
        if *v > max {
            max = *v;
            idx = i as _;
        }
    }
    idx
}
