use ndarray::{array, Array, Array2};

#[test]
fn test() {
    let a1 = Array::<f64, _>::eye(3);
    println!("a1:\n{}", a1);

    let a2 = array![[1, 1, 1], [0, 0, 0]];
    println!("a2:\n{}", a2);
    println!("a2[0,1] {:?}", a2.get((1, 0)));
    println!("a2.t():\n{}", a2.t()); // 转置

    let a3 = Array::from_elem((2, 1), 1);
    println!("a3:\n{}", a3);

    let a = array![1., 1., 1., 1., 1.];
    let b = array![1., 3., 4., 5., 6.];

    println!("a:\n{}", a);
    println!("b:\n{}", b.t());

    println!("r:\n{}", a.dot(&b));
    println!("r:\n{}", b.dot(&a));
}

#[test]
fn test2() {
    let a1 = Array2::from_shape_vec((2, 1), vec![1, 2]).unwrap();
    let b1 = Array2::from_shape_vec((2, 1), vec![1, 2]).unwrap();
    let a2 = Array2::from_shape_fn((2, 2), |(i, j)| i * 10 + j);

    println!(
        "row:{} col:{} shape:{:?} a:\n{}",
        a1.rows().into_iter().count(),
        a1.columns().into_iter().count(),
        a1.shape(),
        a1
    );

    println!("a1+b1:\n{}", a1 + b1);

    println!("a2[0,1] {}:\n{}", a2[(0, 1)], a2);
}
