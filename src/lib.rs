use ndarray::{Array2, ArrayView2, ArrayViewMut2};

pub struct NeuralNetworkModel {
    pub layers: Vec<Box<dyn Layer>>,
}

pub type Mat = Array2<f32>;
pub type MatView<'a> = ArrayView2<'a, f32>;
pub type MatViewMut<'a> = ArrayViewMut2<'a, f32>;

pub trait Layer {
    // 正向传播
    fn forward(&mut self, input: &MatView, training: bool) -> Mat;
    // 反向传播 并 更新权重和偏置
    fn backward_and_update(&mut self, grads: &MatView, learning_rate: f32) -> Mat;
}

// 没有激活函数的全连接层
pub struct DenseLayerNoActive {
    // 每个神经元与上一层所有神经元边的权重, n行j列,n是本层神经元个数,j是前一层神经元个数
    pub w: Mat,
    // 每个神经元的偏置, n行1列
    pub b: Mat,
    // 本层神经元的激活值
    pub a: Option<Mat>,
    pub input: Option<Mat>,
}

impl Layer for DenseLayerNoActive {
    fn forward(&mut self, input: &MatView, training: bool) -> Mat {
        // 计算每个神经元激活值 w1*a1 + w2*a2 + ... + wn*an + b
        // 矩阵计算,一次算出结果, w的每行乘以输入的一列最后加b
        let r = self.w.dot(input) + &self.b;
        if training {
            self.a = Some(r.clone());
            self.input = Some(input.to_owned());
        }
        r
    }

    // 每个神经元有多(k)条入边返回的梯度是 n行k列
    // z=w*a+b 对w求导是a, 对b求导是1
    // 每个神经元看作有多条出边,链式法则后仍要累加(大多情况后一层是激活函数层,只有1条出边,但不排除其他可能)
    fn backward_and_update(&mut self, grads: &MatView, learning_rate: f32) -> Mat {
        let a = match self.input {
            Some(ref v) => v,
            None => {
                panic!("set training=true when forward")
            }
        };

        let mut bias_grads = Mat::zeros(self.b.raw_dim());
        let mut w_grads = Mat::zeros(self.w.raw_dim());

        // 对每个神经元求所有w和b的偏导, 每个w的导数都是与其相乘的a, w不需要参与, 对b的偏导是1
        for (i, _) in self.w.columns().into_iter().enumerate() {
            // 累加当前神经元每条出边上的偏导, grads的每行,都是前一层某个神经元和本层连线的偏导
            for g in grads.rows().into_iter() {
                // b在这里求 链式法则相乘
                bias_grads[(i, 0)] += g[i] * 1.;
                //每个神经元上都有和前一层神经元的边, 连接w和a
                for (k, a) in a.rows().into_iter().enumerate() {
                    w_grads[(i, k)] += a[0] * g[i];
                }
            }
        }

        // 更新偏置
        let (i, j) = (self.w.shape()[0], self.w.shape()[1]);
        for i in 0..i {
            for j in 0..j {
                self.w[(i, j)] -= learning_rate * w_grads[(i, j)];
            }
            self.b[(i, 0)] -= learning_rate * bias_grads[(i, 0)];
        }

        // 入边只和w有关系,不用返回偏置上的偏导
        w_grads
    }
}

// 使用激活函数sigmod的层
#[derive(Debug)]
pub struct SigmodLayer {
    // 本层神经元的激活值
    pub a: Option<Mat>,
}

impl SigmodLayer {
    pub fn new() -> Self {
        Self { a: None }
    }
}

impl Layer for SigmodLayer {
    // 激活函数层每个神经元只有一条入边, 只是对上层的输出做一个转换, 矩阵形状n行1列
    fn forward(&mut self, input: &MatView, training: bool) -> Mat {
        let out = input.map(|x| sigmod(*x));
        // 只有在训练时候才保存输出值，反向传播会用到
        if training {
            self.a = Some(out.clone());
        }
        out
    }

    // 激活函数层反向传播, 对sigmod(x)求导即可, 每个神经元只有一条入边,返回的梯度是 n行1列
    // simod(x)求导是 sigmod(x)*(1-sigmod(x))
    // 每个神经元有多条出边,链式法则后要累加结果
    fn backward_and_update(&mut self, grads: &MatView, _: f32) -> Mat {
        // sigmod(x)的值
        let a = match self.a {
            Some(ref v) => v,
            None => {
                panic!("set training=true when forward")
            }
        };

        // 激活函数层的输入和输出数量是相等的, 返回值长度和前一层神经元数量一致
        let mut r = Mat::from_shape_fn((a.len(), 1), |(_, _)| 0.);

        // 对每个神经元求梯度
        for (i, out) in a.iter().enumerate() {
            // 累加当前神经元每条出边的偏导
            for g in grads.rows().into_iter() {
                // 链式法则,与输入偏导相乘
                // 当前神经元为 i, 所以g也取每行第i个
                r[(i, 0)] += g[i] * (out * (1.0 - out));
            }
        }

        // 激活函数层没有任何存储任何权重和偏置,无需update
        r
    }
}

/// sigmod(X) = 1/(1 + e^(-x))
pub fn sigmod(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod test {
    use ndarray::array;

    use crate::{sigmod, Layer, Mat, SigmodLayer};

    #[test]
    fn test() {
        assert_eq!(sigmod(0.), 0.5);
        assert_eq!(sigmod(0.) * (1. - sigmod(0.)), 0.25);

        let mut s = SigmodLayer::new();
        let a = s.forward(&array![[0.], [0.]].view(), true);
        println!("a:\n{}", a);
        assert_eq!(a, array![[0.5], [0.5]]);
        let g = s.backward_and_update(&array![[0.5, 0.5]].view(), 0.1);
        println!("g:\n{}", g);
        assert_eq!(g, array![[0.125], [0.125]])
    }
}
