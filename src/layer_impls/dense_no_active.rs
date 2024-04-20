use crate::{Layer, LayerCache, Mat, MatView};

// 没有激活函数的全连接层
pub struct DenseLayerNoActive {
    // 每个神经元与上一层所有神经元边的权重, n行j列,n是本层神经元个数,j是前一层神经元个数
    pub w: Mat,
    // 每个神经元的偏置, n行1列
    pub b: Mat,
}

impl Layer for DenseLayerNoActive {
    fn forward(&mut self, input: &MatView, training: bool) -> (Mat, LayerCache) {
        // 计算每个神经元激活值 w1*a1 + w2*a2 + ... + wn*an + b
        // 矩阵计算,一次算出结果, w的每行乘以输入的一列最后加b
        let r = self.w.dot(input) + &self.b;
        let mut cache = vec![];
        if training {
            cache.push(input.to_owned());
        }
        (r, cache)
    }

    // 每个神经元有多(k)条入边返回的梯度是 n行k列
    // z=w*a+b 对w求导是a, 对b求导是1
    // 每个神经元看作有多条出边,链式法则后仍要累加(大多情况后一层是激活函数层,只有1条出边,但不排除其他可能)
    fn backward(&mut self, grads: &MatView, cache_forward: &LayerCache) -> (Mat, LayerCache) {
        let a = cache_forward[0].view();

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

        let grads_cache = vec![bias_grads, w_grads.clone()];

        // 入边只和w有关系,不用返回偏置上的偏导
        (w_grads, grads_cache)
    }

    fn update(&mut self, learning_rate: f32, grades: &LayerCache) {
        let bias_grads = grades[0].view();
        let w_grads = grades[1].view();
        // 更新偏置
        let (i, j) = (self.w.shape()[0], self.w.shape()[1]);
        for i in 0..i {
            for j in 0..j {
                self.w[(i, j)] -= learning_rate * w_grads[(i, j)];
            }
            self.b[(i, 0)] -= learning_rate * bias_grads[(i, 0)];
        }
    }
}

#[cfg(test)]
mod test {
    

    use ndarray::array;

    use crate::{
        layer_impls::{DenseLayerNoActive}, Layer,
    };

    #[test]
    fn test() {
        let mut d = DenseLayerNoActive {
            w: array![[2., 2.], [2., 2.]],
            b: array![[0.1], [0.1]],
        };

        let (a, f_cache) = d.forward(&array![[0.5], [1.]].view(), true);
        println!("a:\n{}", a);
        assert_eq!(a, array![[3.1], [3.1]]);
        let (g, b_cache) = d.backward(&array![[3.1, 3.1], [3.1, 3.1]].view(), &f_cache);
        println!(
            "g:\n{}, b_cache:\ng_b:\n{}\ng_w\n{}",
            g, b_cache[0], b_cache[1]
        );
        assert_eq!(g, array![[3.1, 6.2], [3.1, 6.2]]);
        d.update(0.1, &b_cache);
        println!("d.w:\n{}\nd.b\n{}", d.w, d.b);
    }
}
