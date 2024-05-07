use crate::{Layer, LayerCache, Mat, MatView};

#[derive(Debug)]
pub struct ReLULayer {}

impl ReLULayer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Layer for ReLULayer {
    // 激活函数层每个神经元只有一条入边, 只是对上层的输出做一个转换, 矩阵形状n行1列
    fn forward(&mut self, input: &MatView, training: bool) -> (Mat, LayerCache) {
        let out = input.map(|x| x.max(0.));
        // 只有在训练时候才保存输出值，反向传播会用到
        let mut cache = vec![];
        if training {
            cache.push(input.to_owned());
        }
        (out, cache)
    }

    // relu偏导 x > 0 为1 其他情况为0
    fn backward(&mut self, grads: &MatView, cache_forward: &LayerCache) -> (Mat, LayerCache) {
        // 本层input的值
        let a = cache_forward[0].view();

        // 当前每个神经元上的偏导，n个神经元，每个神经元只有一条出边
        let mut r = Mat::from_shape_fn((a.len(), 1), |(_, _)| 0.);

        // 对每个神经元求梯度
        for (i, input) in a.iter().enumerate() {
            // 累加当前神经元每条出边的偏导
            for g in grads.rows().into_iter() {
                // 链式法则,与输入偏导相乘
                // 当前神经元为 i, 所以g也取每行第i个
                r[(i, 0)] += g[i] * (if *input > 0. { 1. } else { 0. });
            }
        }

        // 对上一层来说，本层相当于只有一个节点，所以矩阵形状应该是 (1,n)
        r = r.t().to_owned();

        (r, vec![])
    }

    fn update(&mut self, _learning_rate: f32, _gradss: &LayerCache) {
        //不需要做任何事情
    }
}

#[cfg(test)]
mod test {
    use ndarray::array;

    use crate::{layer_impls::relu::ReLULayer, Layer};

    #[test]
    fn test() {
        let mut s = ReLULayer::new();
        let (a, f_cache) = s.forward(&array![[0.], [1.]].view(), true);
        assert_eq!(a, array![[0.], [1.]]);
        let (g, b_cache) = s.backward(&array![[0.5, 0.5]].view(), &f_cache);
        println!("g:\n{}", g);
        assert_eq!(g, array![[0., 0.5]]);
        s.update(0.1, &b_cache);
    }
}
