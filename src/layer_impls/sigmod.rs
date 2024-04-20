use crate::{sigmod, Layer, LayerCache, Mat, MatView};
// 使用激活函数sigmod的层
#[derive(Debug)]
pub struct SigmodLayer {}

impl SigmodLayer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Layer for SigmodLayer {
    // 激活函数层每个神经元只有一条入边, 只是对上层的输出做一个转换, 矩阵形状n行1列
    fn forward(&mut self, input: &MatView, training: bool) -> (Mat, LayerCache) {
        let out = input.map(|x| sigmod(*x));
        // 只有在训练时候才保存输出值，反向传播会用到
        let mut cache = vec![];
        if training {
            cache.push(out.clone());
        }
        (out, cache)
    }

    // 激活函数层反向传播, 对sigmod(x)求导即可, 每个神经元只有一条入边,返回的梯度是 n行1列
    // simod(x)求导是 sigmod(x)*(1-sigmod(x))
    // 每个神经元有多条出边,链式法则后要累加结果
    fn backward(&mut self, grads: &MatView, cache_forward: &LayerCache) -> (Mat, LayerCache) {
        // sigmod(x)的值
        let a = cache_forward[0].view();

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
        (r, vec![])
    }

    fn update(&mut self, _learning_rate: f32, _gradss: &LayerCache) {
        //不需要做任何事情
    }
}

#[cfg(test)]
mod test {
    use ndarray::array;

    use crate::{layer_impls::SigmodLayer, sigmod, Layer};

    #[test]
    fn test() {
        assert_eq!(sigmod(0.), 0.5);
        assert_eq!(sigmod(0.) * (1. - sigmod(0.)), 0.25);

        let mut s = SigmodLayer::new();
        let (a, f_cache) = s.forward(&array![[0.], [0.]].view(), true);
        println!("a:\n{}", a);
        assert_eq!(a, array![[0.5], [0.5]]);
        let (g, b_cache) = s.backward(&array![[0.5, 0.5]].view(), &f_cache);
        println!("g:\n{}", g);
        assert_eq!(g, array![[0.125], [0.125]]);
        s.update(0.1, &b_cache);
    }
}
