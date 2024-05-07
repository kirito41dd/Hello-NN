use crate::{Layer, LayerCache, Mat, MatView};

#[derive(Debug)]
pub struct SoftmaxLayer {}

impl SoftmaxLayer {
    pub fn new() -> Self {
        Self {}
    }
}
impl Layer for SoftmaxLayer {
    // 输入为上层激活值，n行1列
    fn forward(&mut self, input: &MatView, training: bool) -> (Mat, LayerCache) {
        let sum = input.fold(0., |acc, b| acc + b.exp());
        let out = input.map(|x| x.exp() / sum);
        // 只有在训练时候才保存输出值，反向传播会用到
        let mut cache = vec![];
        if training {
            cache.push(out.clone());
        }
        (out, cache)
    }

    // 前一层节点数量和本层节点数量一致，直接对交叉熵损失函数求导而不是对softmax求导，这样更简单
    // 一定在最后一层
    fn backward(&mut self, grads: &MatView, cache_forward: &LayerCache) -> (Mat, LayerCache) {
        // 本层输出值 a[i] = softmax[i]
        let a = cache_forward[0].view();

        // 当前每个神经元上的偏导，n个神经元，每个神经元只有一条出边
        // L/z = pi - yi
        let mut r = &a.view() - &grads.view();

        // 直接算出了上层输出对交叉熵的偏导，所以矩阵形状应该是 (1,n)
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

    use crate::Layer;

    use super::SoftmaxLayer;

    #[test]
    fn test() {
        let mut l = SoftmaxLayer::new();
        let (out, cache) = l.forward(&array![[2.], [3.], [5.]].view(), true);
        assert_eq!(out, array![[0.042010065], [0.1141952], [0.8437947]]);
        let (g, _) = l.backward(&array![[0.0], [1.0], [0.0]].view(), &cache);
        assert_eq!(g, array![[0.042010065, -0.8858048, 0.8437947]]);
    }
}
