pub mod layer_impls;

use ndarray::{Array2, ArrayView2, ArrayViewMut2};

pub struct NeuralNetworkModel {
    pub layers: Vec<Box<dyn Layer>>,
}

pub type Mat = Array2<f32>;
pub type MatView<'a> = ArrayView2<'a, f32>;
pub type MatViewMut<'a> = ArrayViewMut2<'a, f32>;
pub type LayerCache = Vec<Mat>; // 每个layer的中间结果和梯度缓存

pub trait ToLayerCache {
    fn to_layer_cache(self) -> LayerCache;
}

impl ToLayerCache for Mat {
    fn to_layer_cache(self) -> LayerCache {
        vec![self]
    }
}

pub trait Layer {
    // 正向传播
    // 返回：本层输出 & 本层中间结果
    fn forward(&mut self, input: &MatView, training: bool) -> (Mat, LayerCache);
    // 反向传播
    // grads: 后面一层传递过来的梯度
    // cache_forward: 本层正向传播时的输入和激活值，内容为forward的返回
    // 返回: 本层向前一层传递的梯度 & 本层所有梯度值
    fn backward(&mut self, grads: &MatView, cache_forward: &LayerCache) -> (Mat, LayerCache);
    // 更新权重和偏置
    // grads: 本层调整参考的梯度, 内容格式与backward返回的一致
    fn update(&mut self, learning_rate: f32, grads: &LayerCache);
}

/// sigmod(X) = 1/(1 + e^(-x))
pub fn sigmod(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
