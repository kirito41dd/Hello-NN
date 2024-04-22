pub mod layer_impls;
pub mod util;



use ndarray::{Array2, ArrayView2, ArrayViewMut2};

use crate::util::calc_all_grads_avg;

pub struct NeuralNetworkModel {
    pub layers: Vec<Box<dyn Layer>>,
}
impl NeuralNetworkModel {
    pub fn new() -> Self {
        NeuralNetworkModel { layers: vec![] }
    }
    pub fn push_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }
    pub fn predict(&mut self, data: &MatView) -> Mat {
        let mut pre = data.to_owned();
        for layer in self.layers.iter_mut() {
            let (a, _) = layer.forward(&pre.view(), false);
            pre = a;
        }
        pre
    }

    pub fn fit(&mut self, datas: &[Mat], labels: &[Mat], learning_rate: f32) -> f32 {
        let mut forward_cache: Vec<Vec<LayerCache>> = vec![]; // forwart_cache[i][j] 表示第i个样本的第j层缓存
        let mut out_cache: Vec<Mat> = vec![]; // out_cache[i] 表示第i个样本的正向传播结果

        let batch_size = datas.len();
        let mut loss = 0.;
        // 对每个样本进行正向传播, 算出方差
        for i in 0..batch_size {
            forward_cache.push(vec![]);
            let data = datas[i].view();
            let label = labels[i].view();
            let mut pre = data.to_owned();
            for (_j, layer) in self.layers.iter_mut().enumerate() {
                let (a, f_cache) = layer.forward(&pre.view(), true);
                forward_cache[i].push(f_cache);
                pre = a;
            }
            //println!("a:\n{}\nlabel:\n{}", pre, label);
            let diff = &label - &pre.view();
            let a2 = &diff.view() * &diff.view();
            let suma2 = a2.sum();
            loss += suma2;

            //println!("diff:\n{} a2:{}", diff, suma2);
            out_cache.push(pre);
        }
        loss = loss / batch_size as f32;

        // 对每个样本进行反向传播，得到梯度，初始的偏导是 2(a-b)/n
        // 收集每个样本的全量梯度，将他们的梯度相加
        let mut grad_cache = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let a = &out_cache[i].view();
            let b = &labels[i].view();
            let mut grads = ((a - b) * 2.) / batch_size as f32;
            grads = grads.t().to_owned(); // [[1],[2],[3]] -> [[1,2,3]]

            // println!("a    : {}", a.t());
            // println!("b    : {}", b.t());
            // println!("grads: {}", grads);

            // let mut b = String::new();
            // std::io::stdin().read_line(&mut b);

            let mut cache = vec![vec![]; self.layers.len()];
            for j in (0..self.layers.len()).rev() {
                // println!("g:\n{}", grads);
                // println!(
                //     "now layer {}, caceh forward:\n{:?}",
                //     j, &self.forward_cache[i][j]
                // );
                // 从后往前
                let layer = &mut self.layers[j];
                let (g, backwark_cache) = layer.backward(&grads.view(), &forward_cache[i][j]);
                grads = g;
                //println!("layer {} cache len:{}", j, backwark_cache.len());
                cache[j] = backwark_cache;
            }
            //println!("item {}, cache len:{}", i, cache.len());
            grad_cache.push(cache);
        }

        // 把每个样本的全量梯度相加，然后取平均，作为最终调整参数的梯度
        let cache: Vec<LayerCache> = calc_all_grads_avg(&grad_cache);
        //println!("avg cache len:{}", cache.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // println!("layer {} cache:", i);
            // for v in &cache[i] {
            //     println!("->\n{}", v);
            // }
            layer.update(learning_rate, &cache[i]);
        }

        //println!("loss: {}", loss);
        return loss;
    }
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
