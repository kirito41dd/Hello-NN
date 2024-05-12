

use ndarray_rand::rand_distr::Normal;

use rand::thread_rng;
use rand::{distributions::Distribution, seq::SliceRandom};

use crate::{
    layer_impls::{DenseLayerNoActive, ReLULayer, SigmodLayer, SoftmaxLayer},
    LayerCache, Mat, NeuralNetworkModel,
};

impl NeuralNetworkModel {
    pub fn push_dense_sigmod_layer(&mut self, pre_cnt: usize, cell_cnt: usize) {
        self.push_layer(DenseLayerNoActive::new_with(
            pre_cnt,
            cell_cnt,
            DistCustomWrap::new(Normal::new(0., 1.).unwrap(), |v| v * 0.01),
        ));
        self.push_layer(SigmodLayer::new());
    }
    pub fn push_dense_softmax_layer(&mut self, pre_cnt: usize, cell_cnt: usize) {
        self.push_layer(DenseLayerNoActive::new_with(
            pre_cnt,
            cell_cnt,
            DistCustomWrap::new(Normal::new(0., 1.).unwrap(), |v| v * 0.01),
        ));
        self.push_layer(SoftmaxLayer::new());
    }
    pub fn push_dense_relu_layer(&mut self, pre_cnt: usize, cell_cnt: usize) {
        self.push_layer(DenseLayerNoActive::new_with(
            pre_cnt,
            cell_cnt,
            DistCustomWrap::new(Normal::new(0., 1.).unwrap(), |v| v * 0.01),
        ));
        self.push_layer(ReLULayer::new());
    }
}

// 入参数组：每个样本，每层的梯度缓存
// 累加所有样本梯度，然后求平均
// 返回avg的梯度
pub fn calc_all_grads_avg(all_grades: &Vec<Vec<LayerCache>>) -> Vec<LayerCache> {
    let cnt = all_grades.len();
    if cnt == 0 {
        return vec![];
    }
    let mut base = all_grades[0].clone();

    // 从第二个样本开始，往base上累加
    for i in 1..cnt {
        // 当前样本
        let now = &all_grades[i];
        // 每层每层加
        for (j, layer_now) in now.iter().enumerate() {
            let layer_base: &mut Vec<Mat> = &mut base[j];
            // 每个梯度累加
            for (k, g_now) in layer_now.iter().enumerate() {
                let g_base: &mut Mat = &mut layer_base[k];
                *g_base = &g_base.view() + &g_now.view();
            }
        }
    }

    // 求平均
    base.iter_mut().for_each(|v: &mut Vec<Mat>| {
        v.iter_mut().for_each(|g: &mut Mat| {
            *g = &g.view() / cnt as f32;
        })
    });

    base
}

pub fn shuffle<A, B>(a: Vec<A>, b: Vec<B>) -> (Vec<A>, Vec<B>) {
    let mut ab = a.into_iter().zip(b.into_iter()).collect::<Vec<(A, B)>>();
    let mut rng = thread_rng();
    let _shuffle = ab.shuffle(&mut rng);
    let mut ra = Vec::with_capacity(ab.len());
    let mut rb = Vec::with_capacity(ab.len());
    for (a, b) in ab {
        ra.push(a);
        rb.push(b)
    }
    (ra, rb)
}

pub fn pause() {
    let mut s = String::new();
    _ = std::io::stdin().read_line(&mut s);
}

// 处理初始化的随机数，可以对random的随机值进行转换
#[derive(Clone)]
pub struct DistCustomWrap<I, F> {
    inner: I,
    f: F,
}

impl<I, F> DistCustomWrap<I, F>
where
    I: Distribution<f32>,
    F: Fn(f32) -> f32,
{
    pub fn new(inner: I, f: F) -> Self {
        DistCustomWrap { inner, f }
    }
}

impl<I, F> Distribution<f32> for DistCustomWrap<I, F>
where
    I: Distribution<f32>,
    F: Fn(f32) -> f32,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        (self.f)(self.inner.sample(rng))
    }
}

#[cfg(test)]
mod test {
    use ndarray::array;

    use crate::LayerCache;

    use super::{calc_all_grads_avg, shuffle};

    #[test]
    fn test_calc_all_grads_avg() {
        let c1: LayerCache = vec![array![[1., 2.,]], array![[100., 100.,]]];
        let c2: LayerCache = vec![array![[10., 20.,]], array![[0., 0.,]]];
        let item1: Vec<LayerCache> = vec![c1, c2];

        let c1: LayerCache = vec![array![[10., 20.,]], array![[1000., 1000.,]]];
        let c2: LayerCache = vec![array![[100., 200.,]], array![[0., 0.,]]];
        let item2: Vec<LayerCache> = vec![c1, c2];
        let all = vec![item1, item2];

        let result = calc_all_grads_avg(&all);

        let c1: LayerCache = vec![array![[5.5, 11.,]], array![[550., 550.,]]];
        let c2: LayerCache = vec![array![[55., 110.]], array![[0., 0.,]]];
        let item3: Vec<LayerCache> = vec![c1, c2];

        assert_eq!(result, item3);

        let c1: LayerCache = vec![array![[1., 2.,]], array![[100., 100.,]]];
        let c2: LayerCache = vec![array![[10., 20.,]], array![[0., 0.,]]];
        let item4: Vec<LayerCache> = vec![c1, c2];
        let all = vec![
            item4.clone(),
            item4.clone(),
            item4.clone(),
            item4.clone(),
            item4,
        ];
        let result = calc_all_grads_avg(&all);
        assert_eq!(all[0], result);
    }

    #[test]
    fn test_shuffle() {
        let (a, b) = shuffle(vec![1, 2, 3, 4, 5, 6, 7], vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(a, b);
    }
}
