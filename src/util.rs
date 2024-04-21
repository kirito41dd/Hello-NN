use crate::{
    layer_impls::{DenseLayerNoActive, SigmodLayer},
    LayerCache, Mat, NeuralNetworkModel,
};

impl NeuralNetworkModel {
    pub fn push_dense_layer(&mut self, pre_cnt: usize, cell_cnt: usize) {
        self.push_layer(DenseLayerNoActive::new(pre_cnt, cell_cnt));
        self.push_layer(SigmodLayer::new());
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

#[cfg(test)]
mod test {
    use ndarray::array;

    use crate::{LayerCache, Mat};

    use super::calc_all_grads_avg;

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
        for layer_cahce in result.iter() {
            println!("layer_ca:\n");
            for g in layer_cahce.iter() {
                println!("g:\n{}", g);
            }
        }

        let c1: LayerCache = vec![array![[5.5, 11.,]], array![[550., 550.,]]];
        let c2: LayerCache = vec![array![[55., 110.]], array![[0., 0.,]]];
        let item3: Vec<LayerCache> = vec![c1, c2];

        assert_eq!(result, item3);
    }
}
