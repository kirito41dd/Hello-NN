use crate::Loss;

pub struct MSE {
    sum: f32,
    total: usize,
}

impl MSE {
    pub fn new() -> Self {
        MSE { sum: 0., total: 0 }
    }
}

impl Loss for MSE {
    fn sum_loss(&mut self, result: &crate::MatView, label: &crate::MatView) {
        let diff = label - result;
        let a2 = &diff.view() * &diff.view();
        self.sum += a2.sum();
        self.total += 1;
    }

    fn loss(&self) -> f32 {
        self.sum / self.total as f32
    }

    fn grads(&mut self, result: &crate::MatView, label: &crate::MatView) -> crate::Mat {
        let grads = (result - label) * 2.;
        grads.t().to_owned() // [[1],[2],[3]] -> [[1,2,3]]
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}
