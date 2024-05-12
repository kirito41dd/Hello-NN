use crate::Loss;

pub struct CrossEntropy {
    acc: usize,
    total: usize,
}

impl CrossEntropy {
    pub fn new() -> Self {
        Self { acc: 0, total: 0 }
    }
}

impl Loss for CrossEntropy {
    // 输出结果 和 期望结果 都是n行1列
    fn sum_loss(&mut self, result: &crate::MatView, label: &crate::MatView) {
        self.total += 1;
        let mut max_idx = 0;
        let mut max_v = f32::MIN;
        for (i, v) in result.iter().enumerate() {
            if *v > max_v {
                max_idx = i;
                max_v = *v;
            }
        }
        if label[(max_idx, 0)] == 1. {
            self.acc += 1;
        }
    }
    // 1 - 正确率
    fn loss(&self) -> f32 {
        let r = 1. - self.acc as f32 / self.total as f32;
        r
    }

    // 交叉熵梯度直接传label值, 网络最后一层必须是softmax
    fn grads(&mut self, _result: &crate::MatView, label: &crate::MatView) -> crate::Mat {
        label.to_owned()
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}
