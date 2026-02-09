use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

/// Combined policy-value network for Policy Gradient agent.
///
/// Shares the DQN conv backbone with dual output heads:
/// ```text
/// Input:  [batch, 3, 6, 7]
/// Conv1:  3 -> 32 channels, 3x3 kernel  =>  [batch, 32, 4, 5]
/// ReLU
/// Conv2:  32 -> 64 channels, 3x3 kernel =>  [batch, 64, 2, 3]
/// ReLU
/// Flatten: 64*2*3 = 384
/// FC_shared: 384 -> 128, ReLU
/// Policy head: 128 -> 7  (logits, one per column)
/// Value head:  128 -> 1  (state value estimate)
/// ```
#[derive(Module, Debug)]
pub struct PolicyValueNetwork<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc_shared: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct PolicyValueNetworkConfig {}

impl PolicyValueNetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyValueNetwork<B> {
        PolicyValueNetwork {
            conv1: Conv2dConfig::new([3, 32], [3, 3]).init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3]).init(device),
            fc_shared: LinearConfig::new(384, 128).init(device),
            policy_head: LinearConfig::new(128, 7).init(device),
            value_head: LinearConfig::new(128, 1).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> PolicyValueNetwork<B> {
    /// Forward pass: input [batch, 3, 6, 7] -> (logits [batch, 7], value [batch, 1]).
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = input.dims()[0];

        let x = self.relu.forward(self.conv1.forward(input));
        let x = self.relu.forward(self.conv2.forward(x));
        let x = x.reshape([batch_size as i32, 384]);
        let x = self.relu.forward(self.fc_shared.forward(x));

        let logits = self.policy_head.forward(x.clone());
        let value = self.value_head.forward(x);

        (logits, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu<f32, i32>;

    #[test]
    fn test_policy_value_network_output_shapes() {
        let device = Default::default();
        let config = PolicyValueNetworkConfig {};
        let network = config.init::<TestBackend>(&device);

        let input = Tensor::zeros([2, 3, 6, 7], &device);
        let (logits, value) = network.forward(input);
        assert_eq!(logits.shape().dims, [2, 7]);
        assert_eq!(value.shape().dims, [2, 1]);
    }

    #[test]
    fn test_policy_value_network_single_input() {
        let device = Default::default();
        let config = PolicyValueNetworkConfig {};
        let network = config.init::<TestBackend>(&device);

        let input = Tensor::zeros([1, 3, 6, 7], &device);
        let (logits, value) = network.forward(input);
        assert_eq!(logits.shape().dims, [1, 7]);
        assert_eq!(value.shape().dims, [1, 1]);
    }
}
