use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::*;

/// DQN network architecture for Connect Four.
///
/// ```text
/// Input:  [batch, 3, 6, 7]
/// Conv1:  3 -> 32 channels, 3x3 kernel, same pad  =>  [batch, 32, 6, 7]
/// ReLU
/// Conv2:  32 -> 64 channels, 3x3 kernel, same pad =>  [batch, 64, 6, 7]
/// ReLU
/// Flatten: 64*6*7 = 2688
/// FC1:    2688 -> 128, ReLU
/// FC2:    128 -> 7  (Q-values, one per column)
/// ```
#[derive(Module, Debug)]
pub struct DqnNetwork<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct DqnNetworkConfig {}

impl DqnNetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DqnNetwork<B> {
        DqnNetwork {
            conv1: Conv2dConfig::new([3, 32], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            fc1: LinearConfig::new(2688, 128).init(device),
            fc2: LinearConfig::new(128, 7).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> DqnNetwork<B> {
    /// Forward pass: input [batch, 3, 6, 7] -> output [batch, 7] Q-values.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = input.dims()[0];

        let x = self.relu.forward(self.conv1.forward(input));
        let x = self.relu.forward(self.conv2.forward(x));
        let x = x.reshape([batch_size as i32, 2688]);
        let x = self.relu.forward(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu<f32, i32>;

    #[test]
    fn test_network_output_shape() {
        let device = Default::default();
        let config = DqnNetworkConfig {};
        let network = config.init::<TestBackend>(&device);

        let input = Tensor::zeros([2, 3, 6, 7], &device);
        let output = network.forward(input);
        assert_eq!(output.shape().dims, [2, 7]);
    }

    #[test]
    fn test_network_single_input() {
        let device = Default::default();
        let config = DqnNetworkConfig {};
        let network = config.init::<TestBackend>(&device);

        let input = Tensor::zeros([1, 3, 6, 7], &device);
        let output = network.forward(input);
        assert_eq!(output.shape().dims, [1, 7]);
    }
}
