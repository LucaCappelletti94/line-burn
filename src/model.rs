use crate::data::LineBatch;
use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, loss::BinaryCrossEntropyLossConfig,
    },
    prelude::*,
    tensor::{
        Int, Tensor,
        backend::{AutodiffBackend, Backend},
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config, Debug)]
pub struct LineModelConfig {
    pub dimension: usize,
    pub num_nodes: usize,
}

#[derive(Module, Debug)]
pub struct LineModel<B: Backend> {
    embedding: Embedding<B>,
    batch_norm: BatchNorm<B>,
}

impl<B: Backend> LineModel<B> {
    pub fn new(config: &LineModelConfig, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(config.num_nodes, config.dimension).init(device),
            batch_norm: BatchNormConfig::new(config.dimension).init(device),
        }
    }

    /// Returns the logits (dot products) for the given node pairs.
    pub fn forward(&self, src: Tensor<B, 1, Int>, dst: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        let batch_size = src.dims()[0];
        let src = src.reshape([batch_size, 1]); // [Batch, 1]
        let dst = dst.reshape([batch_size, 1]); // [Batch, 1]

        let u = self.embedding.forward(src); // [Batch, 1, Emb]
        let v = self.embedding.forward(dst); // [Batch, 1, Emb]

        // Remove the sequence dimension: [Batch, 1, Emb] -> [Batch, Emb]
        let u = u.squeeze::<2>();
        let v = v.squeeze::<2>();

        // Apply BatchNorm
        let u = self.batch_norm.forward(u);
        let v = self.batch_norm.forward(v);

        // u * v -> [Batch, Emb]
        // sum_dim(1) -> [Batch, 1]
        // squeeze::<1>() -> [Batch]
        (u * v).sum_dim(1).squeeze::<1>()
    }

    pub fn forward_loss(
        &self,
        src: Tensor<B, 1, Int>,
        dst: Tensor<B, 1, Int>,
        labels: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let logits = self.forward(src, dst);

        BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&logits.device())
            .forward(logits, labels)
    }

    /// Get the embeddings for all nodes.
    pub fn get_embeddings(&self) -> Tensor<B, 2> {
        self.embedding.weight.val()
    }
}

impl<B: AutodiffBackend> TrainStep<LineBatch<B>, RegressionOutput<B>> for LineModel<B> {
    fn step(&self, item: LineBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let logits = self.forward(item.src.clone(), item.dst.clone());
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&logits.device())
            .forward(logits.clone(), item.labels.clone());

        let grads = loss.backward();

        let output = logits.unsqueeze_dim(1);
        let targets = item.labels.float().unsqueeze_dim(1);

        TrainOutput::new(
            self,
            grads,
            RegressionOutput {
                loss,
                output,
                targets,
            },
        )
    }
}

impl<B: Backend> ValidStep<LineBatch<B>, RegressionOutput<B>> for LineModel<B> {
    fn step(&self, item: LineBatch<B>) -> RegressionOutput<B> {
        let logits = self.forward(item.src.clone(), item.dst.clone());
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&logits.device())
            .forward(logits.clone(), item.labels.clone());

        let output = logits.unsqueeze_dim(1);
        let targets = item.labels.float().unsqueeze_dim(1);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}
