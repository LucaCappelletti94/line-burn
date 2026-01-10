use crate::{
    data::{CooGraph, LineBatch, LineBatcher, LineItem},
    model::{LineModel, LineModelConfig},
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamWConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};
use std::sync::Arc;

#[derive(Config, Debug)]
pub struct LineTrainingConfig {
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 256)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1)]
    pub negative_samples: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    pub optimizer: AdamWConfig,
}

pub fn run<B: AutodiffBackend>(
    artifact_dir: &str,
    graph: Arc<CooGraph>,
    valid_graph: Arc<CooGraph>,
    config: LineTrainingConfig,
    model_config: LineModelConfig,
    device: B::Device,
) -> LineModel<B::InnerBackend> {
    let batcher = LineBatcher::new(graph.num_nodes, config.negative_samples);

    // Train dataloader
    let dataloader_train = DataLoaderBuilder::<B, LineItem, LineBatch<B>>::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(graph.as_ref().clone());

    // Valid dataloader
    let dataloader_valid =
        DataLoaderBuilder::<B::InnerBackend, LineItem, LineBatch<B::InnerBackend>>::new(
            batcher.clone(),
        )
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_graph.as_ref().clone());

    let model = LineModel::new(&model_config, &device);
    let optim = config.optimizer.init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optim, config.learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained.model
}
