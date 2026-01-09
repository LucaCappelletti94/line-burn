use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig, BatchNorm, BatchNormConfig};
use burn::tensor::backend::{Backend, AutodiffBackend};
use burn::tensor::{Tensor, Int, Distribution};
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use indicatif::{ProgressBar, ProgressStyle};

/// A graph represented in Coordinate Format (COO).
/// Stores edges as pairs of indices (row, col).
#[derive(Clone, Debug)]
pub struct CooGraph {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl CooGraph {
    pub fn new(row_indices: Vec<usize>, col_indices: Vec<usize>, num_nodes: usize) -> Self {
        assert_eq!(row_indices.len(), col_indices.len(), "Row and column indices must have same length");
        let num_edges = row_indices.len();
        Self {
            row_indices,
            col_indices,
            num_nodes,
            num_edges,
        }
    }
}

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
    /// 
    /// # Arguments
    /// * `src` - Tensor of source node indices [batch_size]
    /// * `dst` - Tensor of destination node indices [batch_size]
    /// 
    /// # Returns
    /// * Tensor of logits [batch_size]
    pub fn forward(&self, src: Tensor<B, 1, Int>, dst: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        let batch_size = src.dims()[0];
        let src = src.reshape([batch_size, 1]); // [Batch, 1]
        let dst = dst.reshape([batch_size, 1]); // [Batch, 1]

        let u = self.embedding.forward(src); // [Batch, 1, Emb]
        let v = self.embedding.forward(dst); // [Batch, 1, Emb]
        
        // Remove the sequence dimension: [Batch, 1, Emb] -> [Batch, Emb]
        // squeeze::<2> squeezes to rank 2.
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

    /// Computes the First-order LINE loss for a batch of positive and negative edges.
    /// 
    /// This uses Binary Cross Entropy with Logits.
    /// 
    /// # Arguments
    /// * `src` - Tensor of source node indices [batch_size]
    /// * `dst` - Tensor of destination node indices [batch_size]
    /// * `labels` - Tensor of binary labels (1 for existing edge, 0 for negative sample) [batch_size]
    /// 
    /// # Returns
    /// * Scalar loss tensor
    pub fn forward_loss(&self, src: Tensor<B, 1, Int>, dst: Tensor<B, 1, Int>, labels: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        let logits = self.forward(src, dst);
        
        BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&logits.device())
            .forward(logits, labels)
    }

    /// Get the embeddings for all nodes.
    pub fn get_embeddings(&self) -> Tensor<B, 2> {
        // Return the weight matrix for the embedding layer.
        // The weight matrix is [num_nodes, dimension].
        self.embedding.weight.val()
    }
}

/// Trains the LINE model using the provided graph and configuration.
/// Returns the trained model, the training loss history (per step), and validation loss history (step, loss).
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    graph: CooGraph,
    valid_graph: Option<&CooGraph>,
    config: LineModelConfig,
    epochs: usize,
    batch_size: usize,
    lr: f64,
) -> (LineModel<B>, Vec<f32>, Vec<(usize, f32)>) {
    let mut model = LineModel::new(&config, device);
    let mut optim = AdamConfig::new().init();
    let mut loss_history = Vec::new();
    let mut val_history = Vec::new();

    let num_edges = graph.num_edges;
    let num_nodes = graph.num_nodes;
    let half_batch = batch_size / 2;
    let num_batches = if half_batch > 0 { num_edges / half_batch } else { 0 };

    // Convert indices to i32 for creating Int tensors
    let row_indices_data: Vec<i32> = graph.row_indices.iter().map(|&x| x as i32).collect();
    let col_indices_data: Vec<i32> = graph.col_indices.iter().map(|&x| x as i32).collect();

    let all_rows = Tensor::<B, 1, Int>::from_ints(&row_indices_data[..], device);
    let all_cols = Tensor::<B, 1, Int>::from_ints(&col_indices_data[..], device);

    let total_steps = epochs * num_batches;
    let pb = ProgressBar::new(total_steps as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) Loss: {msg}")
        .unwrap()
        .progress_chars("#>-"));

    let mut step_counter = 0;
    let mut last_validation_loss = None;

    // Validate at end of epoch
    if let Some(vg) = valid_graph {
        let val_loss = evaluate(&model, device, vg, batch_size);
        last_validation_loss = Some(val_loss);
        val_history.push((step_counter, val_loss));
        pb.set_message(format!("{:.4} (Val: {:.4})", loss_history.last().unwrap_or(&0.0), val_loss));
    }

    for _epoch in 0..epochs {
        for _ in 0..num_batches {
            // 1. Sample positive edges using random float indices converted to int
            // Range [0, num_edges)
            let idx = Tensor::<B, 1>::random([half_batch], Distribution::Uniform(0.0, num_edges as f64), device).int();
            let pos_src = all_rows.clone().select(0, idx.clone());
            let pos_dst = all_cols.clone().select(0, idx);
            let pos_labels = Tensor::ones([half_batch], device);

            // 2. Sample negative edges
            // Uniformly random nodes [0, num_nodes)
            let neg_src = Tensor::<B, 1>::random([half_batch], Distribution::Uniform(0.0, num_nodes as f64), device).int();
            let neg_dst = Tensor::<B, 1>::random([half_batch], Distribution::Uniform(0.0, num_nodes as f64), device).int();
            let neg_labels = Tensor::zeros([half_batch], device);

            // 3. Combine
            let src = Tensor::cat(vec![pos_src, neg_src], 0);
            let dst = Tensor::cat(vec![pos_dst, neg_dst], 0);
            let labels = Tensor::cat(vec![pos_labels, neg_labels], 0);

            // 4. Backward
            let loss = model.forward_loss(src, dst, labels);
            
            // Record loss
            let loss_scalar = loss.clone().into_scalar().to_f32();
            loss_history.push(loss_scalar);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if let Some(vl) = last_validation_loss {
                pb.set_message(format!("{:.4} (Val: {:.4})", loss_scalar, vl));
            } else {
                pb.set_message(format!("{:.4}", loss_scalar));
            }
            pb.inc(1);
            step_counter += 1;
        }

        // Validate at end of epoch
        if let Some(vg) = valid_graph {
            let val_loss = evaluate(&model, device, vg, batch_size);
            val_history.push((step_counter, val_loss));
            pb.set_message(format!("{:.4} (Val: {:.4})", loss_history.last().unwrap_or(&0.0), val_loss));
        }
    }
    pb.finish_with_message("Training complete");
    (model, loss_history, val_history)
}

/// Evaluates the model on the provided graph by computing the average loss.
/// This acts as a validation step.
pub fn evaluate<B: AutodiffBackend>(
    model: &LineModel<B>,
    device: &B::Device,
    graph: &CooGraph,
    batch_size: usize,
) -> f32 {
    let num_edges = graph.num_edges;
    let num_nodes = graph.num_nodes;
    let half_batch = batch_size / 2;
    let num_batches = if half_batch > 0 { num_edges / half_batch } else { 0 };

    // Convert indices to i32 for creating Int tensors
    let row_indices_data: Vec<i32> = graph.row_indices.iter().map(|&x| x as i32).collect();
    let col_indices_data: Vec<i32> = graph.col_indices.iter().map(|&x| x as i32).collect();

    let all_rows = Tensor::<B, 1, Int>::from_ints(&row_indices_data[..], device);
    let all_cols = Tensor::<B, 1, Int>::from_ints(&col_indices_data[..], device);

    let mut total_loss = 0.0;
    
    // We iterate once over the dataset
    for _ in 0..num_batches {
        // 1. Sample positive edges using random float indices converted to int
        let idx = Tensor::<B, 1>::random([half_batch], Distribution::Uniform(0.0, num_edges as f64), device).int();
        let pos_src = all_rows.clone().select(0, idx.clone());
        let pos_dst = all_cols.clone().select(0, idx);
        let pos_labels = Tensor::ones([half_batch], device);

        // 2. Sample negative edges
        let neg_src = Tensor::<B, 1>::random([half_batch], Distribution::Uniform(0.0, num_nodes as f64), device).int();
        let neg_dst = Tensor::<B, 1>::random([half_batch], Distribution::Uniform(0.0, num_nodes as f64), device).int();
        let neg_labels = Tensor::zeros([half_batch], device);

        // 3. Combine
        let src = Tensor::cat(vec![pos_src, neg_src], 0);
        let dst = Tensor::cat(vec![pos_dst, neg_dst], 0);
        let labels = Tensor::cat(vec![pos_labels, neg_labels], 0);

        // 4. Forward (no backward)
        let loss = model.forward_loss(src, dst, labels);
        total_loss += loss.into_scalar().to_f32();
    }
    
    if num_batches > 0 {
        total_loss / num_batches as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_cuda::{Cuda, CudaDevice};
    use burn_autodiff::Autodiff;
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::path::Path;
    use plotters::prelude::*;
    use rand::seq::SliceRandom;
    use rand::rng;

    #[derive(serde::Deserialize, Debug)]
    struct Record {
        #[serde(rename = "")]
        _idx: usize,
        src: usize,
        dst: usize,
    }

    fn load_cora_data() -> (Vec<(usize, usize)>, usize) {
        let file_path = Path::new("cora.csv");
        if !file_path.exists() {
             panic!("cora.csv not found at {}", file_path.display());
        }

        let file = File::open(file_path).unwrap();
        let mut rdr = csv::Reader::from_reader(file);
        
        let mut edges = Vec::new();
        let mut nodes = HashSet::new();
        
        for result in rdr.deserialize::<Record>() {
            let record: Record = result.expect("Failed to parse record");
            edges.push((record.src, record.dst));
            nodes.insert(record.src);
            nodes.insert(record.dst);
        }

        let mut node_map = HashMap::new();
        let mut sorted_nodes: Vec<usize> = nodes.into_iter().collect();
        sorted_nodes.sort();
        
        for (i, &node_id) in sorted_nodes.iter().enumerate() {
            node_map.insert(node_id, i);
        }
        
        let num_nodes = sorted_nodes.len();
        println!("Num nodes: {}", num_nodes);
        assert_eq!(num_nodes, 2708, "Expected 2708 nodes from the provided csv file");
        
        let mapped_edges: Vec<(usize, usize)> = edges.into_iter().map(|(s, d)| (node_map[&s], node_map[&d])).collect();
        (mapped_edges, num_nodes)
    }

    fn edges_to_coo(edges: &[(usize, usize)], num_nodes: usize) -> CooGraph {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        
        for (src, dst) in edges {
            row_indices.push(*src);
            col_indices.push(*dst);
        }
        CooGraph::new(row_indices, col_indices, num_nodes)
    }

    #[test]
    fn test_cora_training() {
        let (mut edges, num_nodes) = load_cora_data();
        
        // Train/Test Split
        let mut rng = rng();
        edges.shuffle(&mut rng);
        
        let split_idx = (edges.len() as f32 * 0.8) as usize;
        let train_edges = &edges[..split_idx];
        let test_edges = &edges[split_idx..];
        
        println!("Train edges: {}, Test edges: {}", train_edges.len(), test_edges.len());
        
        let train_graph = edges_to_coo(train_edges, num_nodes);
        let test_graph = edges_to_coo(test_edges, num_nodes);
        
        let device = NdArrayDevice::Cpu;
        type Backend = Autodiff<NdArray>;
        
        let config = LineModelConfig {
            dimension: 16,
            num_nodes,
        };
        
        let (model, history, val_history) = train::<Backend>(
            &device,
            train_graph,
            Some(&test_graph),
            config,
            1000, 
            256, // batch size
            0.01, // lr
        );
        
        // Evaluate
        let test_loss = evaluate(&model, &device, &test_graph, 256);
        println!("Test Loss: {:.4}", test_loss);

        let embeddings = model.get_embeddings();
        let dims = embeddings.dims();
        assert_eq!(dims[0], num_nodes);
        assert_eq!(dims[1], 16);
        
        println!("Training complete. Embeddings shape: {:?}", dims);

        // Plotting with Pantone Colors
        // Pantone 18-1750 Viva Magenta: RGB(187, 38, 73)
        let viva_magenta = RGBColor(187, 38, 73);
        // Pantone 17-3938 Very Peri: RGB(102, 103, 171)
        let very_peri = RGBColor(102, 103, 171);
        // Pantone 19-4052 Classic Blue: RGB(15, 76, 129)
        let classic_blue = RGBColor(15, 76, 129);

        let root = BitMapBackend::new("loss_curve.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        let max_loss = history.iter().cloned().fold(0./0., f32::max).max(
            val_history.iter().map(|&(_, l)| l).fold(0./0., f32::max)
        );
        let min_loss = history.iter().cloned().fold(0./0., f32::min).min(
            val_history.iter().map(|&(_, l)| l).fold(0./0., f32::min)
        );
        
        let mut chart = ChartBuilder::on(&root)
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f32..history.len() as f32, min_loss..max_loss)
            .unwrap();

        chart.configure_mesh()
            .x_desc("Training Steps")
            .y_desc("Loss")
            .label_style(("sans-serif", 15).into_font().color(&very_peri))
            .draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                history.iter().enumerate().map(|(x, y)| (x as f32, *y)),
                &viva_magenta,
            ))
            .unwrap()
            .label("Training Loss")
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &viva_magenta));

        if !val_history.is_empty() {
            chart
                .draw_series(LineSeries::new(
                    val_history.iter().map(|&(x, y)| (x as f32, y)),
                    &classic_blue,
                ))
                .unwrap()
                .label("Validation Loss")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &classic_blue));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
            
        println!("Loss curve saved to loss_curve.png");
    }

    #[test]
    #[ignore = "requires cuda"]
    fn test_cora_training_cuda() {
        let (edges, num_nodes) = load_cora_data();
        let graph = edges_to_coo(&edges, num_nodes);
        
        let config = LineModelConfig {
            dimension: 16,
            num_nodes,
        };
        
        let device = CudaDevice::default();
        type Backend = Autodiff<Cuda>;
        
        let (model, _history, _) = train::<Backend>(
            &device,
            graph,
            None,
            config,
            2, // epochs
            128, // batch size
            0.01, // lr
        );
        
        let embeddings = model.get_embeddings();
        println!("Training complete (CUDA). Embeddings shape: {:?}", embeddings.dims());
        assert_eq!(embeddings.dims()[0], num_nodes);
    }
}
