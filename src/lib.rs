pub mod data;
pub mod model;
pub mod training;

pub use data::CooGraph;
pub use model::{LineModel, LineModelConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{LineTrainingConfig, run};
    use burn::optim::AdamWConfig;
    use burn_autodiff::Autodiff;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use rand::rng;
    use rand::seq::SliceRandom;
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

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

        let mapped_edges: Vec<(usize, usize)> = edges
            .into_iter()
            .map(|(s, d)| (node_map[&s], node_map[&d]))
            .collect();
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

        println!(
            "Train edges: {}, Test edges: {}",
            train_edges.len(),
            test_edges.len()
        );

        let train_graph = Arc::new(edges_to_coo(train_edges, num_nodes));
        let test_graph = Arc::new(edges_to_coo(test_edges, num_nodes));

        let device = NdArrayDevice::Cpu;
        type Backend = Autodiff<NdArray>;

        let model_config = LineModelConfig {
            dimension: 16,
            num_nodes,
        };

        // Initialize config with defaults
        let mut training_config = LineTrainingConfig::new(AdamWConfig::new());
        training_config.num_epochs = 1000;

        let artifact_dir = "/tmp/line-burn-test";
        std::fs::create_dir_all(artifact_dir).ok();

        let model = run::<Backend>(
            artifact_dir,
            train_graph,
            test_graph.clone(),
            training_config,
            model_config,
            device,
        );

        let embeddings = model.get_embeddings();
        let dims = embeddings.dims();
        assert_eq!(dims[0], num_nodes);
        assert_eq!(dims[1], 16);

        println!("Training complete. Embeddings shape: {:?}", dims);
    }

    #[test]
    #[ignore = "requires cuda"]
    fn test_cora_training_cuda() {
        let (edges, num_nodes) = load_cora_data();
        // Just use all edges for both train/valid for this smoke test
        let graph = Arc::new(edges_to_coo(&edges, num_nodes));

        let config = LineModelConfig {
            dimension: 16,
            num_nodes,
        };

        let device = burn_cuda::CudaDevice::default();
        type Backend = Autodiff<burn_cuda::Cuda>;

        let mut training_config = LineTrainingConfig::new(AdamWConfig::new());
        training_config.num_epochs = 2;
        training_config.batch_size = 128; // Smaller batch if needed

        // Artifact dir must be unique or we overwrite
        let artifact_dir = "/tmp/line-burn-test-cuda";

        let model = run::<Backend>(
            artifact_dir,
            graph.clone(),
            graph.clone(),
            training_config,
            config,
            device,
        );

        let embeddings = model.get_embeddings();
        println!(
            "Training complete (CUDA). Embeddings shape: {:?}",
            embeddings.dims()
        );
        assert_eq!(embeddings.dims()[0], num_nodes);
    }
}
