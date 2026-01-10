use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Distribution, Int, Tensor, backend::Backend},
};

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
        assert_eq!(
            row_indices.len(),
            col_indices.len(),
            "Row and column indices must have same length"
        );
        let num_edges = row_indices.len();
        Self {
            row_indices,
            col_indices,
            num_nodes,
            num_edges,
        }
    }
}

pub type LineItem = (usize, usize);

impl Dataset<LineItem> for CooGraph {
    fn get(&self, index: usize) -> Option<LineItem> {
        if index >= self.num_edges {
            return None;
        }
        Some((self.row_indices[index], self.col_indices[index]))
    }

    fn len(&self) -> usize {
        self.num_edges
    }
}

#[derive(Clone, Debug)]
pub struct LineBatch<B: Backend> {
    pub src: Tensor<B, 1, Int>,
    pub dst: Tensor<B, 1, Int>,
    pub labels: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct LineBatcher {
    num_nodes: usize,
    negative_samples: usize,
}

impl LineBatcher {
    pub fn new(num_nodes: usize, negative_samples: usize) -> Self {
        Self {
            num_nodes,
            negative_samples,
        }
    }
}

impl<B: Backend> Batcher<B, LineItem, LineBatch<B>> for LineBatcher {
    fn batch(&self, items: Vec<LineItem>, device: &B::Device) -> LineBatch<B> {
        let batch_size = items.len();

        // 1. Positive edges
        let (pos_src, pos_dst): (Vec<i32>, Vec<i32>) =
            items.iter().map(|&(u, v)| (u as i32, v as i32)).unzip();

        let pos_src = Tensor::from_ints(&pos_src[..], device);
        let pos_dst = Tensor::from_ints(&pos_dst[..], device);
        let pos_labels = Tensor::ones([batch_size], device);

        // 2. Negative edges
        // We generate `negative_samples` per positive edge.
        // Total negative samples = batch_size * negative_samples
        let num_neg = batch_size * self.negative_samples;

        // Random sampling for negative edges
        // Note: This naive sampling might sample existing edges or duplicates,
        // which is standard for simple LINE implementations / word2vec.
        let neg_src = Tensor::<B, 1>::random(
            [num_neg],
            Distribution::Uniform(0.0, self.num_nodes as f64),
            device,
        )
        .int();

        let neg_dst = Tensor::<B, 1>::random(
            [num_neg],
            Distribution::Uniform(0.0, self.num_nodes as f64),
            device,
        )
        .int();

        let neg_labels = Tensor::zeros([num_neg], device);

        // If negative_samples > 1, we need to repeat pos_src/pos_dst if we wanted to pair them specifically,
        // but LINE typically just treats them as a bag of pairs.
        // However, to keep it balanced if negative_samples is 1, we just concat.

        let src = Tensor::cat(vec![pos_src, neg_src], 0);
        let dst = Tensor::cat(vec![pos_dst, neg_dst], 0);
        let labels = Tensor::cat(vec![pos_labels, neg_labels], 0);

        LineBatch { src, dst, labels }
    }
}
