# Stein LoRA
### Training Low-Rank Adaptors (LoRA) for LLMs using Stein variational gradient descent (SVGD).

## Plan
- Use [Stein VI](https://arxiv.org/pdf/1608.04471) to train [LoRA](https://arxiv.org/pdf/2106.09685) with the hopes that this improves finetuning performance both on Q&A datasets and for RLHF.


## Related work
- Bayesian LoRA ([paper](https://openreview.net/pdf?id=FJiUyzOF1m#subsection.E.1)) ([github](https://github.com/MaximeRobeyns/bayesian_lora/tree/master))
    - Trains using a Laplace approximation to the posterior over LoRA parameters.
- [On Stein Variational Neural Network Ensembles](https://arxiv.org/pdf/2106.10760) ([implementation here](https://github.com/Pascal314/SmalldataMNIST))
    - Explores how to apply SVGD succesfully on neural networks, suggesting a variety of potentially useful kernels.

## Questions
- 

## TODO
- Get baselines working and reproduce previous LoRA results
- Dig out papers that make stein more reasonable:
    - With an added SGLD noise term in each update ([paper](https://arxiv.org/pdf/2106.10760)) $\sum_{j=1}^n \sqrt{\frac{2 \mathcal{K}_{ij}}{\epsilon_t}} \eta_j$ where
        - $ \mathcal{K}_{ij} = \frac{1}{n} k(W_i, W_j) \mathbb{I}_{d \times d}$
        - $\eta_j \sim \mathcal{N}(0, \mathbb{I}_{d \times d})$
        - $\epsilon_t$ is step size/learning rate
- use autodiff and check updates match with hand-computed gradients
    - compare speed


