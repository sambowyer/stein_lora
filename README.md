# Stein LoRA
### Training Low-Rank Adaptors (LoRA) for LLMs using Stein variational gradient descent (SVGD).

## Plan
- Use [Stein VI](https://arxiv.org/pdf/1608.04471) to train [LoRA](https://arxiv.org/pdf/2106.09685) with the hopes that this improves finetuning performance both on Q&A datasets and for RLHF.


## Related work
- Bayesian LoRA ([paper](https://openreview.net/pdf?id=FJiUyzOF1m#subsection.E.1)) ([github](https://github.com/MaximeRobeyns/bayesian_lora/tree/master))
    - Finds posterior post-hoc using a Laplace approximation and KFAC.
- [On Stein Variational Neural Network Ensembles](https://arxiv.org/pdf/2106.10760) ([implementation here](https://github.com/Pascal314/SmalldataMNIST))
    - Explores how to apply SVGD succesfully on neural networks, suggesting a variety of potentially useful kernels.

## Questions
- Why is the autograd of A (and of B) exactly equal to the hand-derived update of A (and of B) multiplied by -58.982391357421875? (or in a couple of cases -58.98240280151367 or -58.982398986816406)
    - It is always 58.9824... regardless of sigma (and LoRA layer)
    - It scales linearly with K
    - It is independent of r

## TODO
- Get baselines working and reproduce previous LoRA results
- Dig out papers that make stein more reasonable:
    - With an added SGLD noise term in each update ([paper](https://arxiv.org/pdf/2106.10760)) $\sum_{j=1}^n \sqrt{\frac{2 \mathcal{K}_{ij}}{\epsilon_t}} \eta_j$ where
        - $\mathcal{K}_{ij} = \frac{1}{n} k(W_i, W_j) \mathbb{I}_{d \times d}$
        - $\eta_j \sim \mathcal{N}(0, \mathbb{I}_{d \times d})$
        - $\epsilon_t$ is step size/learning rate
- use autodiff and check updates match with hand-computed gradients
    - compare speed
- use and compare against LoRA variants:
    - different initialisations (currently $A \sim \mathcal{N}(0, r^{-2}), B=0$ as is typical for LoRAs) e.g. pissa 