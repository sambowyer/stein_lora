# Stein LoRA
### Training Low-Rank Adaptors (LoRA) for LLMs using Stein variational gradient descent (SVGD).

## Plan
- Use [Stein VI](https://arxiv.org/pdf/1608.04471) to train [LoRA](https://arxiv.org/pdf/2106.09685) with the hopes that this improves finetuning performance.


## Related work
- Bayesian LoRA ([paper](https://openreview.net/pdf?id=FJiUyzOF1m#subsection.E.1)) ([github](https://github.com/MaximeRobeyns/bayesian_lora/tree/master))
    - Trains using a Laplace approximation to the posterior over LoRA parameters.
- [On Stein Variational Neural Network Ensembles](https://arxiv.org/pdf/2106.10760)
    - Explores how to apply SVGD succesfully on neural networks, suggesting a variety of potentially useful kernels.

## Questions
- How will Stein be affected by the fact that LoRA weights are invariant to permutations (in the typical NN neuron-permutation sense)
    - SVD seems reasonable at first, but might lead to permutation invariance if two or more singular values are very similar (they might then be able to 'swap' without much effect on output)
    - maybe similarity/kernel of big matrix will be easier to work with (and invariant to permuations of LoRA small matrices)
    - OR do kernel on the full lora updates $\Delta W = A B$, as these will be permutation-free AND we can probably compute such a kernel without having to ever fully realise the big update matrix.

## TODO
- Get paralell LoRA adapters working
    - Check param initialisation is sensible
- Get baselines working and reproduce previous LoRA results
- Dig out papers that make stein more reasonable:
    - kernels not necessarily in weight-space, instead on NN outputs or gradients etc.
- Get experiment script set up for easy plug-and-play changes to Stein
- Implement Stein VI
    


