# Stein LoRA
### Training Low-Rank Adaptors (LoRA) for LLMs using Stein variational gradient descent (SVGD).

## Plan
- Use [Stein VI](https://arxiv.org/pdf/1608.04471) to train [LoRA](https://arxiv.org/pdf/2106.09685) with the hopes that this improves finetuning performance.


## Related work
- Bayesian LoRA ([paper](https://openreview.net/pdf?id=FJiUyzOF1m#subsection.E.1)) ([github](https://github.com/MaximeRobeyns/bayesian_lora/tree/master))
    - Trains using a Laplace approximation to the posterior over LoRA parameters.

## Questions
- How will Stein be affected by the fact that LoRA weights are invariant to permutations (in the typical NN neuron-permutation sense)
    - SVD seems reasonable at first, but might lead to permutation invariance if two or more singular values are very similar (they might then be able to 'swap' without much effect on output)
    - maybe similarity/kernel of big matrix will be easier to work with (and invariant to permuations of LoRA small matrices)
- Is it easy to store multiple versions of LoRA weights and plug them in to our model as needed (using standard librarires e.g. from huggingface?)
    - if not, we'll just have to implement LoRA outselves, but this shouldn't be too bad

## TODO
- Get baselines working and reproduce previous LoRA results
- Dig out papers that make stein more reasonable:
    - kernels not necessarily in weight-space, instead on NN outputs or gradients etc.
- Test the swapping-out of LoRA parameters (which act as the 'particles' in our setting)
- Get experiment script set up for easy plug-and-play changes to Stein
- Implement Stein VI
    


