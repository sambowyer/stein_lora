import torch
from .multi_lora import MultiLoraLayer, nn_ParallelLinear

def MSD(adapter: MultiLoraLayer, i,j):
    
    assert len(adapter.active_adapters) == 1
    active_adapter = adapter.active_adapters[0]

    lora_A = adapter.lora_A[active_adapter]
    lora_B = adapter.lora_B[active_adapter]

    d_in = lora_A.in_features
    d_out = lora_B.out_features

    def trace_computation(i_, j_):
        return torch.trace(torch.matmul(
            torch.matmul(lora_A.weight[j_].T, lora_A.weight[i_]),
            torch.matmul(lora_B.weight[i_], lora_B.weight[j_].T)
            )
        )
    return (trace_computation(i, i) + trace_computation(j, j) - 2*trace_computation(i, j)) / (d_in * d_out)

def RBF_kernel(sigma):
    def kernel(adapter: MultiLoraLayer, i, j):
        return torch.exp(-MSD(adapter, i, j) / (2*sigma**2))
    return kernel

def svgd_step(adapter: MultiLoraLayer, kernel, lr, gamma):

    assert len(adapter.active_adapters) == 1
    active_adapter = adapter.active_adapters[0]

    lora_A = adapter.lora_A[active_adapter]
    lora_B = adapter.lora_B[active_adapter]

    K = lora_A.K
    assert K == lora_B.K

    # get log likelihood (loss) gradients
    log_lik_grad_A = lora_A.weight.grad.clone()
    log_lik_grad_B = lora_B.weight.grad.clone()
    # breakpoint()

    # reset grads
    lora_A.weight.grad.zero_()
    lora_B.weight.grad.zero_()

    # # get log prior gradients
    # lora_A_prior_grad = lora_A.weight
    # lora_B_prior_grad = lora_B.weight

    lora_A_log_prior_grad = torch.zeros((K,))
    lora_B_log_prior_grad = torch.zeros((K,))

    # # reset grads
    # lora_A.weight.grad.zero_()
    # lora_B.weight.grad.zero_()


    # construct weight-update tensor
    update_A = torch.zeros_like(lora_A.weight)
    update_B = torch.zeros_like(lora_B.weight)
 
    # get evaluations and gradients of kernel
    for i in range(K):
        for j in range(K):
            kernel_val = kernel(adapter, i, j)

            # breakpoint()
            
            update_A[i].add_(-lr * (kernel_val * (log_lik_grad_A[j] + lora_A_log_prior_grad[j]) - (gamma/K)*lora_A.weight.grad[j]))
            update_B[i].add_(-lr * (kernel_val * (log_lik_grad_B[j] + lora_B_log_prior_grad[j]) - (gamma/K)*lora_B.weight.grad[j]))

            # reset grads
            lora_A.weight.grad.zero_()
            lora_B.weight.grad.zero_()

    # if (lora_A.weight.data == 0).all():
    #     print("lora_A   all zero!")
    # if (lora_B.weight.data == 0).all():
    #     print("lora_B all zero!")
    # if (update_A.data == 0).all():
    #     print("update_A all zero!")
    # if (update_B.data == 0).all():
    #     print("update_B all zero!")


    # apply update
    with torch.no_grad():
        lora_A.weight.add_(update_A)
        lora_B.weight.add_(update_B)

    # # reset grads
    # lora_A.weight.grad.zero_()
    # lora_B.weight.grad.zero_()

class SVGD(torch.optim.Optimizer):
    def __init__(self, model, lr, kernel, gamma):
        defaults = dict(lr=lr, kernel=kernel, gamma=gamma)

        self.modules = [module for module in model.modules() if isinstance(module, MultiLoraLayer)]

        super(SVGD, self).__init__(model.parameters(), defaults)
        # self.param_groups = self.modules

    def step(self, closure=None):
        for module in self.modules:
            svgd_step(module, self.defaults['kernel'], self.defaults['lr'], self.defaults['gamma'])
            
        # breakpoint()
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue
        #         d_p = p.grad
        #         if d_p.is_sparse:
        #             raise RuntimeError('SVGD does not support sparse gradients')
        #         svgd_step(p, group['kernel'], group['lr'], group['gamma'])