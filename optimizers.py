import torch
import math

class MySGD(torch.optim.Optimizer):
    """
    Implements a standard Stochastic Gradient Descent (SGD) optimizer.

    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining
                           parameter groups.
        lr (float): Learning rate (default: 0.01).
        momentum (float, optional): Momentum factor (default: 0).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        dampening (float, optional): Dampening for momentum (default: 0).
        nesterov (bool, optional): Enables Nesterov momentum (default: False).
    """
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(MySGD, self).__init__(params, defaults)

        # Initialize momentum buffer if momentum > 0
        if momentum != 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None and p.grad.is_sparse:
                        raise RuntimeError("SGD with momentum does not support sparse gradients")
                    state = self.state[p]
                    state['momentum_buffer'] = torch.clone(p.data).detach().zero_()

    @torch.no_grad() # Important: operations inside step should not be tracked
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): # Ensure grads are enabled for closure
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        # This case should ideally be handled in __init__ or first step if grad was None initially
                        state['momentum_buffer'] = torch.clone(d_p).detach() # Initialize if not present
                    else:
                        state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(state['momentum_buffer'], alpha=momentum)
                    else:
                        d_p = state['momentum_buffer']
                
                p.data.add_(d_p, alpha=-lr)
        
        return loss

class MyAdam(torch.optim.Optimizer):
    """
    Implements the Adam algorithm.

    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining
                           parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing
                                               running averages of gradient and its
                                               square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator to improve
                               numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        amsgrad (bool, optional): Whether to use the AMSGrad variant of this
                                  algorithm (default: False).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MyAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.grad.is_sparse:
                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                state = self.state[p]
                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                state['step'] += 1
                
                # Apply weight decay if specified
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2) # use .conj() for complex numbers

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(eps)

                bias_correction1 = 1 - beta1 ** state['step']
                step_size = lr / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class MyAdaGrad(torch.optim.Optimizer):
    """
    Implements the Adagrad algorithm.

    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining
                           parameter groups.
        lr (float, optional): Learning rate (default: 1e-2).
        lr_decay (float, optional): Learning rate decay (default: 0).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        initial_accumulator_value (float, optional): Starting value for the
                                                     accumulators (default: 0).
        eps (float, optional): Term added to the denominator to improve
                               numerical stability (default: 1e-10).
    """
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, 
                 initial_accumulator_value=0, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(f"Invalid initial_accumulator_value value: {initial_accumulator_value}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, 
                        initial_accumulator_value=initial_accumulator_value, eps=eps)
        super(MyAdaGrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p, initial_accumulator_value, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if weight_decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients in Adagrad")
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Learning rate decay
                clr = lr / (1 + (state['step'] - 1) * lr_decay)

                # Accumulate squared gradients
                if grad.is_sparse:
                    # Adagrad's sparse update logic is a bit more involved.
                    # For simplicity, we'll keep this dense for now.
                    # A full sparse implementation would update only the affected indices in state['sum'].
                    grad_values = grad._values()
                    state['sum'].index_add_(0, grad._indices()[0], grad_values.pow(2))
                    std_slice = state['sum'].sparse_mask(grad)._values().sqrt_().add_(eps)
                    p.data.index_add_(0, grad._indices()[0], -clr * grad_values / std_slice)
                else:
                    state['sum'].addcmul_(grad, grad, value=1)
                    std = state['sum'].sqrt().add_(eps)
                    p.data.addcdiv_(grad, std, value=-clr)
        return loss