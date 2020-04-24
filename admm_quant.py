import numpy as np
import torch


def apply_quantization(args, model, name_list, device=None):
    for name, weight in model.named_parameters():
        if name in name_list:
            quantized_weight = weight.cpu().detach().numpy()
            Q = model.Q[name].cpu().detach()
            if args.quant_type == "binary":
                quantized_weight[quantized_weight > 0] = model.alpha[name]
                quantized_weight[quantized_weight <= 0] = -model.alpha[name]
                weight.data = torch.Tensor(quantized_weight).to(device)

            elif args.quant_type == "ternary":
                quantized_weight = np.where(quantized_weight > 0.5 * model.alpha[name], model.alpha[name],
                                            quantized_weight)
                quantized_weight = np.where(quantized_weight < -0.5 * model.alpha[name], -model.alpha[name],
                                            quantized_weight)
                quantized_weight = np.where(
                    (quantized_weight <= 0.5 * model.alpha[name]) & (quantized_weight >= -0.5 * model.alpha[name]), 0,
                    quantized_weight)
                weight.data = torch.Tensor(quantized_weight).to(device)

            elif args.quant_type == "fixed":
                quantized_weight = model.alpha[name] * Q
                half_num_bits = args.num_bits - 1
                centroids = []
                for value in range(-2 ** half_num_bits + 1, 2 ** half_num_bits):
                    centroids.append(value)

                for i, value in enumerate(centroids):
                    if i == 0:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] < value + 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    elif i == len(centroids) - 1:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] >= value - 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    else:
                        quantized_weight = np.where((quantized_weight / model.alpha[name] >= value - 0.5) & (
                                quantized_weight / model.alpha[name] < value + 0.5), value * model.alpha[name],
                                                    quantized_weight)
                weight.data = torch.Tensor(quantized_weight).to(device)
            else:
                raise ValueError("Type '{}' is not supported quantized type!".format(args.quant_type))


def project_to_centroid(args, model, W, name, device, print):
    U = model.U[name].cpu().detach().numpy()
    Q = model.Q[name].cpu().detach().numpy()

    alpha = model.alpha[name]

    if abs(alpha) < 1e-8:
        alpha = 1

    num_iter_quant = 6  # caffe code 20, in paper less than 5 iterations

    if args.quant_type == "binary":
        Q = np.where((W + U) > 0, 1, -1)
        alpha = np.sum(np.multiply((W + U), Q))
        QtQ = np.sum(Q ** 2)
        alpha /= QtQ

    elif args.quant_type == "ternary":
        for n in range(num_iter_quant):
            pre_alpha = alpha
            Q = np.where((W + U) / alpha > 0.5, 1, Q)
            Q = np.where((W + U) / alpha < -0.5, -1, Q)
            Q = np.where(((W + U) / alpha >= -0.5) & ((W + U) / alpha <= 0.5), 0, Q)
            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.sum(Q ** 2)
            alpha /= QtQ
            if args.verbose:
                print("at layer {}, alpha({})-alpha({}): {}".format(name, n, n - 1, alpha - pre_alpha))

    # 2bits [-1, 0, 1] 3bits [-3, -2, -1, 0, 1, 2, 3]
    # num_bits >= 3 bits
    elif args.quant_type == "fixed":
        half_num_bits = args.num_bits - 1
        centroids = []
        for value in range(-2 ** half_num_bits + 1, 2 ** half_num_bits):
            centroids.append(value)

        for n in range(num_iter_quant):
            Q = np.where(np.round((W + U) / alpha) <= centroids[0], centroids[0], Q)
            Q = np.where(np.round((W + U) / alpha) >= centroids[-1], centroids[-1], Q)
            Q = np.where((np.round((W + U) / alpha) < centroids[-1]) & (np.round((W + U) / alpha) > centroids[0]),
                         np.round((W + U) / alpha), Q)

            # for i, value in enumerate(centroids):
            #
            #     if i == 0:
            #         Q = np.where(((W + U) / alpha) < (value + 0.5), value, Q)
            #     elif i == len(centroids) - 1:
            #         Q = np.where(((W + U) / alpha) >= (value - 0.5), value, Q)
            #     else:
            #         Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < (value + 0.5)), value,
            #                      Q)

            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.multiply(Q, Q)
            QtQ = np.sum(QtQ)
            if QtQ < 1e-8:
                return Q, 1, Q
            alpha /= QtQ

    model.U[name] = torch.Tensor(U).to(device)
    model.Q[name] = torch.Tensor(Q).to(device)
    model.Z[name] = alpha * model.Q[name]
    model.alpha[name] = alpha

    return model.Z[name], model.alpha[name], model.Q[name]


def admm_initialization(args, model, device, name_list, print):
    model.alpha = {}
    model.rhos = {}
    model.U = {}
    model.Q = {}  # alpha * Q = Z
    model.Z = {}

    for name, param in model.named_parameters():
        print(name)
        if name in name_list:
            model.rhos[name] = args.reg_lambda
            weight = param.cpu().detach().numpy()
            if args.quant_type == "binary" or args.quant_type == "ternary":
                model.alpha[name] = np.mean(np.abs(weight))  # initialize alpha
            elif args.quant_type == "fixed":
                model.alpha[name] = np.mean(np.abs(weight))
                half_num_bits = args.num_bits - 1
                model.alpha[name] = model.alpha[name] / ((2 ** half_num_bits - 1) / 2)
                # model.alpha[name] = np.max(np.abs(weight))
                # half_num_bits = args.num_bits - 1
                # model.alpha[name] /= half_num_bits ** 2 - 1
            if model.alpha[name] < 1e-8:
                model.alpha[name] = 1
                model.U[name] = torch.zeros(param.shape).to(device)
                model.Q[name] = torch.zeros(param.shape).to(device)
                model.Z[name] = torch.zeros(param.shape).to(device)
            else:
                model.U[name] = torch.zeros(param.shape).to(device)
                model.Q[name] = torch.zeros(param.shape).to(device)
                updated_Z, updated_aplha, updated_Q = project_to_centroid(args, model, weight, name, device, print)
                model.Z[name] = updated_Z


def append_admm_loss(model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if name not in model.rhos:
            continue
        admm_loss[name] = 0.5 * model.rhos[name] * (torch.norm(W - model.Z[name] + model.U[name], p=2) ** 2)

    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def z_u_update(args, model, device, epoch, batch_idx, name_list, print):
    # epoch start from zero
    if epoch != 0 and epoch % args.admm_epochs == 0 and batch_idx == 0:
        print("Updating Z, U!!!!!!")
        for name, W in model.named_parameters():
            if name not in name_list:
                continue
            Z_prev = torch.clone(model.Z[name])
            weight = W.cpu().detach().numpy()
            updated_Z, updated_alpha, updated_Q = project_to_centroid(args, model, weight, name,
                                                                      device,
                                                                      print)  # equivalent to Euclidean Projection
            model.rhos[name] = model.rhos[name] * 1.5

            # bound1 = torch.sqrt(torch.sum((W - model.Z[name]) ** 2)).item()
            # bound2 = torch.sqrt(torch.sum((model.Z[name] - Z_prev) ** 2)).item()
            # mu_value = 0.5
            # tau_value = 2  # 1+tau tau belongs (0,1)
            #
            # if mu_value * bound1 >= bound2:
            #     model.rhos[name] = model.rhos[name] * tau_value
            # elif bound1 < mu_value * bound2:
            #     model.rhos[name] = model.rhos[name] / tau_value
            # else:
            #     model.rhos[name] = model.rhos[name]

            if (args.verbose):
                print("at layer {}. W(k+1)-Z(k+1): {}".format(name, torch.sqrt(
                    torch.sum((W - model.Z[name]) ** 2)).item()))
                print("at layer {}, Z(k+1)-Z(k): {}".format(name, torch.sqrt(
                    torch.sum((model.Z[name] - Z_prev) ** 2)).item()))
                print("at layer {}, alpha(k+1):{}, rho(k+1):{}".format(name, model.alpha[name], model.rhos[name]))

            model.U[name] = W - model.Z[name] + model.U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def test_sparsity(model, print):
    """
         test sparsity for every involved layer and the overall compression rate
         """

    def sparsity_check(kernel):
        """
        Return the sparsity of the input torch.tensor
        sparsity = #zero_elements/#tot_elements
        """

        return density, num_zeros, num_tot

    layer = 0
    tot_param = 0
    tot_zeros = 0
    for name, param in model.named_parameters():
        if param.shape.__len__() is 4:
            layer += 1
            num_tot = param.detach().cpu().numpy().size
            num_zeros = param.detach().cpu().eq(0).sum().item()
            sparsity = (num_zeros / num_tot) * 100
            density = 100 - sparsity
            tot_param += num_tot
            tot_zeros += num_zeros
            print("{}, {}, {:.2f}, {}, {}% ".format(layer, name, density, num_tot, num_zeros))
    print("Total sparsity: {:.4f}".format(tot_zeros / tot_param))
