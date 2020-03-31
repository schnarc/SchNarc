import torch
import numpy as np

import schnetpack as spk


class MeanSquaredError(spk.metrics.MeanSquaredError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'MSE_' + target if name is None else name
        super(MeanSquaredError, self).__init__(target, model_output=model_output, bias_correction=bias_correction,
                                                    name=name, element_wise=element_wise)

    def _get_diff(self, y, yp):
        diff = y - yp
        diff = (torch.abs(diff))
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff


class RootMeanSquaredError(MeanSquaredError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'RMSE_' + target if name is None else name
        super(RootMeanSquaredError, self).__init__(target, model_output,
                                                        bias_correction, name,
                                                        element_wise=element_wise)

    def aggregate(self):
        return np.sqrt(self.l2loss / self.n_entries)


class MeanAbsoluteError(spk.metrics.MeanAbsoluteError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'MAE_' + target if name is None else name
        super(MeanAbsoluteError, self).__init__(target, model_output=model_output, bias_correction=bias_correction,
                                                     name=name, element_wise=element_wise)

    def _get_diff(self, y, yp):
        diff = y - yp
        diff = (torch.abs(diff))

        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

#class MeanAbsoluteError_forces(spk.metrics.MeanAbsoluteError):
#
#    def __init__(self, target, model_output=None, bias_correction=None,
#                 name=None, element_wise=False):
#        name = 'MAE_' + target if name is None else name
#        super(MeanAbsoluteError_forces, self).__init__(target, model_output=model_output, has_forces=True, bias_correction=bias_correction,
#                                                     name=name, element_wise=element_wise)
#        print(has_forces)
#    def _get_diff(self, y, yp,has_forces):
#        diff = y - yp
#        diff = (torch.abs(diff))
#
#        if self.bias_correction is not None:
#            diff += self.bias_correction
#        return diff

#class RootMeanSquaredError_forces(MeanSquaredError):
#
#    def __init__(self, target, model_output=None, bias_correction=None,
#                 name=None, element_wise=False):
#        name = 'PhRMSE_' + target if name is None else name
#        super(RootMeanSquaredError_forces, self).__init__(target, model_output,
#                                                        bias_correction, name,
#                                                        element_wise=element_wise)
#
#    def aggregate(self):
#        return np.sqrt(self.l2loss / self.n_entries)


class PhaseMeanSquaredError(spk.metrics.MeanSquaredError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'PhMSE_' + target if name is None else name
        super(PhaseMeanSquaredError, self).__init__(target, model_output=model_output, bias_correction=bias_correction,
                                                    name=name, element_wise=element_wise)

    def _get_diff(self, y, yp):
        diff_a = torch.abs(y - yp)
        diff_b = torch.abs(y + yp)
        diff = torch.min(diff_a, diff_b)
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff


class PhaseRootMeanSquaredError(PhaseMeanSquaredError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'PhRMSE_' + target if name is None else name
        super(PhaseRootMeanSquaredError, self).__init__(target, model_output,
                                                        bias_correction, name,
                                                        element_wise=element_wise)

    def aggregate(self):
        return np.sqrt(self.l2loss / self.n_entries)


class PhaseMeanAbsoluteError(spk.metrics.MeanAbsoluteError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'PhMAE_' + target if name is None else name
        super(PhaseMeanAbsoluteError, self).__init__(target, model_output=model_output, bias_correction=bias_correction,
                                                     name=name, element_wise=element_wise)

    def _get_diff(self, y, yp):
        diff_a = torch.abs(y - yp)
        diff_b = torch.abs(y + yp)
        diff = torch.min(diff_a, diff_b)
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff
class PhaseRootMeanSquaredError_vec(PhaseMeanSquaredError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'PhRMSE_' + target if name is None else name
        super(PhaseRootMeanSquaredError_vec, self).__init__(target, model_output,
                                                        bias_correction, name,
                                                        element_wise=element_wise)

    def aggregate(self):
        return np.sqrt(self.l2loss / self.n_entries)


class PhaseMeanAbsoluteError_vec(spk.metrics.MeanAbsoluteError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'PhMAE_' + target if name is None else name
        super(PhaseMeanAbsoluteError_vec, self).__init__(target, model_output=model_output, bias_correction=bias_correction,
                                                     name=name, element_wise=element_wise)

    def _get_diff(self, y, yp):
        #assume that each state combination was found successfully
        diff = 0.
        for batch_sample in range(y.shape[0]):
            for istate_combi in range(y.shape[1]):
                diff_a = (y[batch_sample,istate_combi,:,:] - yp[batch_sample,istate_combi,:,:] )
                diff_b = (y[batch_sample,istate_combi,:,:] + yp[batch_sample,istate_combi,:,:] )
                mean_a = torch.mean(torch.abs(diff_a))
                mean_b = torch.mean(torch.abs(diff_b))
                if mean_a <= mean_b:
                    diff += diff_a
                else:
                    diff += diff_b
        if self.bias_correction is not None:
            diff += self.bias_correction
        diff = diff/(y.shape[0]*y.shape[1])
        return diff

class PhaseMeanSquaredError_vec(spk.metrics.MeanSquaredError):

    def __init__(self, target, model_output=None, bias_correction=None,
                 name=None, element_wise=False):
        name = 'PhMSE_' + target if name is None else name
        super(PhaseMeanSquaredError_vec, self).__init__(target, model_output=model_output, bias_correction=bias_correction,
                                                     name=name, element_wise=element_wise)

    def _get_diff(self, y, yp):
        #assume that each state combination was found successfully
        diff = 0.
        for batch_sample in range(y.shape[0]):
            for istate_combi in range(y.shape[1]):
                diff_a = (y[batch_sample,istate_combi,:,:] - yp[batch_sample,istate_combi,:,:] )
                diff_b = (y[batch_sample,istate_combi,:,:] + yp[batch_sample,istate_combi,:,:] )
                mean_a = torch.mean((diff_a)**2)
                mean_b = torch.mean((diff_b)**2)
                if mean_a <= mean_b:
                    diff += diff_a**(1/2)
                else:
                    diff += diff_b**(1/2)
        if self.bias_correction is not None:
            diff += self.bias_correction
        diff = (diff/(y.shape[0]*y.shape[1]))**2
        return diff
