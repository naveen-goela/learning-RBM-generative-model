%
% compute_CD_update.m
%
% Description:
% Compute a gradient update for an RBM using 
% contrastive divergence (CD) methods.  
%
% USAGE:
% [gradient] = compute_CD_update(bRBM, data_sample)
%
% INPUTS: 
% data_sample = one sample of visible units of an RBM.
% bRBM = binary restricted Bolztmann machine.
%   bRBM.W   = weight connections matrix of RBM.
%   bRBM.b_v = bias vector for visible units of RBM.
%   bRBM.b_h = bias vector for hidden units of RBM.
%
% OUTPUTS:
% gradient  = gradient computed via CD update. 
%   gradient.W   = gradient update for weights.
%   gradient.b_v = gradient update for bias for visible units.
%   gradient.b_h = gradient update for bias for hidden units. 
%
% Author: N. Goela
% Date: January 31, 2015

function [gradient] = compute_CD_update(bRBM, data_sample)

% Number of Gibbs steps within CD update method.
num_gibbs_steps = 12; 

% Apply contrastive divergence method starting from
% a training data sample of binary visible units. 
h_data = sample_bRBM_hgv(bRBM, data_sample);
v_free = data_sample;
h_free = h_data;

% Gibbs steps for CD update.
for k=1:num_gibbs_steps
    v_free = sample_bRBM_vgh(bRBM, h_free);
    h_free = sample_bRBM_hgv(bRBM, v_free); 
end

% Obtain CD gradient.
gradient.W   = data_sample*h_data' - v_free*h_free';
gradient.b_v = data_sample - v_free;
gradient.b_h = h_data - h_free; 



