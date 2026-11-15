# Appendices

## Appendix A: CORINE Land Cover (CLC) Multi-Label Taxonomy

The BigEarthNet-14K dataset utilizes the CORINE Land Cover (CLC) database nomenclature, representing a hierarchical multi-label taxonomy of 43 distinct environmental and urban classes. The retrieval pipeline tracks semantic overlap (Jaccard Index / Dice Coefficient) across these classes to evaluate true geographic generalization.

### Level 1: Artificial Surfaces
*   **1.1 Urban fabric:** Continuous urban fabric, Discontinuous urban fabric.
*   **1.2 Industrial, commercial and transport units:** Industrial or commercial units, Road and rail networks and associated land, Port areas, Airports.
*   **1.3 Mine, dump and construction sites:** Mineral extraction sites, Dump sites, Construction sites.
*   **1.4 Artificial, non-agricultural vegetated areas:** Green urban areas, Sport and leisure facilities.

### Level 2: Agricultural Areas
*   **2.1 Arable land:** Non-irrigated arable land, Permanently irrigated land, Rice fields.
*   **2.2 Permanent crops:** Vineyards, Fruit trees and berry plantations, Olive groves.
*   **2.3 Pastures:** Pastures.
*   **2.4 Heterogeneous agricultural areas:** Annual crops associated with permanent crops, Complex cultivation patterns, Land principally occupied by agriculture with significant areas of natural vegetation, Agro-forestry areas.

### Level 3: Forest and Semi-Natural Areas
*   **3.1 Forests:** Broad-leaved forest, Coniferous forest, Mixed forest.
*   **3.2 Scrub and/or herbaceous vegetation associations:** Natural grasslands, Moors and heathland, Sclerophyllous vegetation, Transitional woodland-shrub.
*   **3.3 Open spaces with little or no vegetation:** Beaches/dunes/sands, Bare rocks, Sparsely vegetated areas, Burnt areas, Glaciers and perpetual snow.

### Level 4 & 5: Wetlands and Water Bodies
*   **4.1 Inland wetlands:** Inland marshes, Peat bogs.
*   **4.2 Maritime wetlands:** Salt marshes, Salines, Intertidal flats.
*   **5.1 Inland waters:** Water courses, Water bodies.
*   **5.2 Marine waters:** Coastal lagoons, Estuaries, Sea and ocean.

---

## Appendix B: Hyperparameter Grid and Hardware Configuration

The system was optimized for a single-GPU cloud runtime (e.g., NVIDIA T4 / L4 on Google Colab) utilizing Automatic Mixed Precision (AMP) to maximize batch throughput and minimize VRAM consumption.

| Hyperparameter / Configuration | Value / Setting |
| :--- | :--- |
| **Hardware / Engine** | 1x NVIDIA T4 (16GB VRAM), PyTorch 2.x |
| **Precision** | `torch.cuda.amp.GradScaler` (FP16 / FP32 Mixed) |
| **Epochs** | 30 |
| **Batch Size ($B$)** | 32 (yielding a $32 \times 32$ contrastive matrix) |
| **Optimizer** | AdamW |
| **Base Learning Rate ($\eta$)** | $3 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-3}$ |
| **LR Schedule** | Cosine Annealing ($\eta_{\text{min}} = 10^{-6}$) |
| **EMA Target Momentum ($m$)** | $0.996$ |
| **GeM Initial Exponent ($p$)** | $3.0$ (Learnable) |
| **InfoNCE Temperature ($\tau$)** | $0.1$ |
| **IoU False-Negative Threshold ($\tau_{\text{IoU}}$)** | $0.40$ |

---

## Appendix C: Core PyTorch Implementation Listings

### C.1 ConvNeXt-Tiny Stem Channel Adaptation
This snippet demonstrates the physical energy-conserving channel adaptation for the SAR (2-channel) and Optical (10-channel) branches.

> **For beginners:** ConvNeXt was trained on 3-channel (RGB) images. Our SAR images have 2 channels (VV, VH) and our optical images have 10 channels. We need to surgically modify the first layer to accept our data while keeping all the learned feature-detection knowledge from ImageNet pre-training. The trick is to do it in a way that preserves the total "energy" (variance) of the signal passing through.

```python
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def adapt_convnext_stem(model, in_channels, modality='SAR'):
    """
    Replaces the first convolutional layer (the "stem") of a ConvNeXt-Tiny
    model so it can accept satellite images with non-standard channel counts.

    WHY THIS IS NECESSARY:
    ConvNeXt was originally trained on 3-channel RGB ImageNet photos.
    Our inputs are:
      - SAR (Sentinel-1):     2 channels  [VV, VH]  in dB scale
      - Optical (Sentinel-2): 10 channels [B02..B12] in reflectance scale

    We want to keep the pre-trained weights (which learned to detect edges,
    textures, shapes) but adapt only the very first layer to handle our
    channel count. This is called "transfer learning with channel adaptation."

    Args:
        model:       A ConvNeXt-Tiny instance (from torchvision)
        in_channels: Number of input channels (2 for SAR, 10 for Optical)
        modality:    'SAR' or 'Optical' — controls the adaptation strategy

    Returns:
        The same model with its first conv layer replaced.
    """

    # Grab the original first convolutional layer (the "stem" / "patchify" layer).
    # In ConvNeXt-Tiny, model.features[0][0] is a Conv2d(3, 96, kernel=4, stride=4).
    # It converts a 3-channel 224×224 image into 96 feature maps of 56×56.
    original_stem = model.features[0][0]

    # Extract the learned weights from that original layer.
    # Shape: [96 output filters, 3 input channels, 4×4 kernel] = [96, 3, 4, 4]
    # Each of the 96 filters learned to detect a specific low-level pattern.
    original_weights = original_stem.weight.data  # shape: [96, 3, 4, 4]

    # Create a new first conv layer with our desired number of input channels.
    # We keep all other settings the same (96 output filters, 4×4 kernel, stride 4).
    new_stem = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)

    if modality == 'SAR':
        # ---- SAR ADAPTATION (3 channels → 2 channels) ----
        #
        # Strategy: average the 3 RGB filter slices into 1 "mean filter",
        # then repeat it for each of our 2 SAR channels (VV and VH).
        #
        # WHY AVERAGE? Both VV and VH polarizations measure backscatter
        # from the same surface geometry, just at different polarizations.
        # Starting them from the same average RGB filter is a reasonable
        # neutral initialization. Mathematically:
        #
        #   W_SAR[:, 0, :, :] = W_SAR[:, 1, :, :] = mean(W_RGB, dim=channel)
        #
        # This conserves the expected activation variance because:
        #   Var[mean(R,G,B)] × 2 ≈ Var[R] (roughly, assuming equal variance)
        mean_weight = original_weights.mean(dim=1, keepdim=True)  # [96, 1, 4, 4]
        new_stem.weight.data = mean_weight.repeat(1, in_channels, 1, 1)  # [96, 2, 4, 4]

    elif modality == 'Optical':
        # ---- OPTICAL ADAPTATION (3 channels → 10 channels) ----
        #
        # Strategy: tile (repeat) the 3-channel weights 4 times to get 12 channels,
        # then slice to 10 channels. Scale down by factor 3/10 to compensate.
        #
        # WHY SCALE BY 3/10? Without scaling, the conv layer would be summing
        # across 10 channels instead of 3, so activations would be ~10/3 times
        # larger than expected by the pre-trained BatchNorm layers downstream.
        # Scaling by 3/10 restores the expected magnitude. Mathematically:
        #
        #   sum(W_opt * X_10ch) ≈ sum(W_rgb * X_3ch)   (in expectation)
        #   because: (3/10) × 10 ≈ 3 channels' worth of contribution
        tiled_weights = original_weights.repeat(1, 4, 1, 1)  # [96, 12, 4, 4]
        tiled_weights = tiled_weights[:, :in_channels, :, :]  # [96, 10, 4, 4]
        new_stem.weight.data = tiled_weights * (3.0 / in_channels)

    # Copy the bias terms unchanged — they don't depend on channel count.
    new_stem.bias.data = original_stem.bias.data

    # Surgically replace ONLY the first layer of the model. All other layers
    # (the 4 ConvNeXt stages, the layer norms) remain exactly as pretrained.
    model.features[0][0] = new_stem
    return model
```

---

### C.2 Generalized Mean (GeM) Spatial Pooling

The learnable generalized mean pooling layer replacing standard Global Average Pooling.

> **For beginners:** After the ConvNeXt backbone, we have a 3D block of numbers (768 channels × 3 × 3 spatial positions). We need to collapse the spatial dimensions (3×3) into a single vector (768 numbers). Standard "Global Average Pooling" just takes the mean — but that blends a bright unique feature (like a small pier sticking into water) with the dull background. GeM uses a tunable power `p` that acts like a volume knob between "average everything" (p=1) and "just keep the maximum" (p=∞). The network learns the best setting during training.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPool(nn.Module):
    """
    Generalized Mean (GeM) Pooling — a learnable spatial pooling layer.

    MATH:
        g_c = ( (1/HW) * Σ_{h,w} max(F_{c,h,w}, ε)^p )^(1/p)

    where:
        F   = feature map tensor [B, C, H, W]
        p   = learnable exponent (initialized to 3.0)
        ε   = small floor value to avoid log(0)
        g_c = pooled scalar for channel c

    INTUITION OF p:
        p → 1 :  arithmetic mean (= Global Average Pooling)
        p = 3 :  emphasizes bright peaks, suppresses low-activation background
        p → ∞ :  approaches max pooling (only the brightest pixel per channel)

    WHY BETTER THAN AVERAGE POOLING FOR RETRIEVAL?
        In retrieval tasks, discriminative local features matter.
        A small building corner or shoreline is crucial to identify a specific
        location. Average pooling would dilute its activation into the background
        noise. GeM amplifies it.
    """

    def __init__(self, p=3.0, eps=1e-6):
        """
        Args:
            p   : initial value for the learnable pooling exponent.
                  3.0 is a good default — empirically chosen by the retrieval community.
            eps : small constant to prevent 0^p = 0 (would cause NaN in backprop for p<1)
        """
        super(GeMPool, self).__init__()
        # nn.Parameter wraps a tensor so PyTorch's optimizer will update it.
        # We start at p=3.0 and let gradient descent find the optimal value.
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps  # numerical stability floor

    def forward(self, x):
        """
        Args:
            x : feature map tensor of shape [B, C, H, W]
                B = batch size, C = channels (768), H = W = 3 (spatial)

        Returns:
            Pooled feature vector of shape [B, C]
        """
        # Step 1: Apply a floor (clamp) to prevent 0^p issues, then raise to power p.
        #         This is the "power" step of the generalized mean.
        #         Shape stays: [B, C, H, W]
        x_clamped = x.clamp(min=self.eps).pow(self.p)

        # Step 2: Take the spatial average over (H, W).
        #         F.avg_pool2d with kernel=(H,W) collapses the spatial dims.
        #         Shape: [B, C, 1, 1]
        x_pooled = F.avg_pool2d(x_clamped, (x.size(-2), x.size(-1)))

        # Step 3: Take the 1/p-th power (the "root" step of the generalized mean).
        #         Then flatten the [1,1] spatial dims into nothing.
        #         Shape: [B, C]
        return x_pooled.pow(1.0 / self.p).flatten(1)
```

---

### C.3 False-Negative Masked Semantic InfoNCE Loss

The asymmetric contrastive loss utilizing the multi-label Jaccard (IoU) overlap to mask semantically identical terrain patches.

> **For beginners:** InfoNCE works like a multiple-choice test — given the SAR image of patch #42, can the network pick its matching optical image from a batch of 32 options? The "false negative problem" is: what if patch #17 and patch #42 both show coniferous forest? Pushing them apart is *wrong*. We detect these "look-alike" pairs by computing their label overlap (Jaccard / IoU) and simply ignore them in the loss calculation.

```python
import torch
import torch.nn.functional as F

def semantic_infonce_loss(online_z, target_z, labels, tau=0.1, tau_iou=0.40):
    """
    Masked InfoNCE contrastive loss with multi-label Jaccard false-negative suppression.

    HOW INFONCE WORKS (without masking):
        Given a batch of B matching pairs (SAR_i, Optical_i), InfoNCE asks:
        "For SAR_i, which of the B optical embeddings is the correct match?"
        It's a B-class classification problem. The correct answer is always
        index i (the diagonal of the similarity matrix).

        Loss = -log[ exp(sim(SAR_i, Opt_i)/τ) / Σ_j exp(sim(SAR_i, Opt_j)/τ) ]

        τ (temperature) controls sharpness: small τ → sharp peaks → hard training signal.

    WHY WE NEED MASKING:
        If SAR_i and Optical_j both show "Forest + River", pushing them apart is
        semantically wrong. The Jaccard mask detects these pairs and excludes them
        from the denominator sum (by setting their logits to -∞ so exp(-∞) = 0).

    Args:
        online_z  : [B, D] unit-normalized embedding vectors from the ONLINE encoder
                    (the one updated by backprop). D = 256.
        target_z  : [B, D] unit-normalized embedding vectors from the TARGET encoder
                    (the EMA-updated slow copy). D = 256.
        labels    : [B, N_classes] binary multi-label matrix.
                    labels[i, c] = 1 if patch i has CORINE class c, else 0.
                    N_classes = 43 for the full CORINE taxonomy.
        tau       : InfoNCE temperature (default 0.1). Lower = sharper distinctions.
        tau_iou   : Jaccard threshold for false-negative suppression (default 0.40).
                    Pairs with IoU > 0.40 are treated as "too similar to push apart".

    Returns:
        scalar loss value (a single number to call .backward() on)
    """
    B = online_z.size(0)  # batch size

    # =========================================================================
    # STEP 1: Compute the full B×B pairwise Jaccard (IoU) matrix
    # =========================================================================
    # For every pair (i, j) in the batch, compute:
    #   IoU(i, j) = |labels_i ∩ labels_j| / |labels_i ∪ labels_j|
    #
    # Using set algebra: |A ∪ B| = |A| + |B| - |A ∩ B|
    #
    # Efficient batch computation via matrix multiplication:
    #   intersection[i,j] = dot(labels_i, labels_j) = number of shared classes
    intersection = torch.mm(labels, labels.t())   # shape: [B, B]

    # area[i] = number of active labels for sample i
    area = labels.sum(dim=1).unsqueeze(1)         # shape: [B, 1]

    # union[i,j] = area[i] + area[j] - intersection[i,j]
    union = area + area.t() - intersection        # shape: [B, B]

    # IoU matrix — 1e-8 avoids division by zero for all-zero label rows
    iou_matrix = intersection / (union + 1e-8)    # shape: [B, B]
    # iou_matrix[i, i] = 1.0 always (a sample is identical to itself)

    # =========================================================================
    # STEP 2: Build the False-Negative Boolean Mask
    # =========================================================================
    # We want to mask out pairs that are:
    #   - Off-diagonal (i ≠ j — we never mask the actual positive pair)
    #   - Semantically very similar (IoU > tau_iou = 0.40)
    #
    # ~torch.eye creates a matrix that is True everywhere EXCEPT the diagonal.
    off_diagonal_mask = ~torch.eye(B, dtype=torch.bool, device=online_z.device)

    # false_negative_mask[i,j] = True means: "sample j is too similar to sample i,
    # so do NOT treat j as a negative (i.e., ignore it in the loss denominator)."
    false_negative_mask = (iou_matrix > tau_iou) & off_diagonal_mask  # shape: [B, B]

    # =========================================================================
    # STEP 3: Compute the scaled dot-product similarity matrix (the "logits")
    # =========================================================================
    # logits[i, j] = (online_z_i · target_z_j) / τ
    # Because both vectors are L2-normalized, the dot product equals cosine similarity.
    # Dividing by τ=0.1 (small temperature) makes the distribution sharper.
    logits = torch.mm(online_z, target_z.t()) / tau   # shape: [B, B]

    # =========================================================================
    # STEP 4: Apply the False-Negative Mask
    # =========================================================================
    # Set masked positions to a large negative number.
    # exp(-1e9) ≈ 0, so these pairs contribute nothing to the softmax denominator.
    # This is equivalent to removing them from the "competition" in the multiple-choice test.
    logits.masked_fill_(false_negative_mask, -1e9)

    # =========================================================================
    # STEP 5: Compute Contrastive Cross-Entropy Loss
    # =========================================================================
    # The correct match for sample i is always at column index i (the diagonal).
    # target_indices = [0, 1, 2, ..., B-1]
    target_indices = torch.arange(B, device=online_z.device)

    # F.cross_entropy computes: -log(softmax(logits)[i, target_i])
    # = -logits[i,i] + log(Σ_j exp(logits[i,j]))   (the standard InfoNCE formula)
    loss = F.cross_entropy(logits, target_indices)

    return loss
```
