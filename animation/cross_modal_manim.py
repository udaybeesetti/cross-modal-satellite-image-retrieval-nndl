"""
cross_modal_manim.py — Training Pipeline Animation
====================================================

PURPOSE:
    Creates an animated 1080p60 explainer video of the full Cross-Modal
    Satellite Image Retrieval training pipeline using the Manim Community
    library (the same library used by 3Blue1Brown for math animations).

WHAT IT ANIMATES (4 stages):
    Stage 1: Sentinel-1 SAR + Sentinel-2 optical tensors flowing through
             synchronized D₄ dihedral augmentation (flips + rotations).
    Stage 2: Sensor-specific ConvNeXt-Tiny stem channel adaptation
             (3 RGB → 2 SAR channels, or 3 RGB → 10 optical channels).
    Stage 3: Asymmetric Online/Target encoder pair with GeM pooling,
             hypersphere projection, and EMA weight synchronization.
    Stage 4: CORINE multi-label Jaccard gate + masked InfoNCE composite loss.

HOW TO RENDER (from project root):
    source .venv/bin/activate
    manim -pqh animation/cross_modal_manim.py CrossModalArchitecture

    Flags:
        -p  : auto-open the video after rendering
        -qh : high quality (1920×1080 at 60fps)
        -ql : low quality (480p at 15fps, faster for previewing)

OUTPUT:
    media/videos/cross_modal_manim/1080p60/CrossModalArchitecture.mp4

DEPENDENCIES:
    pip install manim
    sudo dnf install cairo-devel pango-devel ffmpeg python3-devel gcc pkg-config
    sudo dnf install texlive-scheme-basic texlive-amsmath texlive-dvisvgm
"""

from manim import *

class CrossModalArchitecture(Scene):
    def construct(self):
        """
        Main animation method. Manim calls construct() once to build and
        render the entire scene. All animation logic lives here.

        The scene is divided into 4 sequential stages, each revealing more
        of the network architecture from left (input) to right (loss).
        """

        # =====================================================================
        # COLOUR PALETTE — 3Blue1Brown inspired dark theme
        # =====================================================================
        # Using a consistent palette makes it easy to follow which parts of
        # the diagram belong to which modality/component.
        self.camera.background_color = "#0e1117"  # near-black background

        C_SAR    = "#58C4DD"   # Cyan  — Sentinel-1 SAR (radar) modality
        C_OPT    = "#F0AC5F"   # Gold  — Sentinel-2 Optical modality
        C_ONLINE = "#83C167"   # Green — Online encoder (trained by backprop)
        C_TARGET = "#5CD0B3"   # Teal  — Target encoder (updated by EMA only)
        C_GEM    = "#9A72AC"   # Purple — CORINE Jaccard gate / GeM pooling
        C_LOSS   = "#FC6255"   # Red   — Loss function node
        C_PANEL  = "#161b22"   # Very dark — card/block fill colour
        C_BORDER = "#30363d"   # Subtle grey — border colour for UI frames

        # =====================================================================
        # HELPER FUNCTIONS — reusable visual building blocks
        # =====================================================================

        # --- Helper: Layered Tensor Stack (3b1b style) ---
        # Creates a visual stack of overlapping squares to represent a
        # multi-channel tensor (like a SAR image with 2 channels, or an
        # optical image with 10 channels).
        #
        # Parameters:
        #   num_layers : number of squares to stack (visual only, not actual channels)
        #   side_len   : size of each square in scene units
        #   color      : edge + tint colour (C_SAR = cyan, C_OPT = gold, etc.)
        #   title_tex  : bold label shown above the stack (LaTeX string)
        #   shape_tex  : shape annotation shown below (e.g., R^{B×2×120×120})
        def make_tensor_stack(num_layers, side_len, color, title_tex, shape_tex):
            layers = VGroup()
            dx, dy = 0.055, 0.045
            for i in range(num_layers):
                sq = Square(
                    side_length=side_len,
                    stroke_color=color,
                    stroke_width=1.8,
                    fill_color=color,
                    fill_opacity=0.18,
                ).shift(RIGHT * i * dx + UP * i * dy)
                layers.add(sq)
            layers.move_to(ORIGIN)
            t_lbl = Tex(title_tex, color=color).scale(0.42).next_to(layers, UP, buff=0.12)
            s_lbl = MathTex(shape_tex, color=GREY_B).scale(0.38).next_to(layers, DOWN, buff=0.12)
            return VGroup(layers, t_lbl, s_lbl)

        # --- Helper: Clean 3b1b Module Block ---
        # Creates a rounded rectangle card representing a network module
        # (e.g., the ConvNeXt stem, the online encoder, the loss node).
        # The 'dashed' option draws a dashed border to signal "no gradient"
        # (used for the target/EMA encoder — it does not receive backprop).
        #
        # Parameters:
        #   w, h       : width and height of the card
        #   color      : border and title accent colour
        #   title_tex  : bold module name (LaTeX)
        #   math_tex   : formula or shape annotation inside the card
        #   dashed     : if True, draws dashed border (= stop-gradient / no backprop)
        def make_block(w, h, color, title_tex, math_tex, dashed=False):
            rect = RoundedRectangle(
                corner_radius=0.08,
                width=w,
                height=h,
                fill_color=C_PANEL,
                fill_opacity=0.95,
                stroke_color=color,
                stroke_width=2.2,
            )
            if dashed:
                border = DashedVMobject(
                    RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2.2),
                    num_dashes=32,
                )
                bg = RoundedRectangle(
                    corner_radius=0.08, width=w, height=h,
                    fill_color=C_PANEL, fill_opacity=0.65, stroke_width=0
                )
                rect = VGroup(bg, border)

            t1 = Tex(title_tex, color=color).scale(0.42)
            t2 = MathTex(math_tex, color=WHITE).scale(0.38)
            content = VGroup(t1, t2).arrange(DOWN, buff=0.1).move_to(rect.get_center())
            return VGroup(rect, content)

        # --- Helper: Bottom 3b1b Equation Bar (Guaranteed No Overlap) ---
        # This is the HUD ("heads-up display") bar at the bottom of each stage.
        # It shows the stage title and the key mathematical equations, acting
        # like subtitles for the animation. The bar is pre-created once and
        # the content is replaced at each stage transition.
        hud_frame = RoundedRectangle(
            corner_radius=0.1,
            width=13.4,
            height=1.95,
            fill_color=C_PANEL,
            fill_opacity=0.95,
            stroke_color=C_BORDER,
            stroke_width=2,
        ).to_edge(DOWN, buff=0.18)

        # make_hud: fills the bottom bar with stage-specific text.
        # The bar itself (hud_frame) stays; only the text changes between stages.
        def make_hud(stage_title, eq_top_tex, eq_bot_tex, accent_color):
            title = Tex(stage_title, color=accent_color).scale(0.46)
            eq1 = MathTex(eq_top_tex, color=WHITE).scale(0.48)
            eq2 = MathTex(eq_bot_tex, color=GREY_B).scale(0.44)

            grp = VGroup(title, eq1, eq2).arrange(DOWN, buff=0.14)
            if grp.width > 12.8:
                grp.scale_to_fit_width(12.8)
            grp.move_to(hud_frame.get_center())
            return grp

        # --- HEADER ---
        # The title bar at the top of the scene, displayed throughout the animation.
        header = Tex(
            r"\textbf{CrossModalSAROpticalNet} \,---\, Asymmetric Dual-Tower Metric Learning",
            color=WHITE,
        ).scale(0.58).to_edge(UP, buff=0.18)
        header_line = Line(LEFT * 6.7, RIGHT * 6.7, color=C_BORDER, stroke_width=1.5).next_to(header, DOWN, buff=0.1)
        self.add(header, header_line, hud_frame)

        # =========================================================================
        # STAGE 1: MULTI-SENSOR TENSORS & SYNCHRONIZED D4 AUGMENTATION
        # =========================================================================
        # What this stage shows:
        #   - Two input tensors arrive: SAR (2 channels) and Optical (10 channels)
        #   - Both pass through a *synchronized* D4 augmentation block
        #   - The same random transformation T (flip/rotate) is applied to BOTH
        #     modalities to keep pixel alignment (e.g., a river must point the
        #     same direction in both the radar and camera image)
        #   - Augmented views → Online encoders; Clean views → Target encoders
        #
        # The HUD (bottom bar) shows the normalization formulas:
        #   SAR:     (clip(σ⁰_dB, -35, 0) + 35) / 35    → maps dB to [0,1]
        #   Optical: clip(DN / 10000, 0, 1)              → maps reflectance to [0,1]
        sar_stack = make_tensor_stack(
            2, 0.75, C_SAR, r"\textbf{Sentinel-1 SAR}", r"X_{\text{S1}} \in \mathbb{R}^{B \times 2 \times 120 \times 120}"
        ).move_to(LEFT * 5.5 + UP * 1.95)

        opt_stack = make_tensor_stack(
            6, 0.65, C_OPT, r"\textbf{Sentinel-2 MSI}", r"X_{\text{S2}} \in \mathbb{R}^{B \times 10 \times 120 \times 120}"
        ).move_to(LEFT * 5.5 + DOWN * 0.15)

        d4_box = make_block(
            1.75, 2.7, WHITE,
            r"\textbf{Sync } $D_4$",
            r"(T(X_{\text{S1}}),\, T(X_{\text{S2}}))"
        ).move_to(LEFT * 3.35 + UP * 0.9)

        arr_s1_d4 = Arrow(sar_stack[0].get_right(), d4_box.get_left() + UP * 0.85, buff=0.12, color=C_SAR, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)
        arr_s2_d4 = Arrow(opt_stack[0].get_right(), d4_box.get_left() + DOWN * 0.85, buff=0.12, color=C_OPT, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)

        hud_1 = make_hud(
            r"\textbf{Stage 1: Radiometric Normalization \& Co-Registered Dihedral ($D_4$) Routing}",
            r"\tilde{X}_{\text{S1}} = \frac{\text{clip}(\sigma^0_{\text{dB}}, -35, 0) + 35}{35}, \qquad \tilde{X}_{\text{S2}} = \text{clip}\!\left(\frac{\text{DN}}{10000}, 0, 1\right), \qquad T \in D_4",
            r"\text{Online Query Branch } (\theta_q) \leftarrow \text{Augmented } T(X) \quad\big|\quad \text{Target Key Branch } (\theta_k) \leftarrow \text{Clean Unaugmented } X",
            C_SAR,
        )

        # Animate Stage 1: fade in both input tensors, arrows, and the D4 block,
        # then rotate both stacks by 90° simultaneously to show synchronization.
        self.play(LaggedStart(FadeIn(sar_stack, shift=RIGHT * 0.2), FadeIn(opt_stack, shift=RIGHT * 0.2), lag_ratio=0.2), FadeIn(hud_1), run_time=1.2)
        self.play(GrowArrow(arr_s1_d4), GrowArrow(arr_s2_d4), FadeIn(d4_box), run_time=1.0)
        # Synchronized 90-degree rotation of the tensor slices inside the stacks
        # This visually demonstrates that the SAME augmentation is applied to both.
        self.play(
            Rotate(sar_stack[0], angle=PI / 2),
            Rotate(opt_stack[0], angle=PI / 2),
            run_time=1.1,
        )
        self.wait(1.2)

        # =========================================================================
        # STAGE 2: ENERGY-CONSERVING CONVNEXT-TINY STEMS & BACKBONE
        # =========================================================================
        # What this stage shows:
        #   - Each modality enters its own "stem" (first conv layer) that adapts
        #     the channel count from 2 or 10 → 96 feature maps
        #   - The SAR stem uses the averaged RGB filter repeated twice
        #   - The Optical stem tiles and scales the RGB filters across 10 channels
        #
        # After the stem: both branches share the same ConvNeXt-Tiny architecture
        # and progress through 4 stages of progressively deeper features:
        #   (B, 96, 30, 30) → (B, 192, 15, 15) → (B, 384, 7, 7) → (B, 768, 3, 3)
        stem_sar = make_block(
            2.15, 0.95, C_SAR,
            r"\textbf{SAR Stem } $4\times 4$",
            r"2 \to 96 \;\; (\text{stride } 4)"
        ).move_to(LEFT * 1.05 + UP * 2.15)

        stem_opt = make_block(
            2.15, 0.95, C_OPT,
            r"\textbf{Opt Stem } $4\times 4$",
            r"10 \to 96 \;\; (\text{stride } 4)"
        ).move_to(LEFT * 1.05 + UP * 0.95)

        arr_d4_st1 = Arrow(d4_box.get_right() + UP * 0.95, stem_sar.get_left(), buff=0.1, color=C_SAR, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)
        arr_d4_st2 = Arrow(d4_box.get_right() + UP * 0.05, stem_opt.get_left(), buff=0.1, color=C_OPT, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)

        hud_2 = make_hud(
            r"\textbf{Stage 2: Energy-Conserving ConvNeXt-Tiny Stem Weight Initialization}",
            r"W_{\text{S1}} = \text{repeat}_2\!\left(\frac{1}{3}\sum_{c=1}^{3} W_{\text{RGB}}[:,c,:,:]\right) \in \mathbb{R}^{96\times 2\times 4\times 4}, \qquad W_{\text{S2}} = \frac{3}{10}\,\text{repeat}_4(W_{\text{RGB}})[:,:10,:,:] \in \mathbb{R}^{96\times 10\times 4\times 4}",
            r"\text{Spatial Hierarchy: } (B, C, 120, 120) \xrightarrow{\text{Stem } 4\times 4} (B, 96, 30, 30) \xrightarrow{\text{Stages 1--4}} F \in \mathbb{R}^{B \times 768 \times 3 \times 3}",
            C_OPT,
        )

        self.play(
            ReplacementTransform(hud_1, hud_2),
            GrowArrow(arr_d4_st1), GrowArrow(arr_d4_st2),
            FadeIn(stem_sar, shift=RIGHT * 0.15),
            FadeIn(stem_opt, shift=RIGHT * 0.15),
            run_time=1.2,
        )
        self.wait(1.8)

        # =========================================================================
        # STAGE 3: ASYMMETRIC ONLINE / TARGET ENCODERS, GEM POOLING & EMA
        # =========================================================================
        # What this stage shows:
        #   - Online encoders (green): receive augmented views, trained by backprop
        #   - Target encoders (teal, dashed border): receive clean views, NO backprop
        #     The dashed border visually encodes "stop-gradient"
        #   - EMA arrow (red dashed): θ_k ← 0.996·θ_k + 0.004·θ_q
        #     The target slowly "follows" the online encoder like a moving average
        #   - Both encoders end with:
        #     GeM pooling (768×3×3 → 768-d vector) → MLP → L2-normalize → S^255
        #
        # KEY: We also sync the BatchNorm running statistics via EMA,
        #      not just the conv weights. Without this, the target encoder's
        #      internal normalization would drift relative to the online encoder.
        online_enc = make_block(
            2.5, 1.35, C_ONLINE,
            r"\textbf{Online Encoder } $(\theta_q)$",
            r"\text{GeM}(p\!=\!3.0) \to \text{MLP} \to \mathbb{S}^{255}"
        ).move_to(RIGHT * 1.75 + UP * 1.85)

        target_enc = make_block(
            2.5, 1.35, C_TARGET,
            r"\textbf{Target Encoder } $(\theta_k)$",
            r"\text{Stop-Grad} \;\big|\; \text{Clean Views}",
            dashed=True,
        ).move_to(RIGHT * 1.75 + DOWN * 0.25)

        arr_st1_on = Arrow(stem_sar.get_right(), online_enc.get_left() + UP * 0.28, buff=0.1, color=C_ONLINE, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)
        arr_st2_on = Arrow(stem_opt.get_right(), online_enc.get_left() + DOWN * 0.28, buff=0.1, color=C_ONLINE, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)

        # Clean unaugmented route from D4 directly to Target Encoder
        arr_clean_tar = Arrow(
            d4_box.get_right() + DOWN * 1.0,
            target_enc.get_left(),
            buff=0.1, color=C_TARGET, stroke_width=2.5, max_tip_length_to_length_ratio=0.12
        )
        clean_lbl = Tex(r"Clean $X_{\text{S1}}, X_{\text{S2}}$ (No Aug)", color=C_TARGET).scale(0.32).next_to(arr_clean_tar, DOWN, buff=0.05)

        ema_arrow = DashedLine(
            online_enc.get_bottom(), target_enc.get_top(),
            color=C_LOSS, stroke_width=2.8
        ).add_tip(tip_length=0.15)
        ema_tex = MathTex(r"\theta_k \leftarrow m\theta_k + (1\!-\!m)\theta_q", color=C_LOSS).scale(0.34).next_to(ema_arrow, RIGHT, buff=0.08)

        hud_3 = make_hud(
            r"\textbf{Stage 3: Learnable GeM Pooling ($p=3.0$), Hypersphere Projection \& EMA Sync}",
            r"g_c = \left(\frac{1}{H'W'}\sum_{u,v} \max(F_{c,u,v}, 10^{-6})^p\right)^{\!1/p} \in \mathbb{R}^{768}, \qquad z = \frac{\phi(g)}{\|\phi(g)\|_2} \in \mathbb{S}^{255}",
            r"\text{EMA Update } (m=0.996): \quad \theta_k \leftarrow 0.996\,\theta_k + 0.004\,\theta_q \quad\text{with synchronized BatchNorm1d } (\mu_{\text{tar}}, \sigma^2_{\text{tar}})",
            C_ONLINE,
        )

        self.play(
            ReplacementTransform(hud_2, hud_3),
            GrowArrow(arr_st1_on), GrowArrow(arr_st2_on),
            GrowArrow(arr_clean_tar), FadeIn(clean_lbl),
            FadeIn(online_enc), FadeIn(target_enc),
            Create(ema_arrow), FadeIn(ema_tex),
            run_time=1.4,
        )
        self.wait(2.0)

        # =========================================================================
        # STAGE 4: HYPERSPHERE S^255 & JACCARD-MASKED INFONCE MATRIX
        # =========================================================================
        # What this stage shows:
        #   - The CORINE Jaccard Gate (purple): computes pairwise label overlap
        #     IoU(i,j) = |Y_i ∩ Y_j| / |Y_i ∪ Y_j|  for every pair in the batch
        #     If IoU > 0.40: mask that pair (set logit → -∞) so it doesn't get
        #     pushed apart (false negative suppression)
        #   - The Masked InfoNCE Loss (red): computes the contrastive objective
        #     using the online embeddings as queries against the target embeddings
        #   - The composite loss = 0.5 × cross-modal + 0.25 × self-distillation
        #
        # The HUD shows the full InfoNCE formula with the Jaccard masking condition
        # and the final weighted composite loss equation.
        iou_gate = make_block(
            2.35, 1.15, C_GEM,
            r"\textbf{CORINE Jaccard Gate}",
            r"\text{IoU}_{ij} = \frac{|Y_i \cap Y_j|}{|Y_i \cup Y_j|} > 0.40"
        ).move_to(RIGHT * 5.15 + UP * 1.95)

        loss_node = make_block(
            2.35, 1.35, C_LOSS,
            r"\textbf{Masked InfoNCE}",
            r"l_{ij} = -\infty \;\text{if } M_{ij}=1"
        ).move_to(RIGHT * 5.15 + DOWN * 0.25)

        arr_on_loss = Arrow(online_enc.get_right(), loss_node.get_left() + UP * 0.28, buff=0.1, color=C_ONLINE, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)
        arr_tar_loss = Arrow(target_enc.get_right(), loss_node.get_left() + DOWN * 0.28, buff=0.1, color=C_TARGET, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)
        arr_gate_loss = Arrow(iou_gate.get_bottom(), loss_node.get_top(), buff=0.08, color=C_GEM, stroke_width=2.5, max_tip_length_to_length_ratio=0.22)

        hud_4 = make_hud(
            r"\textbf{Stage 4: Multi-Label Jaccard False-Negative Masking \& Composite InfoNCE}",
            r"\mathcal{L}(U,V) = -\frac{1}{2B}\sum_{i=1}^{B} \log \frac{\exp(u_i^\top v_i / \tau)}{\exp(u_i^\top v_i / \tau) + \sum_{j \neq i,\, \text{IoU}_{ij} \le 0.40} \exp(u_i^\top v_j / \tau)} \qquad (\tau = 0.10)",
            r"\mathcal{L}_{\text{total}} = 0.50\,\big[\mathcal{L}(Z_{\text{S1}}, T_{\text{S2}}) + \mathcal{L}(Z_{\text{S2}}, T_{\text{S1}})\big] + 0.25\,\big[\mathcal{L}(Z_{\text{S1}}, T_{\text{S1}}) + \mathcal{L}(Z_{\text{S2}}, T_{\text{S2}})\big]",
            C_LOSS,
        )

        self.play(\
            ReplacementTransform(hud_3, hud_4),
            FadeIn(iou_gate, shift=DOWN * 0.15),
            FadeIn(loss_node, shift=UP * 0.15),
            GrowArrow(arr_on_loss),
            GrowArrow(arr_tar_loss),
            GrowArrow(arr_gate_loss),
            run_time=1.4,
        )
        # Indicate = briefly pulse/highlight the loss node to draw attention
        self.play(Indicate(loss_node[0], color=C_LOSS, scale_factor=1.04), run_time=1.0)
        # Final pause — holds the complete diagram on screen for the viewer
        self.wait(3.0)