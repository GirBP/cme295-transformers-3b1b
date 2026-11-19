"""
Stanford CME 295: Transformers & Large Language Models
Lecture 1 — Transformers (Afshine & Shervine Amidi)
YouTube: https://www.youtube.com/watch?v=114i2Kz-LZA

Manim Community Edition script styled after 3Blue1Brown (Grant Sanderson).
Run any scene with:
    manim -pqh cme295_lecture1_manim.py Scene1_TokenizationBPE
    manim -pqh cme295_lecture1_manim.py Scene2_Word2VecEmbeddings
    manim -pqh cme295_lecture1_manim.py Scene3_RNNvsAttention
    manim -pqh cme295_lecture1_manim.py Scene4_SelfAttentionQKV
    manim -pqh cme295_lecture1_manim.py Scene5_TransformerArchitecture
Or render all scenes:
    manim -pqh -a cme295_lecture1_manim.py
"""

from manim import *
import numpy as np

# 3Blue1Brown Signature Theme Configuration
BG_3B1B = "#0c1017"
BLUE_3B1B = "#58C4DD"    # Queries / Primary Vectors
YELLOW_3B1B = "#FFFF00"  # Keys / Attention Weights / Highlights
GREEN_3B1B = "#83C167"   # Values / Output Context Vectors
RED_3B1B = "#FC6255"     # Bottlenecks / Causal Mask (-inf)
TEAL_3B1B = "#5CD0B3"    # Encoder / Positional Encoding
PURPLE_3B1B = "#9A72AC"  # Decoder / Softmax Probabilities


class Scene1_TokenizationBPE(Scene):
    """
    00:16:56 — Tokenization: Word-level vs Character-level vs Subword (BPE).
    Visualizes the trade-off between Vocabulary Size |V| and Sequence Length L,
    followed by an animated Byte-Pair Encoding (BPE) merge tree.
    """
    def construct(self):
        self.camera.background_color = BG_3B1B

        title = Text("1. Токенізація та Byte-Pair Encoding (BPE)", font_size=34, color=WHITE)
        subtitle = MathTex(
            r"\text{Компроміс: Розмір словника } |V| \iff \text{Довжина послідовності } L",
            font_size=26,
            color=BLUE_3B1B,
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.4)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2))

        # 1. Word-level row
        word_lbl = Text("Word-level (|V| > 500k, OOV проблема):", font_size=22, color=RED_3B1B)
        word_boxes = VGroup(
            self._make_token_box("unaffable", RED_3B1B),
            MathTex(r"\rightarrow", color=GREY_B),
            self._make_token_box("[UNK]", RED_3B1B),
        ).arrange(RIGHT, buff=0.25)
        row1 = VGroup(word_lbl, word_boxes).arrange(RIGHT, buff=0.5).shift(UP * 1.1)

        # 2. Character-level row
        char_lbl = Text("Character-level (|V| ≈ 256, велике L):", font_size=22, color=YELLOW_3B1B)
        char_boxes = VGroup(*[
            self._make_token_box(ch, YELLOW_3B1B, width=0.55)
            for ch in list("unaffable")
        ]).arrange(RIGHT, buff=0.1)
        row2 = VGroup(char_lbl, char_boxes).arrange(RIGHT, buff=0.5).shift( DOWN * 0.1)

        # 3. Subword (BPE) row
        bpe_lbl = Text("Subword / BPE (|V| ≈ 50k–100k):", font_size=22, color=BLUE_3B1B)
        bpe_boxes = VGroup(
            self._make_token_box("un", BLUE_3B1B),
            self._make_token_box("##aff", TEAL_3B1B),
            self._make_token_box("##able", GREEN_3B1B),
        ).arrange(RIGHT, buff=0.2)
        row3 = VGroup(bpe_lbl, bpe_boxes).arrange(RIGHT, buff=0.5).shift(DOWN * 1.3)

        for row in [row1, row2, row3]:
            row.move_to([0, row.get_center()[1], 0])

        self.play(FadeIn(row1, shift=RIGHT * 0.3))
        self.wait(0.6)
        self.play(FadeIn(row2, shift=RIGHT * 0.3))
        self.wait(0.6)
        self.play(FadeIn(row3, shift=RIGHT * 0.3))
        self.wait(1.5)

    def _make_token_box(self, text_str, color, width=None):
        txt = Text(text_str, font="Monospace", font_size=22, color=WHITE)
        box_w = width if width else max(1.1, txt.width + 0.45)
        rect = RoundedRectangle(
            corner_radius=0.12,
            width=box_w,
            height=0.65,
            stroke_color=color,
            stroke_width=2.5,
            fill_color=color,
            fill_opacity=0.15,
        )
        txt.move_to(rect.get_center())
        return VGroup(rect, txt)


class Scene2_Word2VecEmbeddings(Scene):
    """
    00:37:27 — Word2Vec (CBOW & Skip-gram) and Dense Semantic Vector Space R^d.
    Visualizes the projection from sparse one-hot vectors e_i in {0,1}^{|V|}
    to dense embeddings v_i = E e_i in R^d, showing the famous parallelogram:
        v(king) - v(man) + v(woman) ≈ v(queen).
    """
    def construct(self):
        self.camera.background_color = BG_3B1B

        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": BLUE_3B1B,
                "stroke_width": 1,
                "stroke_opacity": 0.18,
            },
            axis_config={"stroke_opacity": 0.4},
        )
        self.add(plane)

        title = Text("2. Word2Vec: Семантичний простір ембедингів ℝᵈ", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.35).add_background_rectangle(color=BG_3B1B, opacity=0.85)
        self.play(FadeIn(title))

        # Coordinates in 2D slice of R^d
        p_man = np.array([-2.8, -1.2, 0])
        p_king = np.array([-1.8, 1.5, 0])
        p_woman = np.array([1.2, -1.0, 0])
        p_queen = p_woman + (p_king - p_man)  # Exact parallelogram relation

        dots = VGroup(
            Dot(p_man, color=BLUE_3B1B, radius=0.1),
            Dot(p_king, color=YELLOW_3B1B, radius=0.1),
            Dot(p_woman, color=GREEN_3B1B, radius=0.1),
            Dot(p_queen, color=PURPLE_3B1B, radius=0.1),
        )
        labels = VGroup(
            MathTex(r"\vec{v}_{\text{man}}", color=BLUE_3B1B).next_to(p_man, DOWN),
            MathTex(r"\vec{v}_{\text{king}}", color=YELLOW_3B1B).next_to(p_king, UP),
            MathTex(r"\vec{v}_{\text{woman}}", color=GREEN_3B1B).next_to(p_woman, DOWN),
            MathTex(r"\vec{v}_{\text{queen}}", color=PURPLE_3B1B).next_to(p_queen, UP),
        )

        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.2))
        self.play(LaggedStart(*[Write(l) for l in labels], lag_ratio=0.2))

        vec_royalty_1 = Arrow(p_man, p_king, buff=0.12, color=YELLOW_3B1B, stroke_width=4)
        vec_royalty_2 = Arrow(p_woman, p_queen, buff=0.12, color=YELLOW_3B1B, stroke_width=4)
        vec_gender_1 = DashedLine(p_man, p_woman, color=TEAL_3B1B)
        vec_gender_2 = DashedLine(p_king, p_queen, color=TEAL_3B1B)

        self.play(GrowArrow(vec_royalty_1), Create(vec_gender_1))
        self.play(TransformFromCopy(vec_royalty_1, vec_royalty_2), Create(vec_gender_2))

        formula = MathTex(
            r"\vec{v}(\text{king}) - \vec{v}(\text{man}) + \vec{v}(\text{woman}) \approx \vec{v}(\text{queen})",
            font_size=32,
            color=YELLOW_3B1B,
        ).to_edge(DOWN, buff=0.45).add_background_rectangle(color=BG_3B1B, opacity=0.9)

        self.play(Write(formula))
        self.wait(2)


class Scene3_RNNvsAttention(Scene):
    """
    00:48:26 — RNN / LSTM Sequential Bottleneck vs Direct Parallel Self-Attention.
    """
    def construct(self):
        self.camera.background_color = BG_3B1B

        title = Text("3. Послідовні RNN/LSTM проти прямого зв'язку Attention", font_size=32)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        words = ["The", "trophy", "didn't", "fit", "because", "it"]
        xs = np.linspace(-5.0, 5.0, len(words))

        # Top: RNN chain
        rnn_nodes = VGroup()
        rnn_arrows = VGroup()
        for i, (w, x) in enumerate(zip(words, xs)):
            opacity = max(0.18, 0.85 - i * 0.13)
            circle = Circle(radius=0.42, color=RED_3B1B, fill_opacity=opacity).move_to([x, 1.2, 0])
            lbl = MathTex(f"h_{i+1}", font_size=26).move_to(circle.get_center())
            w_txt = Text(w, font="Monospace", font_size=20, color=GREY_B).next_to(circle, DOWN, buff=0.2)
            rnn_nodes.add(VGroup(circle, lbl, w_txt))
            if i > 0:
                arr = Arrow([xs[i - 1] + 0.45, 1.2, 0], [x - 0.45, 1.2, 0], buff=0, color=RED_3B1B)
                rnn_arrows.add(arr)

        rnn_caption = Text("RNN: Послідовно O(N) кроків + затухання градієнта", font_size=22, color=RED_3B1B)
        rnn_caption.move_to([0, 2.25, 0])

        self.play(FadeIn(rnn_caption), LaggedStart(*[FadeIn(n) for n in rnn_nodes], lag_ratio=0.15))
        self.play(LaggedStart(*[GrowArrow(a) for a in rnn_arrows], lag_ratio=0.2))

        # Bottom: Self-Attention O(1) direct arc from "it" to "trophy"
        att_caption = Text("Self-Attention: Прямий зв'язок O(1) на будь-якій відстані", font_size=22, color=BLUE_3B1B)
        att_caption.move_to([0, -0.35, 0])

        att_nodes = VGroup()
        for i, (w, x) in enumerate(zip(words, xs)):
            c = YELLOW_3B1B if i == 1 else (BLUE_3B1B if i == 5 else GREY_C)
            box = RoundedRectangle(corner_radius=0.1, width=1.4, height=0.6, color=c, fill_opacity=0.2)
            box.move_to([x, -2.1, 0])
            t_lbl = Text(w, font="Monospace", font_size=20, color=WHITE).move_to(box.get_center())
            att_nodes.add(VGroup(box, t_lbl))

        arc_main = CurvedArrow(
            start_point=[xs[5], -1.8, 0],
            end_point=[xs[1], -1.8, 0],
            angle=TAU / 5,
            color=YELLOW_3B1B,
            stroke_width=6,
        )
        weight_lbl = MathTex(r"\alpha_{\text{it} \to \text{trophy}} = 0.64", font_size=26, color=YELLOW_3B1B)
        weight_lbl.next_to(arc_main, UP, buff=0.1)

        self.play(FadeIn(att_caption), FadeIn(att_nodes))
        self.play(Create(arc_main), Write(weight_lbl))
        self.wait(2)


class Scene4_SelfAttentionQKV(Scene):
    """
    00:59:54 — Scaled Dot-Product Self-Attention (Query, Key, Value):
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def construct(self):
        self.camera.background_color = BG_3B1B

        title = Text("4. Scaled Dot-Product Self-Attention (Q, K, V)", font_size=32)
        title.to_edge(UP, buff=0.35)

        eq = MathTex(
            r"\text{Attention}(", r"Q", r", ", r"K", r", ", r"V", r") = \text{softmax}\!\left(\frac{",
            r"Q", r"K^T", r"}{\sqrt{d_k}}\right)", r"V",
            font_size=38,
        )
        eq.set_color_by_tex("Q", BLUE_3B1B)
        eq.set_color_by_tex("K", YELLOW_3B1B)
        eq.set_color_by_tex("V", GREEN_3B1B)
        eq.next_to(title, DOWN, buff=0.35)

        self.play(Write(title), Write(eq))

        # Matrix blocks visualization below
        q_mat = self._matrix_block("Q", r"L \times d_k", BLUE_3B1B).move_to([-4.2, -1.2, 0])
        kt_mat = self._matrix_block("K^T", r"d_k \times L", YELLOW_3B1B).move_to([-1.6, -1.2, 0])
        scores_mat = self._matrix_block(r"\alpha = \text{softmax}", r"L \times L", PURPLE_3B1B).move_to([1.2, -1.2, 0])
        v_mat = self._matrix_block("V", r"L \times d_v", GREEN_3B1B).move_to([4.0, -1.2, 0])

        times1 = MathTex(r"\times", font_size=36).move_to([-2.9, -1.2, 0])
        arrow1 = MathTex(r"\xrightarrow{\div \sqrt{d_k}}", font_size=30, color=TEAL_3B1B).move_to([-0.2, -1.2, 0])
        times2 = MathTex(r"\times", font_size=36).move_to([2.6, -1.2, 0])

        self.play(
            FadeIn(q_mat, shift=UP * 0.2),
            FadeIn(times1),
            FadeIn(kt_mat, shift=UP * 0.2),
        )
        self.play(Write(arrow1), FadeIn(scores_mat, scale=0.9))
        self.play(FadeIn(times2), FadeIn(v_mat, shift=UP * 0.2))

        note = Text(
            "Масштабування на 1/√dₖ запобігає насиченню Softmax (де градієнти → 0)",
            font_size=22,
            color=TEAL_3B1B,
        ).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(note))
        self.wait(2)

    def _matrix_block(self, label_tex, dim_tex, color):
        rect = RoundedRectangle(
            corner_radius=0.12,
            width=1.9,
            height=1.7,
            stroke_color=color,
            stroke_width=3,
            fill_color=color,
            fill_opacity=0.18,
        )
        lbl = MathTex(label_tex, font_size=30, color=WHITE).move_to(rect.get_center() + UP * 0.2)
        dim = MathTex(dim_tex, font_size=22, color=color).move_to(rect.get_center() + DOWN * 0.4)
        return VGroup(rect, lbl, dim)


class Scene5_TransformerArchitecture(Scene):
    """
    01:09:35 — Full Transformer Encoder-Decoder & Look-Ahead Causal Mask (-inf).
    """
    def construct(self):
        self.camera.background_color = BG_3B1B

        title = Text("5. Архітектура Transformer: Encoder, Decoder та Causal Mask", font_size=30)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        # Encoder stack
        enc_frame = RoundedRectangle(
            corner_radius=0.15, width=3.6, height=4.6, color=TEAL_3B1B, fill_opacity=0.08
        ).move_to([-3.8, -0.4, 0])
        enc_title = Text("ENCODER (N×)", font_size=22, color=TEAL_3B1B).next_to(enc_frame.get_top(), DOWN, buff=0.2)
        enc_b1 = self._layer_pill("Multi-Head Self-Attention", TEAL_3B1B).move_to([-3.8, -1.5, 0])
        enc_b2 = self._layer_pill("Add & LayerNorm", BLUE_3B1B).move_to([-3.8, -0.6, 0])
        enc_b3 = self._layer_pill("Feed-Forward Network", GREEN_3B1B).move_to([-3.8, 0.3, 0])

        # Decoder stack
        dec_frame = RoundedRectangle(
            corner_radius=0.15, width=3.8, height=4.6, color=PURPLE_3B1B, fill_opacity=0.08
        ).move_to([0.4, -0.4, 0])
        dec_title = Text("DECODER (N×)", font_size=22, color=PURPLE_3B1B).next_to(dec_frame.get_top(), DOWN, buff=0.2)
        dec_b1 = self._layer_pill("Masked Self-Attention (-∞)", RED_3B1B).move_to([0.4, -1.7, 0])
        dec_b2 = self._layer_pill("Cross-Attention (Q_dec, K_enc, V_enc)", YELLOW_3B1B).move_to([0.4, -0.7, 0])
        dec_b3 = self._layer_pill("FFN + Linear + Softmax", GREEN_3B1B).move_to([0.4, 0.3, 0])

        kv_arrow = CurvedArrow(
            enc_b3.get_right(),
            dec_b2.get_left(),
            angle=-TAU / 8,
            color=YELLOW_3B1B,
            stroke_width=4,
        )
        kv_lbl = MathTex("K, V", font_size=26, color=YELLOW_3B1B).next_to(kv_arrow, UP, buff=0.1)

        # Causal Mask Matrix on the right
        mask_tex = MathTex(
            r"M_{\text{causal}} = \begin{bmatrix}"
            r"\alpha_{11} & -\infty & -\infty & -\infty \\"
            r"\alpha_{21} & \alpha_{22} & -\infty & -\infty \\"
            r"\alpha_{31} & \alpha_{32} & \alpha_{33} & -\infty \\"
            r"\alpha_{41} & \alpha_{42} & \alpha_{43} & \alpha_{44}"
            r"\end{bmatrix}",
            font_size=24,
            color=WHITE,
        ).move_to([4.55, -0.4, 0])
        mask_title = Text("Causal Mask (Look-Ahead):", font_size=20, color=RED_3B1B).next_to(mask_tex, UP, buff=0.25)

        self.play(Create(enc_frame), Write(enc_title), FadeIn(enc_b1), FadeIn(enc_b2), FadeIn(enc_b3))
        self.play(Create(dec_frame), Write(dec_title), FadeIn(dec_b1), FadeIn(dec_b2), FadeIn(dec_b3))
        self.play(Create(kv_arrow), Write(kv_lbl))
        self.play(Write(mask_title), Write(mask_tex))
        self.wait(2.5)

    def _layer_pill(self, text_str, color):
        rect = RoundedRectangle(
            corner_radius=0.1, width=3.2, height=0.62, color=color, fill_color=color, fill_opacity=0.2
        )
        txt = Text(text_str, font_size=16, color=WHITE).move_to(rect.get_center())
        return VGroup(rect, txt)
