# 🌌 Stanford CME 295: Transformers & LLMs — Лекція 1 (Візуальний конспект у стилі 3Blue1Brown)

> [!NOTE]
> **Курс:** Stanford CME 295 — *Transformers & Large Language Models* (Autumn 2026)
> **Автори лекції:** Afshine Amidi & Shervine Amidi (Stanford University)
> **Відеозапис:** [Stanford CME295 | Lecture 1 — Transformers (YouTube)](https://www.youtube.com/watch?v=114i2Kz-LZA)
> **Готовий Manim-код (Python):** [cme295_lecture1_manim.py](file:///Users/bibo/.gemini/antigravity/scratch/cme295_lecture1_manim/cme295_lecture1_manim.py)

---

## 🎞️ Галерея візуальних шпаргалок 3Blue1Brown (Огляд усіх 5 тем лекції)

Гортайте слайди каруселі нижче для швидкого візуального огляду всієї лекції — від токенізації BPE до повної архітектури Encoder–Decoder:

````carousel
![1. Токенізація та дерево злиття Byte-Pair Encoding (BPE)](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig1_tokenization_bpe_3b1b.svg)
<!-- slide -->
![2. Word2Vec: CBOW, Skip-gram та геометрія простору ембедингів ℝᵈ](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig2_word2vec_space_3b1b.svg)
<!-- slide -->
![3. Послідовне «пляшкове горло» RNN/LSTM проти прямого доступу Self-Attention](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig3_rnn_vs_attention_3b1b.svg)
<!-- slide -->
![4. Матрична механіка Scaled Dot-Product Self-Attention (Q, K, V) та Multi-Head Attention](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig4_self_attention_qkv_3b1b.svg)
<!-- slide -->
![5. Повна архітектура Transformer: Encoder, Decoder, Causal Mask (-∞) та Positional Encoding](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig5_transformer_arch_3b1b.svg)
````

---

## ⏱️ Таймлайн та структура Лекції 1

| Таймкод | Розділ | Ключове питання, яке вирішується |
| :--- | :--- | :--- |
| `00:00:00` | **Вступ та огляд курсу** | Як еволюціонували мовні моделі до сучасних LLM? |
| `00:16:56` | **1. Токенізація (Tokenization)** | Як розбити довільний текст на дискретні одиниці без вибуху словника $|V|$? |
| `00:37:27` | **2. Ембединги та Word2Vec** | Як перетворити індекси слів на вектори $\vec{v} \in \mathbb{R}^d$, де близькість = схожість змісту? |
| `00:48:26` | **3. Послідовні моделі (RNN, LSTM)** | Чому рекурентні мережі забувають далекий контекст і не масштабуються на GPU? |
| `00:55:18` | **4. Механізм Уваги (Attention)** | Як з'єднати будь-які два токени напряму за $O(1)$ кроків? |
| `00:59:54` | **5. Self-Attention ($Q, K, V$)** | Як матриці Запитів ($Q$), Ключів ($K$) та Значень ($V$) динамічно будують контекст? |
| `01:09:35` | **6. Архітектура Transformer** | Як працюють Positional Encoding, Residual + LayerNorm, Encoder та Decoder? |
| `01:22:16` | **7. Обчислювальні трюки** | Навіщо ділити на $\sqrt{d_k}$ та підставляти $-\infty$ у Causal Mask? |
| `01:27:35` | **8. Наскрізний приклад (End-to-End)** | Чим відрізняється паралельне тренування (Teacher Forcing) від авторегресійного інференсу? |

---

## 1. Токенізація та Byte-Pair Encoding (BPE) — `00:16:56`

![Токенізація: Word vs Character vs Subword BPE](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig1_tokenization_bpe_3b1b.svg)

Нейронна мережа не розуміє рядків тексту — вона приймає послідовність індексів токенів $(t_1, t_2, \dots, t_L)$, де кожен $t_i \in \{1, \dots, |V|\}$.

### Три підходи та їхній компроміс:

| Стратегія | Розмір словника $\|V\|$ | Довжина контексту $L$ | Головна перевага | Фатальний недолік |
| :--- | :--- | :--- | :--- | :--- |
| **Word-level** *(по словах)* | Величезний ($> 500\,000$) | Мінімальна | Кожне слово має цілісний зміст | **OOV (Out-of-Vocabulary)**: будь-яке нове слово або друкарська помилка стає токеном `[UNK]`; `look`, `looked`, `looking` не ділять спільний корінь |
| **Character-level** *(по символах)* | Мінімальний ($\approx 256$ байтів) | У $4\text{–}6$ разів довша | Жодного `[UNK]`, крихітна матриця ембедингів | Окремі літери не мають семантики; квадратична складність Attention $O(L^2)$ зростає у **25+ разів** |
| **Subword-level** *(BPE / WordPiece)* | Оптимальний ($32\text{k} - 128\text{k}$) | Компактна | Часті слова = 1 токен; рідкісні розкладаються на морфеми (`un` + `##aff` + `##able`); немає `[UNK]` | Потребує попереднього статистичного навчання токенізатора на корпусі |

> [!TIP]
> **Інтуїція 3Blue1Brown для алгоритму BPE:**
> Уявіть стиснення даних. Ми починаємо з окремих символів `["l", "o", "w", "e", "r"]`. На кожній ітерації ми шукаємо пару сусідніх токенів із максимальною частотою в корпусі:
> $$V_{k+1} = V_k \cup \left\{ \underset{(a, b) \in V_k \times V_k}{\arg\max}\ \text{freq}(a, b) \right\}$$
> Зливаючи `("e","r") → "er"`, потім `("l","o") → "lo"`, потім `("lo","w") → "low"`, ми автоматично відкриваємо корені та суфікси мови без жодних лінгвістичних правил!

---

## 2. Ембединги слів та Word2Vec (CBOW і Skip-gram) — `00:37:27`

![Word2Vec: CBOW, Skip-gram та семантичний простір](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig2_word2vec_space_3b1b.svg)

### Від ортогональних One-Hot векторів до геометрії змісту
Якщо закодувати слово індексом $i$ як **One-Hot вектор** $e_i \in \{0, 1\}^{|V|}$, то всі слова попарно перпендикулярні в $|V|$-вимірному просторі:
$$e_{\text{кіт}}^T e_{\text{кошеня}} = 0, \quad e_{\text{кіт}}^T e_{\text{бетон}} = 0$$

Замість цього ми множимо $e_i$ на навчальну матрицю ембедингів $E \in \mathbb{R}^{d \times |V|}$ (де $d \ll |V|$, наприклад $d = 768$), отримуючи щільний вектор ($i$-й стовпчик матриці $E$):
$$v_i = E e_i \in \mathbb{R}^d$$

### Як навчити матрицю $E$ без розмітки? (Self-Supervised Proxy Tasks)
Ідея **Word2Vec** (Mikolov et al.): *«Значення слова визначається його оточенням»*. Ми беремо ковзне вікно радіуса $c$ навколо центрального слова $w_t$:

1. **CBOW (Continuous Bag-of-Words):** За векторами слів контексту $\{w_{t-c}, \dots, w_{t+c}\} \setminus \{w_t\}$ (через їхнє середнє $\bar{v}_t = \frac{1}{2c}\sum_{j \neq 0} v_{w_{t+j}}$) передбачаємо центральне слово $w_t$:
   $$\mathcal{L}_{\text{CBOW}} = -\sum_{t=1}^{T} \log P(w_t \mid w_{t-c}, \dots, w_{t+c})$$
2. **Skip-gram:** За центральним словом $w_t$ передбачаємо кожне слово його контексту $w_{t+j}$ незалежно:
   $$\mathcal{L}_{\text{Skip-gram}} = -\sum_{t=1}^{T} \sum_{\substack{-c \le j \le c \\ j \neq 0}} \log P(w_{t+j} \mid w_t), \quad P(w_O \mid w_I) = \frac{\exp(u_{w_O}^T v_{w_I})}{\sum_{w \in V} \exp(u_w^T v_{w_I})}$$

> [!IMPORTANT]
> **Геометричний наслідок:** Оскільки логарифм імовірності пропорційний скалярному добутку $u_{w_O}^T v_{w_I}$, слова, які з'являються в однакових контекстах, змушені мати близькі вектори, а семантичні відношення перетворюються на паралельні вектори зсуву:
> $$\vec{v}(\text{king}) - \vec{v}(\text{man}) + \vec{v}(\text{woman}) \approx \vec{v}(\text{queen})$$

---

## 3. Чому RNN та LSTM поступилися Attention? — `00:48:26`

![Послідовні RNN/LSTM проти прямого зв'язку Self-Attention](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig3_rnn_vs_attention_3b1b.svg)

У рекурентних мережах (RNN, GRU, LSTM) прихований стан $h_t$ обчислюється строго послідовно:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

### Три критичні проблеми рекурентності:
1. **Інформаційне «пляшкове горло» (Bottleneck):** Вектор фіксованої розмірності $h_t \in \mathbb{R}^d$ має одночасно пам'ятати всі попередні слова речення.
2. **Затухання та вибух градієнтів (Vanishing / Exploding Gradients):** Під час Backpropagation Through Time (BPTT) градієнт від кінця речення $T$ до першого слова є ланцюговим добутком $T-1$ матриць Якобі:
   $$\frac{\partial \mathcal{L}_T}{\partial h_1} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=2}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$
   Якщо власні значення якобіанів $< 1$, градієнт експоненційно прямує до $0$ — модель фізично не може вивчити зв'язок між займенником `"it"` у кінці абзацу та іменником `"trophy"` на його початку.
3. **Відсутність паралелізації ($O(L)$ послідовних кроків):** GPU має тисячі ядер, але під час тренування RNN вони простоюють, очікуючи завершення кроку $t-1$, щоб почати крок $t$.

---

## 4. Механізм Self-Attention: Query, Key, Value — `00:55:18`

![Матрична механіка Scaled Dot-Product Self-Attention та Multi-Head Attention](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig4_self_attention_qkv_3b1b.svg)

Замість того щоб передавати інформацію по ланцюжку $h_1 \to h_2 \to \dots \to h_L$, **Self-Attention** дозволяє кожному токену напряму звернутися до кожного іншого токена за **$O(1)$ довжини шляху**.

### Геометрична інтуїція 3Blue1Brown: Що роблять $Q, K, V$?
З кожного вхідного вектора токена $x_i \in \mathbb{R}^{d_{\text{model}}}$ за допомогою трьох навчальних матриць $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ та $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ ми отримуємо три вектори:

* 🔵 **Query ($q_i = x_i W_Q$):** Вектор-запит («Яку інформацію я шукаю в контексті?»). Наприклад, вектор $q_{\text{it}}$ шукає неживий іменник однини.
* 🟡 **Key ($k_j = x_j W_K$):** Вектор-ключ («Які ознаки я пропоную іншим токенам?»). Вектор $k_{\text{trophy}}$ сигналізує: «Я іменник однини, предмет».
* 🟢 **Value ($v_j = x_j W_V$):** Вектор-значення («Який семантичний вміст я передам у новий вектор токена $i$, якщо мій ключ співпаде із запитом?»).

### Повне матричне рівняння Scaled Dot-Product Attention

Для матриці всіх токенів $X \in \mathbb{R}^{L \times d_{\text{model}}}$:
$$Q = X W_Q \in \mathbb{R}^{L \times d_k}, \quad K = X W_K \in \mathbb{R}^{L \times d_k}, \quad V = X W_V \in \mathbb{R}^{L \times d_v}$$

$$\text{Attention}(Q, K, V) = \underbrace{\text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right)}_{A \in \mathbb{R}^{L \times L}} V \in \mathbb{R}^{L \times d_v}$$

де елемент матриці уваги $\alpha_{ij}$ у рядку $i$ та стовпчику $j$ дорівнює:
$$\alpha_{ij} = \frac{\exp\!\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)}{\sum_{m=1}^{L} \exp\!\left(\frac{q_i \cdot k_m}{\sqrt{d_k}}\right)}, \qquad z_i = \sum_{j=1}^{L} \alpha_{ij} v_j$$

> [!CAUTION]
> **Математичне доведення: Навіщо ділити на $\sqrt{d_k}$? (`01:22:16`)**
> Нехай компоненти векторів $q, k \in \mathbb{R}^{d_k}$ є незалежними випадковими величинами з $\mathbb{E}[q_m] = \mathbb{E}[k_m] = 0$ та дисперсією $\text{Var}(q_m) = \text{Var}(k_m) = 1$.
> Тоді для кожного доданка $q_m k_m$:
> $$\mathbb{E}[q_m k_m] = \mathbb{E}[q_m]\mathbb{E}[k_m] = 0, \qquad \text{Var}(q_m k_m) = \mathbb{E}[q_m^2]\mathbb{E}[k_m^2] = 1 \cdot 1 = 1$$
> Оскільки скалярний добуток є сумою $d_k$ незалежних доданків $q \cdot k = \sum_{m=1}^{d_k} q_m k_m$, його дисперсія зростає лінійно з розмірністю $d_k$:
> $$\text{Var}(q \cdot k) = \sum_{m=1}^{d_k} \text{Var}(q_m k_m) = d_k \implies \text{Std}(q \cdot k) = \sqrt{d_k}$$
> Якщо $d_k = 64$, стандартне відхилення становить $8$, тому деякі логіти сягають $+15 \dots +25$. Тоді функція $\text{softmax}$ вироджується у сходинку (`[0, 1, 0, 0]`), де похідна $\frac{\partial \alpha_i}{\partial s_j} = \alpha_i(\delta_{ij} - \alpha_j) \approx 0$, і **градієнти зникають**. Ділення на $\sqrt{d_k}$ нормалізує дисперсію назад до $\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{(\sqrt{d_k})^2} = 1$!

### Multi-Head Attention (Багатоголова увага)
Замість одного великого простору уваги модель запускає $H$ паралельних «голів», кожна зі своїми матрицями $\left(W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}\right)$ розмірності $d_k = d_v = \frac{d_{\text{model}}}{H}$:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O, \quad W_O \in \mathbb{R}^{H d_v \times d_{\text{model}}}$$
Завдяки цьому `head₁` може відстежувати синтаксичний зв'язок дієслова з підметом, `head₂` — зв'язок займенника `"it"` з іменником `"trophy"`, а `head₃` — інтонаційний або часовий контекст.

---

## 5. Повна архітектура Transformer: Encoder, Decoder та Трюки — `01:09:35`

![Повна архітектура Transformer: Encoder, Decoder та Causal Mask](/Users/bibo/.gemini/antigravity/brain/c9ac2aff-9cc2-41e9-805c-67057b5e7eb6/fig5_transformer_arch_3b1b.svg)

### 1. Синусоїдальне Позиційне Кодування (Positional Encoding) — `01:12:19`
Операція Self-Attention симетрична відносно перестановки токенів (якщо поміняти слова місцями, їхні контекстні вектори будуть тими самими). Щоб передати порядок слів, до вектора ембедингу на позиції $pos \in \{0, \dots, L-1\}$ додається позиційний вектор $\text{PE}_{pos} \in \mathbb{R}^{d_{\text{model}}}$:
$$\text{PE}_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right), \qquad \text{PE}_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
*Чому саме синуси і косинуси?* За формулою додавання кутів вектор для позиції $pos + \Delta$ можна подати як ортогональну матрицю повороту $R_\Delta$, помножену на вектор позиції $pos$. Це дозволяє головам уваги легко обчислювати **відносну відстань $\Delta$** між токенами через скалярний добуток.

### 2. Залишкові з'єднання (Residual Connections) та LayerNorm — `01:16:27`
Кожен блок у Encoder та Decoder має вигляд:
$$z_{\text{out}} = \text{LayerNorm}(x + \text{SubLayer}(x))$$
* **Skip-Connection ($x + \text{SubLayer}(x)$):** Забезпечує безперешкодний потік градієнтів крізь десятки шарів, бо похідна $\frac{\partial (x + f(x))}{\partial x} = I + \frac{\partial f(x)}{\partial x}$ завжди має одиничну матрицю $I$.
* **Layer Normalization (на відміну від BatchNorm):** Обчислює середнє $\mu_i$ та дисперсію $\sigma_i^2$ **вздовж координат $d_{\text{model}}$ одного токена**, тому не залежить від довжини речення чи розміру батча:
  $$\text{LayerNorm}(z_i) = \gamma \odot \frac{z_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta$$

### 3. Causal Masking (Маскування майбутнього за допомогою $-\infty$) — `01:18:19`
У першому підшарі **Decoder** під час паралельного навчання (Teacher Forcing) токен на позиції $t$ має право бачити лише токени $1, 2, \dots, t$. Для цього перед Softmax додається матриця маски $M_{\text{causal}}$:
$$M_{ij} = \begin{cases} 0, & \text{якщо } j \le i \text{ (минуле і теперішнє дозволені)} \\ -\infty, & \text{якщо } j > i \text{ (майбутнє заблоковане)} \end{cases}$$
Оскільки $e^{-\infty} = 0$, після $\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)$ імовірності уваги до всіх майбутніх токенів стають строго нульовими без витоку інформації!

---

## 📊 Головна порівняльна таблиця для іспитів CME 295

| Характеристика | RNN / LSTM | Transformer Encoder (напр. BERT) | Transformer Decoder (напр. GPT) | Повний Encoder–Decoder (напр. T5) |
| :--- | :--- | :--- | :--- | :--- |
| **Напрямок контексту** | Послідовний (зліва направо) | **Двонапрямний** (бачить усі токени) | **Однонапрямний / Causal** (бачить лише $\le t$) | Encoder — двонапрямний, Decoder — Causal + Cross |
| **Послідовних операцій** | $O(L)$ (не паралелізується) | **$O(1)$** (повна паралелізація) | **$O(1)$** при тренуванні, $O(L)$ при генерації | **$O(1)$** при тренуванні |
| **Довжина шляху між токенами** | $O(L)$ (затухання сигналу) | **$O(1)$** (прямий скалярний добуток) | **$O(1)$** | **$O(1)$** |
| **Обчислювальна складність на шар** | $O(L \cdot d^2)$ | $O(L^2 \cdot d + L \cdot d^2)$ | $O(L^2 \cdot d + L \cdot d^2)$ | $O(L_{\text{enc}}^2 d + L_{\text{dec}}^2 d + L_{\text{enc}} L_{\text{dec}} d)$ |
| **Типові задачі** | Часові ряди, старі Seq2Seq | Класифікація, NER, пошукові ембединги | Авторегресійна генерація тексту, чат-боти | Машинний переклад, сумаризація |

---

## 🎬 Як зрендерити власні 3Blue1Brown MP4-відео з Manim-скрипта

Усі 5 сцен лекції вже написані на Python з використанням бібліотеки `manim` у файлі [cme295_lecture1_manim.py](file:///Users/bibo/.gemini/antigravity/scratch/cme295_lecture1_manim/cme295_lecture1_manim.py):

```bash
# 1. Перейдіть до папки проєкту
cd /Users/bibo/.gemini/antigravity/scratch/cme295_lecture1_manim

# 2. Встановіть Manim Community Edition (якщо ще не встановлено)
pip install manim

# 3. Зрендеріть усі 5 сцен у високій якості 1080p60 (-pqh -a)
manim -pqh -a cme295_lecture1_manim.py
```
