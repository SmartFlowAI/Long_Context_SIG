题前注明：

本文的数学证明主要借鉴 YaRN 论文中对于 RoPE 的表述，小白也能看得懂，看不懂那可能是我表达有问题，请尽情提出批评建议

关于后文$e^{im\theta}$的可视化代码会在评论区释出，请佬们帮忙 review 一下看有无问题。

是笔者初学长上下文的笔记，苏神的博客还没研究明白，后续研究懂了有问题会在本文更新

> 后续会继续看 YaRN 论文更新一个系列，主要是因为最近做 Dynamic-NTK 改底数之后微调性能变差，在找原因

# RoPE 代码

[ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://arxiv.org/abs/2104.09864)

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # `torch.arange(0, self.dim, 2)`: 生成一个从 `0` 开始，到 `self.dim` 结束（不包含 `self.dim`），步长为 `2` 的一维张量。这意味着张量将包含所有的偶数索引，直到小于 `self.dim` 的最大偶数
        # 将上述生成的偶数索引除以 `self.dim`，即将每个索引标准化到 `[0, 1)` 区间内。这样的标准化可以帮助调整每个频率成分的增长速度，使其在整个维度范围内平滑分布
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # torch.arange(start, end, step, device, dtype)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        # torch.einsum(equation, *operands)
        # "i,j->ij" 将第一个输入的每个元素与第二个输入的每个元素进行组合，形成一个二维张量
        # 操作数是 `t` 和 `self.inv_freq`。`t` 是一个一维张量，包含从 `0` 到 `self.max_seq_len_cached - 1` 的整数。`self.inv_freq` 是一个与模型的位置编码相关频率的倒数（inverse frequencies）。其实就是关于$\theta_i$的张量。

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # torch.cat(tensors, dim)
        # `tensors` 是 `(freqs, freqs)`，也就是将同一个张量 `freqs` 连接两次。
        # 参数 `dim=-1` 指定了拼接的维度。在 PyTorch 中，`-1` 表示最后一个维度。对于 `freqs` 如果是二维张量，比如形状为 `[n, m]`，那么 `dim=-1` 表示沿着每行的最内层维度进行连接。
        emb = torch.cat((freqs, freqs), dim=-1)
        # 此函数沿指定维度连接张量。这里， `freqs` 沿着最后一个维度与其自身连接（`dim=-1`）。此操作本质上使频率内容加倍，为同时进行余弦和正弦计算做好准备。通过复制频率，每对（余弦和正弦）并排设置，简化了后续操作。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        # 张量注册为 PyTorch 模块内的缓冲区。缓冲区与参数类似，但在反向传播期间不会更新。它们用于存储模型用于计算的常量或状态。这`persistent=False`参数指定该缓冲区不应该是由保存的模型状态的一部分
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
```

**详解：**

```
inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
```

`inv_freq` = $\theta = \frac{1}{base ^ \frac{2i}{dim}} $

`inv_freq` 用于生成不同位置的正弦和余弦编码的频率。通过调整基数和指数的标准化，这种方法能够生成一个丰富且分布均匀的频率集合，这些频率集合随后可以用于计算每个位置的正弦和余弦值

---

```python
# torch.arange(start, end, step, device, dtype)
t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
# torch.einsum(equation, *operands)
# "i,j->ij" 将第一个输入的每个元素与第二个输入的每个元素进行组合，形成一个二维张量
# 操作数是 `t` 和 `self.inv_freq`。`t` 是一个一维张量，包含从 `0` 到 `self.max_seq_len_cached - 1` 的整数。`self.inv_freq` 是一个与模型的位置编码相关频率的倒数（inverse frequencies）。
freqs = torch.einsum("i,j->ij", t, self.inv_freq)
```

`einsum` 操作在这种情况下实质上是在进行外积计算。外积是一种矩阵运算，用于生成两个向量的所有可能的元素对的乘积组成的矩阵。在这个具体场景中：

- `t`（假设其尺寸为 `N`）和 `self.inv_freq`（假设其尺寸为 `M`）进行外积操作，结果是一个形状为 `N x M` 的矩阵。
- 对于输出矩阵中的每一个元素 `freqs[i, j]`，它的值是 `t[i] * self.inv_freq[j]`。这意味着对于 `t` 中的每个时间步或位置索引，都与 `self.inv_freq` 中的每个频率值相乘，从而创建一个频率矩阵。

`torch.einsum("i,j->ij", t, self.inv_freq)` 操作执行以下步骤：

1.  对于 `t` 中的每个元素 `t[i]`，它与 `self.inv_freq` 中的每个元素 `self.inv_freq[j]` 相乘，得到一个二维张量，其中每个元素都是 `t[i] * self.inv_freq[j]`。这个过程可以看作是在 `t` 的每个位置上，都有一个完整的 `self.inv_freq` 向量与之相乘。

最终结果是一个二维张量，其中 `ij` 位置上的元素是 `t` 的第 `i` 个元素与 `self.inv_freq` 的第 `j` 个元素的乘积，并对所有的 `i` 和 `j` 进行求和。

这种操作通常用于构建位置编码或者将顺序信息集成到模型中，这在自然语言处理中非常常见，特别是在处理序列数据时。在 `LlamaRotaryEmbedding` 类中，`t` 可以看作是时间步（或位置）的表示，而 `self.inv_freq` 则是提供每个时间步特定频率的向量。这个操作最终生成一个二维张量，它同时包含了时间（位置）信息和频率信息，被后续的 `_set_cos_sin_cache` 方法用于构建旋转嵌入。

举个例子方便理解：

```python
self.base = 10000
self.dim = 8  # Example dimension
inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
max_seq_len_cached = 4  # Example maximum sequence length
t = torch.arange(max_seq_len_cached, device=device, dtype=inv_freq.dtype)

```

此时

```
inv_freq = 1.0 / (10000 ** (torch.tensor([0, 2, 4, 6]).float() / 8))
inv_freq = tensor([1.0000, 0.1000, 0.0100, 0.0010])
t = tensor([0, 1, 2, 3])
result = torch.einsum("i,j->ij", t, inv_freq)

```

计算得到

```
result = torch.einsum("i,j->ij", torch.tensor([0, 1, 2, 3]), torch.tensor([1.0000, 0.1000, 0.0100, 0.0010]))
result = tensor([[0.0000, 0.0000, 0.0000, 0.0000],
                 [1.0000, 0.1000, 0.0100, 0.0010],
                 [2.0000, 0.2000, 0.0200, 0.0020],
                 [3.0000, 0.3000, 0.0300, 0.0030]])
```

---

```
# torch.cat(tensors, dim)
emb = torch.cat((freqs, freqs), dim=-1)
```

`tensors` 是 `(freqs, freqs)`，也就是将同一个张量 `freqs` 连接两次。

参数 `dim=-1` 指定了拼接的维度。在 PyTorch 中，`-1` 表示最后一个维度。对于 `freqs` 如果是二维张量，比如形状为 `[n, m]`，那么 `dim=-1` 表示沿着每行的最内层维度进行连接，也就是列维度。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c6acf21065344364b3a4930cda8f1258~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=DPgj6sSlJpjxBH40sj5uc%2BIJQIE%3D)

### 为什么要拼接 `freqs` 两次

原因在于生成位置编码时通常需要同时使用正弦和余弦函数。为了生成完整的位置编码，你通常需要两组频率值：一组用于计算正弦值，另一组用于计算余弦值。通过将 `freqs` 与自身进行拼接，我们可以在一个张量中同时容纳用于计算正弦和余弦的频率值，从而简化后续的处理步骤。

### 如何使用这个拼接后的张量 `emb`

- 在这个例子中，拼接后的张量 `emb` 将直接用于计算所有位置的正弦和余弦值。这是通过对 `emb` 应用 `.cos()` 和 `.sin()` 函数完成的。
- 由于 `emb` 包含了每个位置的频率值两次，`emb.cos()` 将为每个频率计算余弦值，`emb.sin()` 将为每个频率计算正弦值。这意味着最终得到的余弦和正弦缓存（buffer）中，每个位置的编码将由相应的正弦和余弦值并列组成。

# RoPE 数学解释

给定一系列向量$\mathbf{x}_1,\cdotp\cdotp\cdotp,\mathbf{x}_{z}\in\mathbb{R}^{[\mathrm{D}]}$ 按照[RoFormer，也就是 RoPE 的论文](https://arxiv.org/abs/2104.09864) 的符号表示，注意力层首先将向量转换为 q 和 k

$$\mathbf{q}_{m}=f_{q}(\mathbf{x}_{m},m)\in\mathbb{R}^{|D|},\:\mathbf{k}_{n}=f_{k}(\mathbf{x}_{n},n)\in\mathbb{R}^{|D|}.$$

$q$是$m$维，$k$是$n$维

之后计算 softmax :  
$$\mathrm{softmax}(\frac{\mathbf{q}_m^T\mathbf{k}_n} {\sqrt{|D|}})$$ (其实就是注意力机制那个计算方法）

> where $\mathbf{q}_m,\mathbf{k}_n$ are considered as column vectors so that q$_m^T\mathbf{k}_n$ is simply the Euclidean inner product.

以上都和我们之前学的东西没有任何区别。

> In RoPE we first assume that $|D|$ is even and identify the embedding space and the hidden states as complex vector spaces:

但是在 RoPE 中，假设维度$|D|$是偶数 ("even"),以及将实数向量空间$\mathbb{R}^{|D|}$与复教向量空间$\mathbb{C}^{|D|/2}$相同构 (cong)即

$$\mathbb{R}^{|D|}\cong\mathbb{C}^{|D|/2}$$

$\mathbb{R}^{|D|}$表示一个由 $|D|$个实数组成的向量空间。

$\mathbb{C}^{|D|/2}$表示一个由$|D|/2$ 个复数组成的向量空间。

两个向量空间具有相同的结构和维度。这里的同构表示我们可以无损地从实数向量空间转换到复数向量空间，反之亦然。

在这种情况下

> where the inner product $q^{T}k$ becomes the real part of the standard Hermitian inner product Re($q^*k$). More
> specifically, the isomorphisms interleave the real part and the complex part <br>$$\left(\left(\mathbf{x}_{m}\right)_{1},\cdots,\left(\mathbf{x}_{m}\right)_{|D|}\right)\mapsto\left(\left(\mathbf{x}_{m}\right)_{1}+i\left(\mathbf{x}_{m}\right)_{2},\cdots,\left(\left(\mathbf{x}_{m}\right)_{|D|-1}+i\left(\mathbf{x}_{m}\right)_{|D|}\right)\right),\\\left(\left(\mathbf{q}_{m}\right)_{1},\cdots,\left(\mathbf{q}_{m}\right)_{|D|}\right)\mapsto\left(\left(\mathbf{q}_{m}\right)_{1}+i\left(\mathbf{q}_{m}\right)_{2},\cdots,\left(\left(\mathbf{q}_{m}\right)_{|D|-1}+i\left(\mathbf{q}_{m}\right)_{|D|}\right)\right).$$

首先解释什么是 the standard Hermitian inner product 标准赫米特（Hermitian）内积：

标准赫米特（Hermitian）内积是一种用于定义两个复数向量之间相互作用的重要概念。对于两个复数向量$\mathbf{u}=(u_1,u_2,...,u_n)$和$\mathbf{v}=(v_1,v_2,...,v_n)$,其中元素
$u_i,v_i\in\mathbb{C}$,它们的标准赫米特内积定义为：

$$\langle\mathbf{u},\mathbf{v}\rangle=\sum_{i=1}^n\overline{u_i}v_i$$

这里，$\overline{u_i}$表示$u_i$的共轭复数。如果$u_i=a+bi$ (其中$a,b$是实数，$i$是虚数单位),则其共轭$\overline u_i=a-bi$。

也就是说假设

$\mathbf{u}=\begin{bmatrix}a_1+b_1i\\c_1+d_1i\\e_1+f_1i\end{bmatrix}\\\mathbf{v}=\begin{bmatrix}a_2+b_2i\\c_2+d_2i\\e_2+f_2i\end{bmatrix}$

然后，计算$\overline{\mathbf{u}}^T\mathbf{v}:$
$$\langle\mathbf{u},\mathbf{v}\rangle=\begin{bmatrix}a_1-b_1i&c_1-d_1i&e_1-f_1i\end{bmatrix}\begin{bmatrix}a_2+b_2i\\c_2+d_2i\\e_2+f_2i\end{bmatrix}$$

将这些元素相乘并求和：

$$\langle\mathbf{u},\mathbf{v}\rangle=(a_1-b_1i)(a_2+b_2i)+(c_1-d_1i)(c_2+d_2i)+(e_1-f_1i)(e_2+f_2i)$$

分别展开每个乘积：

$$(a_1-b_1i)(a_2+b_2i)=a_1a_2+b_1b_2+i(b_2a_1-b_1a_2),\\(c_1-d_1i)(c_2+d_2i)=c_1c_2+d_1d_2+i(d_2c_1-d_1c_2),\\(e_1-f_1i)(e_2+f_2i)=e_1e_2+f_1f_2+i(f_2e_1-f_1e_2).$$

将这些结果相加：

$$\langle\mathbf{u},\mathbf{v}\rangle=(a_1a_2+b_1b_2+c_1c_2+d_1d_2+e_1e_2+f_1f_2)+i(b_2a_1-b_1a_2+d_2c_1−d_1c_2+f_2e_1−f_1e_2)$$

这里的实部和虚部分别是内积的实部和虚部，通常只关心实部，即赫米特内积的实部：

$$\mathrm{Re}(\langle\mathbf{u},\mathbf{v}\rangle)=a_1a_2+b_1b_2+c_1c_2+d_1d_2+e_1e_2+f_1f_2$$

这个时候我们可以根据$q^{T}k$是赫米特内积的实部反向推出$qk$应该是

$\mathbf{q}=\begin{bmatrix}a_1\\b_1\\c_1\\d_1\\e_1\\f_1\end{bmatrix}$
$\mathbf{k}=\begin{bmatrix}a_2\\b_2\\c_2\\d_2\\e_2\\f_2\end{bmatrix}$

也就是说会把$a_1,c_1,e_1$作为实部，$b_1,d_1,f_1$作为虚部交错插入

$$\left(\left(\mathbf{x}_{m}\right)_{1},\cdots,\left(\mathbf{x}_{m}\right)_{|D|}\right)\mapsto\left(\left(\mathbf{x}_{m}\right)_{1}+i\left(\mathbf{x}_{m}\right)_{2},\cdots,\left(\left(\mathbf{x}_{m}\right)_{|D|-1}+i\left(\mathbf{x}_{m}\right)_{|D|}\right)\right),\\\left(\left(\mathbf{q}_{m}\right)_{1},\cdots,\left(\mathbf{q}_{m}\right)_{|D|}\right)\mapsto\left(\left(\mathbf{q}_{m}\right)_{1}+i\left(\mathbf{q}_{m}\right)_{2},\cdots,\left(\left(\mathbf{q}_{m}\right)_{|D|-1}+i\left(\mathbf{q}_{m}\right)_{|D|}\right)\right).$$ 就是将相邻的实数元素配对转换为复数

这样我们完成了下式的第一步推导

$$\begin{aligned}&\langle f_{q}(\mathbf{x}_{m},m),f_{k}(\mathbf{x}_{n},n)\rangle_{\mathbb{R}}\\&=\:\mathrm{Re}(\langle f_{q}(\mathbf{x}_{m},m),f_{k}(\mathbf{x}_{n},n)\rangle_{\mathbb{C}})\\&=\:\mathrm{Re}(\mathbf{x}_{m}^{*}\mathbf{W}_{q}^{*}\mathbf{W}_{k}\mathbf{x}_{n}e^{\mathrm{i}\theta(m-n)})\\&=\:g(\mathbf{x}_{m},\mathbf{x}_{n},m-n).\end{aligned}$$

也即

$$\begin{aligned}&\langle f_{q}(\mathbf{x}_{m},m),f_{k}(\mathbf{x}_{n},n)\rangle_{\mathbb{R}}\\&=\:\mathrm{Re}(\langle f_{q}(\mathbf{x}_{m},m),f_{k}(\mathbf{x}_{n},n)\rangle_{\mathbb{C}}).\end{aligned}$$

---

**下面我们开始第三步的推导**

已知复数空间赫米特内积定义$$\langle\mathbf{u},\mathbf{v}\rangle=\sum_{i=1}^n\overline{u_i}v_i$$

则
$$\mathrm{Re}(\langle f_{q}(\mathbf{x}_{m},m),f_{k}(\mathbf{x}_{n},n)\rangle_{\mathbb{C}}) = \langle f_{q}^*(\mathbf{x}_{m},m)f_{k}(\mathbf{x}_{n},n)\rangle_{\mathbb{C}}. $$

$$ f*{q}^\*(\mathbf{x}*{m},m)$$ 是 $$ f*{q}(\mathbf{x}*{m},m)$$的共轭复数

我们现在要把
$$ f*{q}^\*(\mathbf{x}*{m},m) $$ 和 旋转的$\theta$联系起来得到

$$f_q(\mathbf{x}_m,m)=e^{im\theta}\mathbf{W}_q\mathbf{x}_m$$

---

$\theta = \begin{pmatrix}\cos m\theta_d&-\sin m\theta_d\\\sin m\theta_d&\cos m\theta_d\\\end{pmatrix}$ 代表了一个二维平面上绕原点的逆时针旋转 $\theta$ 弧度的变换。

设$z$是任意复数$z = x + yi$

$ze^{i\theta} = (x+yi)(\cos\theta + \sin\theta i) = (\cos\theta x - \sin\theta y ) + (\cos \theta y + \sin \theta x)i $

使用线代表示

$$\begin{pmatrix}\cos \theta_d&-\sin \theta_d\\\sin \theta_d&\cos \theta_d\\\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}\cos\theta x - \sin\theta y\\\cos \theta y + \sin \theta x\end{pmatrix} $$

因此这个变换在复数域中等价于复数乘法：$e^{i\theta}z$

同时由于$\theta = \begin{pmatrix}\cos m\theta_d&-\sin m\theta_d\\\sin m\theta_d&\cos m\theta_d\\\end{pmatrix}$
是一个正交矩阵，它保持了向量的欧几里得长度（或范数）。这意味着旋转操作不改变向量的大小，只改变其方向。这在将信息编码到复数域时非常重要，因为它保证了原始数据的信息量（例如能量）在转换过程中得到保留。

我们可以使用$$\mathbf{q}_{m}=\mathbf{W}_q\mathbf{x}_m，\mathbf{k}_{n}=\mathbf{W}_k\mathbf{x}_n$$实现线性变换 ，同时把上述的$\theta$推广到所有$m\theta_d$

这样就得到 $$f_q(\mathbf{x}_m,m) \\=\begin{pmatrix}\cos m\theta_d&-\sin m\theta_d\\\sin m\theta_d&\cos m\theta_d\\\end{pmatrix}\mathbf{W}_q\mathbf{x}_m \\=(\cos m\theta_d + \sin m\theta_d i)\mathbf{W}_q\mathbf{x}_m \\=   e^{im\theta}\mathbf{W}_q\mathbf{x}_m$$

同理
$$f_k(\mathbf{x}_n,n) = e^{in\theta}\mathbf{W}_k\mathbf{x}_n$$

> In real coordinates, the RoPE can be written using the following function
> $$\begin{gathered}f_{\mathbf{W}}(\mathbf{x}_m,m,\theta_d)=\begin{pmatrix}\cos m\theta_1&-\sin m\theta_1&0&0&\cdots&0&0\\\sin m\theta_1&\cos m\theta_1&0&0&\cdots&0&0\\0&0&\cos m\theta_2&-\sin m\theta_2&\cdots&0&0\\0&0&\sin m\theta_2&\cos m\theta_2&\cdots&0&0\\0&0&0&0&\cdots\cos m\theta_l-\sin m\theta_l\\0&0&0&0&\cdots\sin m\theta_l&\cos m\theta_l\end{pmatrix}\mathbf{W}\mathbf{x}_m,\end{gathered}$$

> so that
> $$f_q=f_{\mathbf{W}_q},\:f_k=f_{\mathbf{W}_k}.$$

> To convert embeddings $\mathbf{x}_m,\mathbf{x}_n$ into query and key vectors, we are first given $\mathbb{R}$-linear operators<br> >$$\mathbf{W}_q,\mathbf{W}_k:\mathbb{R}^{|D|}\to\mathbb{R}^{|D|}.$$

---

那么$\theta_{\tilde{d}}=base^{{-2d}/|\mathcal{D}|} = \frac{1}{base^{\frac{2d}{D}}}$
这个又是为什么呢？

$\theta_{\tilde{d}}$应该是旋转的角度，也就是说这个旋转的角度是为什么这样选择呢？

保证了随着 $d$ （即维度索引）的增加，旋转角度呈指数级衰减，这反映了对远离初始位置（如序列起始位置）的敏感度逐渐降低。

# 底数的变化造成的影响

$m\theta$的图像
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4897449683c3407eb9c6369933a68731~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=XUIATzxh0VnXpbVhkfhEm3YE9mk%3D)

这是$e^{ix}$的图像
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1aa4edd96c984107a6d67459fdfb68b3~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=C%2BWE2nsmfyJcbPciYnoE0Y9thk0%3D)

画出不同 base 下不同维度 d 的$e^{im\theta}$图像

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/174a976004f9427b9a34d8b90a5feb9c~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=RPXyMLi3ePfnu%2Be8BxCrdxEWzmI%3D)

base=10000 （d 作为 z 轴，实部和虚部作为 xy）
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0c86bfeaa5bf4f90abf7024620bf07c3~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=RlymfAkYhFi5ghJesIDDuEnis8c%3D)
base = 500000

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/00e42ef6528145ccaeda2b35f3bf8ccf~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=wmLjlEQE%2B8qFu696PWEY0bVxlKM%3D)

base = 1000000

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/eae4fde0fce8422d9d183ab78d2fa90f~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=HZYOmKu6u3sk6vFfLklSFfSbUF0%3D)

我们来看三视图对比

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a11d21077aa5466fa707a67e9242f88c~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=44%2F9Q1Zr5S8H%2FMc%2FCiySgqzkr1s%3D)

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/02ee016dbfbe423ebe4f182778a0c34f~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=lj%2B%2BeNnTdZ8qcc%2FzOhDUASLV%2Fdw%3D)

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7e194a2b132841da91e1fd67b1b1f84b~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNDEzOTAzODQwNDAwMDMyNyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723995938&x-orig-sign=Joy0erDogrQnjwNLimWOMvxECdg%3D)

可以看到旋转因子$e^{im\theta}$会随着底数变大，在低维度更稠密（变化的比较快），而高维度更稀疏，（变化少）

这个结论我也不知道对不对，佬们可以留言

# 其他

为什么$|D|$需要是偶数？

1.配对实数构成复数：在复数中，每个数由实部和虚部组成，例如$a+bi$。当将
实数向量空间$\mathbb{R}^{|D|}$视为复数向量空间$\mathbb{C}^{|D|/2}$时，实际上是将每两个实数 (一个作为实部，另一个作为虚部)配对构成一个复数。这要求$|D|$必须是偶数， 以确保所有维度都能被完整地配对。

# 参考文献

- [ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://arxiv.org/abs/2104.09864)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
