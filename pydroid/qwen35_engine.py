#!/usr/bin/env python3
"""Qwen3.5-4B dense chat engine, pure stdlib Python for Pydroid 3.

What it does, and where each trick comes from:
- GGUF reader .......... GGUF v3 layout (llama.cpp ggml/src/gguf.cpp)
- Real BPE tokenizer .... qwen2 pre-tokenizer + merge ranks (llama.cpp llama-vocab.cpp,
                          unicode.cpp). Replaces naive word-splitting.
- Correct quant math .... per-tensor type dispatch with exact ggml block layouts
                          (ggml/src/ggml-quants.c): F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K.
                          The old engine read every tensor as Q4_0, which is wrong
                          for Q4_K_M files (super-blocks, not 144-byte blocks).
- Streaming matvec ...... weights are dequantized block-by-block straight into the
                          dot product. A full matrix is NEVER materialized in RAM.
- Qwen3.5 hybrid fwd .... full-attention layers (fused Q+gate proj, per-head QK
                          RMSNorm, NeoX RoPE, sigmoid gate, post-norm) plus Gated
                          DeltaNet SSM layers (causal depthwise conv, L2-normed q/k,
                          scalar decay + beta recurrence), SwiGLU FFN
                          (llama.cpp src/models/qwen35.cpp, ggml gated_delta_net).
- MTP draft layers ...... skipped (block_count - nextn_predict_layers).
- Anti-think ............ <think> token logits forced to -inf during generation.

Needs only: the .gguf file next to it (or edit FULL_MODEL_PATH below).
RAM: mmap + streaming, roughly (KV cache + small buffers) only.
Speed: honest warning - pure-Python 4B forward pass is MINUTES per token.
Keep replies short (N_PREDICT) and prompts short.
"""

import os
import struct
import math
import time
import gc
import sys
import traceback
import mmap
import unicodedata


# --- CONFIGURATION & PATHS ---
MODEL_FILENAME = "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
PYDROID_PRIVATE_DIR = "/storage/emulated/0/Android/data/ru.iiec.pydroid3/files"
FULL_MODEL_PATH = os.path.join(PYDROID_PRIVATE_DIR, MODEL_FILENAME)
GGUF_MAGIC = 0x46554747
N_PREDICT = 128          # max new tokens per reply
SMALL_CACHE_MAX = 512 * 1024  # tensors smaller than this stay in RAM


# --- GGUF TYPE ENUMS (ggml/include/ggml.h) ---
T_F32, T_F16, T_Q4_0, T_Q4_1 = 0, 1, 2, 3
T_Q5_0, T_Q5_1, T_Q8_0 = 6, 7, 8
T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_Q8_K = 10, 11, 12, 13, 14, 15
T_BF16 = 30


# --- TERMINAL COLORS ---
class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'


# =====================================================================
# PART 1: GGUF READER
# =====================================================================
class GGUFParser:
    def __init__(self, fd, mm):
        self.fd = fd
        self.mm = mm
        self.cursor = 0
        self.tensors = {}
        self.metadata = {}
        self._parse_header()

    def _read_bytes(self, size):
        data = self.mm[self.cursor:self.cursor + size]
        self.cursor += size
        return data

    def _read(self, fmt):
        size = struct.calcsize(fmt)
        val = struct.unpack(fmt, self._read_bytes(size))
        return val[0] if len(val) == 1 else val

    def _read_string(self):
        length = self._read("<Q")
        return self._read_bytes(length).decode("utf-8", errors="ignore")

    def _skip_value(self, val_type):
        if val_type in (0, 1, 7):
            self.cursor += 1
        elif val_type in (2, 3):
            self.cursor += 2
        elif val_type in (4, 5, 6):
            self.cursor += 4
        elif val_type in (10, 11, 12):
            self.cursor += 8
        elif val_type == 8:
            self.cursor += self._read("<Q")
        elif val_type == 9:
            arr_type = self._read("<I")
            arr_len = self._read("<Q")
            for _ in range(arr_len):
                self._skip_value(arr_type)

    def _read_value(self, val_type):
        if val_type == 0:
            return self._read("<B")
        elif val_type == 1:
            return self._read("<b")
        elif val_type == 2:
            return self._read("<H")
        elif val_type == 3:
            return self._read("<h")
        elif val_type == 4:
            return self._read("<I")
        elif val_type == 5:
            return self._read("<i")
        elif val_type == 6:
            return self._read("<f")
        elif val_type == 7:
            return self._read("<?")
        elif val_type == 8:
            return self._read_string()
        elif val_type == 9:
            arr_type = self._read("<I")
            arr_len = self._read("<Q")
            return [self._read_value(arr_type) for _ in range(arr_len)]
        elif val_type == 10:
            return self._read("<Q")
        elif val_type == 11:
            return self._read("<q")
        elif val_type == 12:
            return self._read("<d")
        return None

    def _parse_header(self):
        if self._read("<I") != GGUF_MAGIC:
            raise ValueError("Invalid GGUF Header.")
        self.version = self._read("<I")
        if self.version == 1:
            tensor_count, metadata_count = self._read("<I"), self._read("<I")
        else:
            tensor_count, metadata_count = self._read("<Q"), self._read("<Q")

        for _ in range(metadata_count):
            key = self._read_string()
            val_type = self._read("<I")
            self.metadata[key] = self._read_value(val_type)

        for _ in range(tensor_count):
            name = self._read_string()
            n_dims = self._read("<I")
            dims = [self._read("<Q") for _ in range(n_dims)]
            type_enum, offset = self._read("<I"), self._read("<Q")
            self.tensors[name] = {"dims": dims, "type": type_enum, "offset": offset}

        alignment = self.metadata.get("general.alignment", 32)
        pad = (alignment - (self.cursor % alignment)) % alignment
        self.data_offset = self.cursor + pad

    def meta(self, key, default=None):
        return self.metadata.get(key, default)

    def require_tensor(self, *names):
        for n in names:
            if n in self.tensors:
                return n
        close = [k for k in self.tensors if any(k.startswith(n.rsplit(".", 1)[0]) for n in names)]
        raise KeyError("missing tensor %s (similar: %s)" % (names[0], close[:6]))


# =====================================================================
# PART 2: REAL BPE TOKENIZER (qwen2 pre-tokenizer + merge ranks)
# =====================================================================
def _build_byte_maps():
    # GPT-2 style byte mapping (llama.cpp unicode.cpp unicode_byte_to_utf8_map)
    b2u = {}
    u2b = {}
    n = 0
    for b in range(256):
        if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
            ch = chr(b)
        else:
            ch = chr(0x0100 + n)
            n += 1
        b2u[b] = ch
        u2b[ch] = b
    return b2u, u2b


_BYTE2UNI, _UNI2BYTE = _build_byte_maps()


def _is_L(ch):
    return unicodedata.category(ch).startswith("L")


def _is_N(ch):
    return unicodedata.category(ch).startswith("N")


def _qwen2_split(text):
    """Hand port of unicode_regex_split_custom_qwen2 (llama.cpp unicode.cpp).

    Rules in order:
    1. 's 't 're 've 'm 'll 'd (any case)
    2. optional single non(L|N|CR|LF) + letters+
    3. single number
    4. optional space + symbols+ + optional newlines
    5. spaces* + newlines+
    6. trailing spaces
    7. spaces+
    8. else one char
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # rule 1: contractions
        if c == "'" and i + 1 < n:
            rest = text[i + 1:].lower()
            hit = 0
            if rest.startswith("re") or rest.startswith("ve") or rest.startswith("ll"):
                hit = 3
            elif rest[0] in "stmd":
                hit = 2
            if hit:
                out.append(text[i:i + hit])
                i += hit
                continue
        # rule 2: [^L|N|CR|LF]? letters+
        j = i
        if c not in ("\r", "\n") and not _is_L(c) and not _is_N(c):
            j += 1
        k = j
        while k < n and _is_L(text[k]):
            k += 1
        if k > j:
            out.append(text[i:k])
            i = k
            continue
        # rule 3: single number
        if _is_N(c):
            out.append(c)
            i += 1
            continue
        # rule 4: optional space + symbols + newlines
        j = i
        if c == " ":
            j += 1
        k = j
        while k < n and text[k] not in (" ", "\t") and not _is_L(text[k]) and not _is_N(text[k]) \
                and text[k] not in ("\r", "\n"):
            k += 1
        if k > j:
            while k < n and text[k] in ("\r", "\n"):
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # rule 5: spaces* + newlines+
        if c in (" ", "\t", "\r", "\n"):
            k = i
            while k < n and text[k] in (" ", "\t"):
                k += 1
            m = k
            while m < n and text[m] in ("\r", "\n"):
                m += 1
            if m > k:
                out.append(text[i:m])
                i = m
                continue
            # rule 6/7: spaces (trailing or not - same split here)
            k = i
            while k < n and text[k] in (" ", "\t"):
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # rule 8: one char
        out.append(c)
        i += 1
    return out


class BPETokenizer:
    def __init__(self, parser):
        md = parser.metadata
        self.tokens = md.get("tokenizer.ggml.tokens", [])
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.merges = md.get("tokenizer.ggml.merges", [])
        self.ranks = {}
        for i, m in enumerate(self.merges):
            p = m.find(" ", 1)
            if p < 0:
                continue
            self.ranks[(m[:p], m[p + 1:])] = i
        types = md.get("tokenizer.ggml.token_type", [])
        self.token_type = list(types) + [1] * (len(self.tokens) - len(types))
        self.bos_id = md.get("tokenizer.ggml.bos_token_id")
        self.eos_id = md.get("tokenizer.ggml.eos_token_id")
        self.eot_id = md.get("tokenizer.ggml.eot_token_id")
        self.unk_id = md.get("tokenizer.ggml.unknown_token_id")
        self.add_bos = bool(md.get("tokenizer.ggml.add_bos_token", False))
        self.add_eos = bool(md.get("tokenizer.ggml.add_eos_token", False))
        # special texts, longest first (control/user-defined/unknown)
        self.specials = sorted(
            [t for i, t in enumerate(self.tokens)
             if i < len(self.token_type) and self.token_type[i] in (2, 3, 4) and t],
            key=len, reverse=True)
        # end-of-generation set
        self.eog = set()
        for tid in (self.eos_id, self.eot_id):
            if isinstance(tid, int) and 0 <= tid < len(self.tokens):
                self.eog.add(tid)
        for txt in ("<|im_end|>", "<|endoftext|>"):
            if txt in self.token_to_id:
                self.eog.add(self.token_to_id[txt])
        # anti-think ids (logit suppression target)
        self.anti_think = set()
        for i, t in enumerate(self.tokens):
            lt = t.lower()
            if "<think>" in lt or "</think>" in lt or "<|thought|>" in lt:
                self.anti_think.add(i)

    def _split_special(self, text):
        """Split out special tokens longest-first, like tokenizer_st_partition."""
        frags = []  # (is_special, text)
        i = 0
        buf = []
        while i < len(text):
            hit = None
            for s in self.specials:
                if text.startswith(s, i):
                    hit = s
                    break
            if hit is not None:
                if buf:
                    frags.append((False, "".join(buf)))
                    buf = []
                frags.append((True, hit))
                i += len(hit)
            else:
                buf.append(text[i])
                i += 1
        if buf:
            frags.append((False, "".join(buf)))
        return frags

    def _bpe_piece(self, encoded):
        parts = list(encoded)
        if len(parts) < 2:
            return parts
        while True:
            best_i, best_r = -1, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_r, best_i = r, i
            if best_i < 0:
                break
            parts[best_i] = parts[best_i] + parts[best_i + 1]
            del parts[best_i + 1]
        return parts

    def encode(self, text, add_bos=None):
        ids = []
        if add_bos is None:
            add_bos = self.add_bos
        if add_bos and isinstance(self.bos_id, int):
            ids.append(self.bos_id)
        for is_special, frag in self._split_special(text):
            if is_special:
                tid = self.token_to_id.get(frag)
                if tid is not None:
                    ids.append(tid)
                continue
            for piece in _qwen2_split(frag):
                enc = "".join(_BYTE2UNI[b] for b in piece.encode("utf-8"))
                for tok in self._bpe_piece(enc):
                    tid = self.token_to_id.get(tok)
                    if tid is not None:
                        ids.append(tid)
                    else:
                        # unknown bytes are dropped, like llama.cpp BPE
                        for ch in tok:
                            b = _UNI2BYTE.get(ch)
                            if b is None:
                                continue
                            tid2 = self.token_to_id.get(ch)
                            if tid2 is not None:
                                ids.append(tid2)
        return ids if ids else ([self.unk_id] if isinstance(self.unk_id, int) else [0])

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if 0 <= i < len(self.tokens):
                t = self.tokens[i]
                for ch in t:
                    b = _UNI2BYTE.get(ch)
                    if b is not None:
                        out.append(b)
        return bytes(out).decode("utf-8", errors="ignore")


# =====================================================================
# PART 3: MATH
# =====================================================================
def rms_norm_vec(x, w, eps):
    ss = 0.0
    for v in x:
        ss += v * v
    inv = 1.0 / math.sqrt(ss / len(x) + eps)
    n = len(w)
    return [v * inv * (w[i] if i < n else 1.0) for i, v in enumerate(x)]


def l2_norm_vec(x, eps):
    ss = 0.0
    for v in x:
        ss += v * v
    inv = 1.0 / math.sqrt(ss + eps)
    return [v * inv for v in x]


def softmax_vec(x):
    if not x:
        return []
    m = x[0]
    for v in x:
        if v > m:
            m = v
    es = [math.exp(v - m) for v in x]
    s = sum(es)
    return [v / s for v in es]


def silu_vec(x):
    return [v / (1.0 + math.exp(-v if v > -88.0 else -88.0)) for v in x]


def sigmoid_vec(x):
    return [1.0 / (1.0 + math.exp(-v)) for v in x]


# =====================================================================
# PART 4: STREAMING DEQUANT + MATVEC (never materialize a matrix)
# =====================================================================
_F16 = struct.Struct("<e")
_F32 = struct.Struct("<f")


def _f16(b):
    return _F16.unpack(b)[0]


def _k_scales_mins(sc):
    # 6-bit scale/min expansion, ggml get_scale_min_k4
    s = [0] * 8
    m = [0] * 8
    for j in range(4):
        s[j] = sc[j] & 63
        m[j] = sc[j + 4] & 63
    for j in range(4, 8):
        s[j] = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4)
        m[j] = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)
    return s, m


class WeightReader:
    """Zero-copy streaming matvec. Only block-sized chunks touch RAM."""

    def __init__(self, mm, data_offset, tensors):
        self.mm = mm
        self.base = data_offset
        self.tensors = tensors
        self.cache = {}

    def nbytes(self, name):
        d = self.tensors[name]["dims"]
        t = self.tensors[name]["type"]
        n = 1
        for v in d:
            n *= v
        qk, nb = _block_info(t)
        return n // qk * nb

    def cache_small(self, name):
        # full dequant of a small tensor, kept in RAM (bounded by caller)
        if name in self.cache:
            return self.cache[name]
        info = self.tensors[name]
        dims, t = info["dims"], info["type"]
        off = self.base + info["offset"]
        k = dims[0]
        nrows = 1
        for v in dims[1:]:
            nrows *= v
        qk, nb = _block_info(t)
        row_bytes = k // qk * nb
        v = []
        for r in range(nrows):
            v.extend(_dequant_row(self.mm, off + r * row_bytes, t, k))
        self.cache[name] = v
        return v

    def read_row_full(self, name, row):
        """Read one full dequantized row (or a 1-D tensor when row=None)."""
        info = self.tensors[name]
        dims, t = info["dims"], info["type"]
        off = self.base + info["offset"]
        if row is None:
            k = dims[0]
            base = off
        else:
            k = dims[0]
            qk, nb = _block_info(t)
            base = off + row * (k // qk * nb)
        return _dequant_row(self.mm, base, t, k)

    def matvec(self, name, x):
        """y[r] = row_r . x, streaming block by block."""
        info = self.tensors[name]
        dims, t = info["dims"], info["type"]
        k = dims[0]
        nrows = 1
        for v in dims[1:]:
            nrows *= v
        off = self.base + info["offset"]
        qk, nb = _block_info(t)
        nblocks = k // qk
        row_bytes = nblocks * nb
        out = [0.0] * nrows
        if t == T_F32:
            for r in range(nrows):
                base = off + r * row_bytes
                row = struct.unpack("<%df" % k, self.mm[base:base + row_bytes])
                s = 0.0
                for i in range(k):
                    s += row[i] * x[i]
                out[r] = s
            return out
        if t == T_F16:
            for r in range(nrows):
                base = off + r * row_bytes
                row = struct.unpack("<%de" % k, self.mm[base:base + row_bytes])
                s = 0.0
                for i in range(k):
                    s += row[i] * x[i]
                out[r] = s
            return out
        if t == T_BF16:
            for r in range(nrows):
                base = off + r * row_bytes
                raw = self.mm[base:base + row_bytes]
                s = 0.0
                for i in range(k):
                    bits = raw[2 * i] | (raw[2 * i + 1] << 8)
                    s += _F32.unpack("<f", struct.pack("<I", bits << 16))[0] * x[i]
                out[r] = s
            return out
        if t == T_Q8_0:
            for r in range(nrows):
                base = off + r * row_bytes
                acc = 0.0
                for b in range(nblocks):
                    bo = base + b * 34
                    d = _f16(self.mm[bo:bo + 2])
                    blk = self.mm[bo + 2:bo + 34]
                    xs = x[b * 32:(b + 1) * 32]
                    s = 0.0
                    for j in range(32):
                        q = blk[j]
                        s += (q - 256 if q > 127 else q) * xs[j]
                    acc += d * s
                out[r] = acc
            return out
        if t == T_Q4_0:
            for r in range(nrows):
                base = off + r * row_bytes
                acc = 0.0
                for b in range(nblocks):
                    bo = base + b * 18
                    d = _f16(self.mm[bo:bo + 2])
                    blk = self.mm[bo + 2:bo + 18]
                    xs = x[b * 32:(b + 1) * 32]
                    s = 0.0
                    for j in range(16):
                        q = blk[j]
                        s += ((q & 15) - 8) * xs[j] + ((q >> 4) - 8) * xs[j + 16]
                    acc += d * s
                out[r] = acc
            return out
        if t == T_Q4_K:
            for r in range(nrows):
                base = off + r * row_bytes
                acc = 0.0
                for b in range(nblocks):
                    bo = base + b * 144
                    d = _f16(self.mm[bo:bo + 2])
                    dm = _f16(self.mm[bo + 2:bo + 4])
                    sc, mn = _k_scales_mins(self.mm[bo + 4:bo + 16])
                    qs = self.mm[bo + 16:bo + 144]
                    xx = x[b * 256:(b + 1) * 256]
                    for c in range(4):
                        is_ = c * 2
                        d1 = d * sc[is_]
                        m1 = dm * mn[is_]
                        d2 = d * sc[is_ + 1]
                        m2 = dm * mn[is_ + 1]
                        qo = c * 32
                        yo = c * 64
                        for l in range(32):
                            qb = qs[qo + l]
                            acc += (d1 * (qb & 15) - m1) * xx[yo + l] + \
                                   (d2 * (qb >> 4) - m2) * xx[yo + 32 + l]
                out[r] = acc
            return out
        if t == T_Q5_K:
            for r in range(nrows):
                base = off + r * row_bytes
                acc = 0.0
                for b in range(nblocks):
                    bo = base + b * 176
                    d = _f16(self.mm[bo:bo + 2])
                    dm = _f16(self.mm[bo + 2:bo + 4])
                    sc, mn = _k_scales_mins(self.mm[bo + 4:bo + 16])
                    qh = self.mm[bo + 16:bo + 48]
                    qs = self.mm[bo + 48:bo + 176]
                    xx = x[b * 256:(b + 1) * 256]
                    u1, u2 = 1, 2
                    for c in range(4):
                        is_ = c * 2
                        d1 = d * sc[is_]
                        m1 = dm * mn[is_]
                        d2 = d * sc[is_ + 1]
                        m2 = dm * mn[is_ + 1]
                        qo = c * 32
                        yo = c * 64
                        for l in range(32):
                            qb = qs[qo + l]
                            hb = qh[l]
                            qlo = (qb & 15) + (16 if hb & u1 else 0)
                            qhi = (qb >> 4) + (16 if hb & u2 else 0)
                            acc += (d1 * qlo - m1) * xx[yo + l] + \
                                   (d2 * qhi - m2) * xx[yo + 32 + l]
                        u1 <<= 2
                        u2 <<= 2
                out[r] = acc
            return out
        if t == T_Q6_K:
            for r in range(nrows):
                base = off + r * row_bytes
                acc = 0.0
                for b in range(nblocks):
                    bo = base + b * 210
                    ql = self.mm[bo:bo + 128]
                    qh = self.mm[bo + 128:bo + 192]
                    s8 = self.mm[bo + 192:bo + 208]
                    d = _f16(self.mm[bo + 208:bo + 210])
                    xx = x[b * 256:(b + 1) * 256]
                    for h in range(2):
                        qo = h * 64
                        ho = h * 32
                        so = h * 8
                        yo = h * 128
                        for l in range(32):
                            is_ = l // 16
                            hb = qh[ho + l]
                            q1 = ((ql[qo + l] & 15) | (((hb >> 0) & 3) << 4)) - 32
                            q2 = ((ql[qo + l + 32] & 15) | (((hb >> 2) & 3) << 4)) - 32
                            q3 = ((ql[qo + l] >> 4) | (((hb >> 4) & 3) << 4)) - 32
                            q4 = ((ql[qo + l + 32] >> 4) | (((hb >> 6) & 3) << 4)) - 32
                            s1 = s8[so + is_]
                            s1 = s1 - 256 if s1 > 127 else s1
                            s2 = s8[so + is_ + 2]
                            s2 = s2 - 256 if s2 > 127 else s2
                            s3 = s8[so + is_ + 4]
                            s3 = s3 - 256 if s3 > 127 else s3
                            s4 = s8[so + is_ + 6]
                            s4 = s4 - 256 if s4 > 127 else s4
                            acc += d * (s1 * q1 * xx[yo + l] + s2 * q2 * xx[yo + l + 32] +
                                        s3 * q3 * xx[yo + l + 64] + s4 * q4 * xx[yo + l + 96])
                out[r] = acc
            return out
        raise ValueError("unsupported tensor type %d (only F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K)" % t)


def _block_info(t):
    if t == T_F32:
        return 1, 4
    if t in (T_F16, T_BF16):
        return 1, 2
    if t == T_Q4_0:
        return 32, 18
    if t == T_Q8_0:
        return 32, 34
    if t == T_Q4_K:
        return 256, 144
    if t == T_Q5_K:
        return 256, 176
    if t == T_Q6_K:
        return 256, 210
    raise ValueError("unsupported tensor type %d" % t)


def _dequant_row(mm, base, t, k):
    """Full-row dequant (small tensors / embeddings only)."""
    if t == T_F32:
        return list(struct.unpack("<%df" % k, mm[base:base + 4 * k]))
    if t == T_F16:
        return list(struct.unpack("<%de" % k, mm[base:base + 2 * k]))
    if t == T_BF16:
        raw = mm[base:base + 2 * k]
        return [_F32.unpack("<f", struct.pack("<I", (raw[2 * i] | (raw[2 * i + 1] << 8)) << 16))[0]
                for i in range(k)]
    out = []
    qk, nb = _block_info(t)
    nb_total = k // qk
    if t == T_Q8_0:
        for b in range(nb_total):
            bo = base + b * 34
            d = _f16(mm[bo:bo + 2])
            blk = mm[bo + 2:bo + 34]
            for j in range(32):
                q = blk[j]
                out.append(d * (q - 256 if q > 127 else q))
        return out
    if t == T_Q4_0:
        for b in range(nb_total):
            bo = base + b * 18
            d = _f16(mm[bo:bo + 2])
            blk = mm[bo + 2:bo + 18]
            for j in range(16):
                q = blk[j]
                out.append(d * ((q & 15) - 8))
            for j in range(16):
                out.append(d * ((mm[bo + 2 + j] >> 4) - 8))
        return out
    if t == T_Q4_K:
        for b in range(nb_total):
            bo = base + b * 144
            d = _f16(mm[bo:bo + 2])
            dm = _f16(mm[bo + 2:bo + 4])
            sc, mn = _k_scales_mins(mm[bo + 4:bo + 16])
            qs = mm[bo + 16:bo + 144]
            for c in range(4):
                is_ = c * 2
                d1, m1 = d * sc[is_], dm * mn[is_]
                d2, m2 = d * sc[is_ + 1], dm * mn[is_ + 1]
                qo = c * 32
                for l in range(32):
                    qb = qs[qo + l]
                    out.append(d1 * (qb & 15) - m1)
                for l in range(32):
                    out.append(d2 * (qs[qo + l] >> 4) - m2)
        return out
    if t == T_Q5_K:
        for b in range(nb_total):
            bo = base + b * 176
            d = _f16(mm[bo:bo + 2])
            dm = _f16(mm[bo + 2:bo + 4])
            sc, mn = _k_scales_mins(mm[bo + 4:bo + 16])
            qh = mm[bo + 16:bo + 48]
            qs = mm[bo + 48:bo + 176]
            u1, u2 = 1, 2
            for c in range(4):
                is_ = c * 2
                d1, m1 = d * sc[is_], dm * mn[is_]
                d2, m2 = d * sc[is_ + 1], dm * mn[is_ + 1]
                qo = c * 32
                for l in range(32):
                    qb = qs[qo + l]
                    out.append(d1 * ((qb & 15) + (16 if qh[l] & u1 else 0)) - m1)
                for l in range(32):
                    qb = qs[qo + l]
                    out.append(d2 * ((qb >> 4) + (16 if qh[l] & u2 else 0)) - m2)
                u1 <<= 2
                u2 <<= 2
        return out
    if t == T_Q6_K:
        for b in range(nb_total):
            bo = base + b * 210
            ql = mm[bo:bo + 128]
            qh = mm[bo + 128:bo + 192]
            s8 = mm[bo + 192:bo + 208]
            d = _f16(mm[bo + 208:bo + 210])
            sc = [(v - 256 if v > 127 else v) for v in s8]
            for h in range(2):
                qo, ho, so = h * 64, h * 32, h * 8
                for l in range(32):
                    is_ = l // 16
                    hb = qh[ho + l]
                    out.append(d * sc[so + is_] * ((((ql[qo + l] & 15) | (((hb >> 0) & 3) << 4))) - 32))
                for l in range(32):
                    is_ = l // 16
                    hb = qh[ho + l]
                    out.append(d * sc[so + is_ + 2] * ((((ql[qo + l + 32] & 15) | (((hb >> 2) & 3) << 4))) - 32))
                for l in range(32):
                    is_ = l // 16
                    hb = qh[ho + l]
                    out.append(d * sc[so + is_ + 4] * ((((ql[qo + l] >> 4) | (((hb >> 4) & 3) << 4))) - 32))
                for l in range(32):
                    is_ = l // 16
                    hb = qh[ho + l]
                    out.append(d * sc[so + is_ + 6] * ((((ql[qo + l + 32] >> 4) | (((hb >> 6) & 3) << 4))) - 32))
        return out
    raise ValueError("unsupported tensor type %d" % t)


# =====================================================================
# PART 5: MODEL (Qwen3.5 dense hybrid)
# =====================================================================
class Qwen35Engine:
    def __init__(self, model_path):
        self.fd = os.open(model_path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        self.parser = GGUFParser(self.fd, self.mm)
        md = self.parser.metadata
        self.arch = md.get("general.architecture", "qwen35")
        if "moe" in self.arch:
            raise ValueError("MoE arch '%s' not supported by this engine" % self.arch)
        A = self.arch + "."
        self.n_layer_all = int(md[A + "block_count"])
        n_nextn = int(md.get(A + "nextn_predict_layers", 0))
        self.n_layer = self.n_layer_all - n_nextn
        if "blk.0.attn_norm.weight" not in self.parser.tensors:
            raise ValueError("mtp-only file, no trunk blocks")
        self.n_embd = int(self.parser.tensors["token_embd.weight"]["dims"][0])
        self.n_vocab = int(self.parser.tensors["token_embd.weight"]["dims"][1])
        self.H = int(md[A + "attention.head_count"])
        self.Hkv = int(md.get(A + "attention.head_count_kv", self.H))
        self.Dk = int(md.get(A + "attention.key_length", self.n_embd // self.H))
        self.Dv = int(md.get(A + "attention.value_length", self.Dk))
        self.eps = float(md.get(A + "attention.layer_norm_rms_epsilon", 1e-6))
        self.rope_base = float(md.get(A + "rope.freq_base", 10000.0))
        self.n_rot = int(md.get(A + "rope.dimension_count", self.Dk))
        # recurrent (SSM) layer set
        rec = md.get(A + "attention.recurrent_layers")
        if rec:
            self.recr = set(int(v) for v in rec)
        else:
            interval = int(md.get(A + "full_attention_interval", 4))
            self.recr = set(i for i in range(self.n_layer) if (i + 1) % interval != 0)
        # SSM dims
        self.ssm_inner = int(md.get(A + "ssm.inner_size", 0))
        self.ssm_state = int(md.get(A + "ssm.state_size", 0))
        self.ssm_rank = int(md.get(A + "ssm.time_step_rank", 0))
        self.ssm_group = int(md.get(A + "ssm.group_count", 0))
        self.ssm_conv = int(md.get(A + "ssm.conv_kernel", 0))
        # rope freqs (NeoX pairing)
        self.rope_freq = [self.rope_base ** (-2.0 * j / self.Dk) for j in range(self.Dk // 2)]
        self.rd = WeightReader(self.mm, self.parser.data_offset, self.parser.tensors)
        # cache small tensors in RAM
        for name in self.parser.tensors:
            try:
                if self.rd.nbytes(name) <= SMALL_CACHE_MAX:
                    self.rd.cache_small(name)
            except ValueError:
                pass
        # tokenizer
        self.tok = BPETokenizer(self.parser)
        # state
        self.kv = {}   # il -> [k_list, v_list] for attention layers
        self.ssm_s = {}  # il -> [head][i][j] state matrices
        self.ssm_c = {}  # il -> [channel][past] conv windows
        self.pos = 0
        # structural validation (fail fast with a clear message)
        fa = next((i for i in range(self.n_layer) if i not in self.recr), None)
        if fa is None:
            raise ValueError("no full-attention layer found")
        qd = self.parser.tensors["blk.%d.attn_q.weight" % fa]["dims"]
        if qd[1] != 2 * self.H * self.Dk:
            raise ValueError("attn_q rows %d != 2*H*Dk=%d" % (qd[1], 2 * self.H * self.Dk))
        if self.Dv != self.Dk:
            raise ValueError("value_length != key_length, unsupported")
        if self.recr:
            il = next(iter(sorted(self.recr)))
            kd = self.ssm_state * self.ssm_group
            sv = self.ssm_inner // self.ssm_rank
            vd = sv * self.ssm_rank
            wd = self.parser.tensors["blk.%d.attn_qkv.weight" % il]["dims"]
            if wd[1] != 2 * kd + vd:
                raise ValueError("ssm wqkv rows %d != 2*key_dim+value_dim=%d" % (wd[1], 2 * kd + vd))
        n_attn = sum(1 for i in range(self.n_layer) if i not in self.recr)
        print("%s[Engine] arch=%s layers=%d (attn=%d ssm=%d) embd=%d vocab=%d heads=%d kv=%d%s" %
              (Color.GREEN, self.arch, self.n_layer, n_attn, self.n_layer - n_attn,
               self.n_embd, self.n_vocab, self.H, self.Hkv, Color.RESET))

    @staticmethod
    def _conv_step(win, kernel):
        # win oldest..newest, len == len(kernel); silu(dot)
        s = 0.0
        for t in range(len(kernel)):
            s += win[t] * kernel[t]
        return s / (1.0 + math.exp(-s if s > -88.0 else -88.0))

    @staticmethod
    def _delta_step(S, q, k, v, gate, beta):
        # one Gated DeltaNet step, S is Sv x Sv, updated in place, returns out
        Sv = len(v)
        dec = math.exp(gate) if gate < 88.0 else math.exp(88.0)
        for i in range(Sv):
            Si = S[i]
            for j in range(Sv):
                Si[j] *= dec
        delta = [0.0] * Sv
        for j in range(Sv):
            s = 0.0
            for i in range(Sv):
                s += S[i][j] * k[i]
            delta[j] = (v[j] - s) * beta
        for i in range(Sv):
            Si = S[i]
            ki = k[i]
            for j in range(Sv):
                Si[j] += ki * delta[j]
        Iv = 1.0 / math.sqrt(Sv)
        out = [0.0] * Sv
        for j in range(Sv):
            s = 0.0
            for i in range(Sv):
                s += S[i][j] * q[i]
            out[j] = s * Iv
        return out

    # ---- helpers ----
    def small(self, name):
        if name in self.rd.cache:
            return self.rd.cache[name]
        if len(self.parser.tensors[name]["dims"]) != 1:
            raise KeyError("expected 1-D tensor " + name)
        return self.rd.read_row_full(name, None)

    def matvec(self, name, x):
        return self.rd.matvec(name, x)

    def rope(self, v, pos, dh):
        # NeoX pairing with offset n_rot/2 (ggml rotate_pairs, ops.cpp)
        n2 = self.n_rot // 2
        for j in range(n2):
            th = pos * self.rope_freq[j]
            c, s = math.cos(th), math.sin(th)
            a, b = v[j], v[j + n2]
            v[j] = a * c - b * s
            v[j + n2] = a * s + b * c
        return v

    # ---- full attention layer ----
    def attn_layer(self, il, h, pos):
        Dk, H, Hkv = self.Dk, self.H, self.Hkv
        qg = self.matvec("blk.%d.attn_q.weight" % il, h)
        nq = H * Dk
        Q, G = qg[:nq], qg[nq:]
        qn_w = self.small("blk.%d.attn_q_norm.weight" % il)
        kn_w = self.small("blk.%d.attn_k_norm.weight" % il)
        # per-head Q norm + rope
        Qh = []
        for hd in range(H):
            q = Q[hd * Dk:(hd + 1) * Dk]
            Qh.append(self.rope(rms_norm_vec(q, qn_w, self.eps), pos, Dk))
        K = self.matvec("blk.%d.attn_k.weight" % il, h)
        V = self.matvec("blk.%d.attn_v.weight" % il, h)
        Kh, Vh = [], []
        for hd in range(Hkv):
            k = K[hd * Dk:(hd + 1) * Dk]
            Kh.append(self.rope(rms_norm_vec(k, kn_w, self.eps), pos, Dk))
            Dv = self.Dv
            Vh.append(V[hd * Dv:(hd + 1) * Dv])
        if il not in self.kv:
            self.kv[il] = ([], [])
        self.kv[il][0].append(Kh)
        self.kv[il][1].append(Vh)
        nqkv = H // Hkv
        scale = 1.0 / math.sqrt(Dk)
        attn = [0.0] * (H * self.Dv)
        for hd in range(H):
            kh = hd // nqkv
            q = Qh[hd]
            scores = []
            for t in range(pos + 1):
                k = self.kv[il][0][t][kh]
                scores.append(sum(a * b for a, b in zip(q, k)) * scale)
            probs = softmax_vec(scores)
            acc = [0.0] * self.Dv
            for t in range(pos + 1):
                v = self.kv[il][1][t][kh]
                p = probs[t]
                for i in range(self.Dv):
                    acc[i] += p * v[i]
            g = G[hd * Dk:(hd + 1) * Dk]
            base = hd * self.Dv
            for i in range(self.Dv):
                attn[base + i] = acc[i] / (1.0 + math.exp(-g[i]))
        return self.matvec("blk.%d.attn_output.weight" % il, attn)

    # ---- gated DeltaNet SSM layer ----
    def ssm_layer(self, il, h):
        d_inner = self.ssm_inner
        d_state = self.ssm_state
        n_group = self.ssm_group
        n_vheads = self.ssm_rank
        d_conv = self.ssm_conv
        key_dim = d_state * n_group
        Sv = d_inner // n_vheads
        qkv = self.matvec("blk.%d.attn_qkv.weight" % il, h)
        z = self.matvec("blk.%d.attn_gate.weight" % il, h)
        q0, k0, v0 = qkv[:key_dim], qkv[key_dim:2 * key_dim], qkv[2 * key_dim:]
        beta_raw = self.matvec("blk.%d.ssm_beta.weight" % il, h)
        beta = [1.0 / (1.0 + math.exp(-beta_raw[j])) for j in range(n_vheads)]
        alpha = self.matvec("blk.%d.ssm_alpha.weight" % il, h)
        dt = self.small("blk.%d.ssm_dt.bias" % il)
        avec = self.small("blk.%d.ssm_a" % il)
        decay = []
        for j in range(n_vheads):
            sp = alpha[j] + (dt[j] if j < len(dt) else 0.0)
            sp = sp if sp < 88.0 else 88.0
            decay.append(math.log1p(math.exp(sp)) * (avec[j] if j < len(avec) else 1.0))
        # causal depthwise conv with state
        kn = self.rd.cache.get("blk.%d.ssm_conv1d.weight" % il)
        if kn is None:
            raise KeyError("blk.%d.ssm_conv1d.weight not cached (too big?)" % il)
        # conv channels == len(qkv); kernel rows laid out per channel
        if il not in self.ssm_c:
            self.ssm_c[il] = [[0.0] * (d_conv - 1) for _ in range(len(qkv))]
        wins = self.ssm_c[il]
        mixed = q0 + k0 + v0
        conv = [0.0] * len(mixed)
        for c in range(len(mixed)):
            win = wins[c] + [mixed[c]]
            conv[c] = self._conv_step(win, kn[c * d_conv:(c + 1) * d_conv])
            wins[c] = win[1:]
        q_c, k_c = conv[:key_dim], conv[key_dim:2 * key_dim]
        v_c = conv[2 * key_dim:]
        # split heads + L2 norm q/k
        qh = [l2_norm_vec(q_c[j * d_state:(j + 1) * d_state], self.eps) for j in range(n_group)]
        kh = [l2_norm_vec(k_c[j * d_state:(j + 1) * d_state], self.eps) for j in range(n_group)]
        vh = [v_c[j * Sv:(j + 1) * Sv] for j in range(n_vheads)]
        rep = n_vheads // n_group
        if il not in self.ssm_s:
            self.ssm_s[il] = [[[0.0] * Sv for _ in range(Sv)] for _ in range(n_vheads)]
        St = self.ssm_s[il]
        out = [0.0] * (Sv * n_vheads)
        for vh_i in range(n_vheads):
            kh_i = vh_i // rep if rep else 0
            out[vh_i * Sv:(vh_i + 1) * Sv] = self._delta_step(
                St[vh_i], qh[kh_i], kh[kh_i], vh[vh_i], decay[vh_i], beta[vh_i])
        # gated norm: rms(out_head, ssm_norm) * silu(z_head)
        nw = self.small("blk.%d.ssm_norm.weight" % il)
        gated = [0.0] * len(out)
        for vh_i in range(n_vheads):
            seg = out[vh_i * Sv:(vh_i + 1) * Sv]
            nr = rms_norm_vec(seg, nw, self.eps)
            zs = z[vh_i * Sv:(vh_i + 1) * Sv]
            for j in range(Sv):
                sv = zs[j]
                gated[vh_i * Sv + j] = nr[j] * (sv / (1.0 + math.exp(-sv if sv > -88.0 else -88.0)))
        return self.matvec("blk.%d.ssm_out.weight" % il, gated)

    def forward_pass(self, token_id, pos):
        x = self.rd.read_row_full("token_embd.weight", token_id)
        if len(x) != self.n_embd:
            raise ValueError("embed dim mismatch")
        for il in range(self.n_layer):
            h = rms_norm_vec(x, self.small("blk.%d.attn_norm.weight" % il), self.eps)
            if il in self.recr:
                y = self.ssm_layer(il, h)
            else:
                y = self.attn_layer(il, h, pos)
            x = [a + b for a, b in zip(x, y)]
            h2 = rms_norm_vec(x, self.small("blk.%d.attn_post_norm.weight" % il), self.eps)
            g = self.matvec("blk.%d.ffn_gate.weight" % il, h2)
            u = self.matvec("blk.%d.ffn_up.weight" % il, h2)
            act = [0.0] * len(g)
            for i in range(len(g)):
                gv = g[i]
                act[i] = (gv / (1.0 + math.exp(-gv if gv > -88.0 else -88.0))) * u[i]
            d = self.matvec("blk.%d.ffn_down.weight" % il, act)
            x = [a + b for a, b in zip(x, d)]
        x = rms_norm_vec(x, self.small("output_norm.weight"), self.eps)
        try:
            logits = self.matvec("output.weight", x)
        except KeyError:
            logits = self.matvec("token_embd.weight", x)
        return logits

    def close(self):
        self.mm.close()
        os.close(self.fd)


# =====================================================================
# PART 6: CHAT
# =====================================================================
def run_chat_interface():
    try:
        print("\n%s[System] Initializing Qwen3.5 Python Engine...%s" % (Color.CYAN, Color.RESET))
        if not os.path.exists(FULL_MODEL_PATH):
            print("%s[CRITICAL ERROR] Model not found at %s%s" % (Color.RED, FULL_MODEL_PATH, Color.RESET))
            return

        file_size_gb = os.path.getsize(FULL_MODEL_PATH) / (1024 * 1024 * 1024)
        print("%s[System] Mapped %.2f GB to virtual memory.%s" % (Color.CYAN, file_size_gb, Color.RESET))

        engine = Qwen35Engine(FULL_MODEL_PATH)
        tok = engine.tok
        print("%s[System] Engine Online -> Mmap: Active, BPE: Real, Anti-Think: Enabled%s" %
              (Color.GREEN, Color.RESET))
        print("%s====================================================%s" % (Color.CYAN, Color.RESET))
        print("               INTERACTIVE TERMINAL")
        print("         Type 'exit' or 'quit' to terminate.")
        print("%s====================================================\n%s" % (Color.CYAN, Color.RESET))

        pos = 0
        while True:
            user_input = input("%sYou: %s" % (Color.GREEN, Color.RESET))
            if user_input.strip().lower() in ["exit", "quit"]:
                print("%s[System] Safely unmapping memory and shutting down...%s" % (Color.YELLOW, Color.RESET))
                engine.close()
                break

            raw_prompt = user_input + "\n(Answer directly without internal thoughts.)\nAssistant: "
            input_tokens = tok.encode(raw_prompt, add_bos=(pos == 0))

            print("%sAssistant: %s" % (Color.CYAN, Color.RESET), end="", flush=True)

            t0 = time.time()
            for pt in input_tokens[:-1]:
                engine.forward_pass(pt, pos)
                pos += 1

            current_token = input_tokens[-1]
            token_count = 0

            for _ in range(N_PREDICT):
                logits = engine.forward_pass(current_token, pos)
                pos += 1
                # anti-think: block thinking tokens during generation
                for ti in tok.anti_think:
                    if ti < len(logits):
                        logits[ti] = -float("inf")
                best, best_v = 0, logits[0]
                for i in range(1, len(logits)):
                    if logits[i] > best_v:
                        best_v, best = logits[i], i
                current_token = best
                token_count += 1
                word = tok.decode([current_token])

                print(word, end="", flush=True)

                if current_token in tok.eog or "<|im_end|>" in word or "User:" in word:
                    break
                if token_count % 8 == 0:
                    gc.collect()

            dt = time.time() - t0
            print("\n%s[Tokens: %d | %.1fs | Response complete]%s\n" %
                  (Color.MAGENTA, token_count, dt, Color.RESET))

    except Exception:
        print("\n%s[FATAL CRASH] The engine encountered a raw error:%s" % (Color.RED, Color.RESET))
        traceback.print_exc()


if __name__ == "__main__":
    run_chat_interface()
