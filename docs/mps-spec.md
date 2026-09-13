# PipePye — MPS Ingestion Specification & Model Mapping

## 1. Overview and Scope

This document specifies the parsing rules, edge cases, and architectural mapping from external **MPS (Mathematical Programming System)** files into PipePye's internal representation.

PipePye's model ingestion architecture strictly decouples the internal optimization data structure from the input format:

```text
    MPS File (Fixed / Free)
               │
               ▼
       Streaming Parser
               │
               ▼
┌─────────────────────────────┐
│    General Bounded Form     │
│   l_r <= A x <= u_r         │
│   l_c <= x <= u_c           │
│   min c^T x + c_0           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Decoupled LinearProgram    │
│  - CSC Matrix (Simplex)     │
│  - CSR Matrix (PDHG / GPU)  │
│  - String Symbol Tables     │
└─────────────────────────────┘
```

This ensures future ingestion formats (e.g. LP files, Python/C++ APIs, FlatZinc) map directly to the same clean mathematical substrate without solver modifications.

---

## 2. Fixed Format vs. Free Format MPS

### 2.1 The Fixed Format (IBM Punched Card Standard)
Historically defined on 80-column punch cards, fields are strictly demarcated by character offsets:

```text
1         2         3         4         5         6         7         8
1234567890123456789012345678901234567890123456789012345678901234567890
------------------------------------------------------------------------
ROWS
 N  COST
 G  R01
 E  R02
COLUMNS
    X01       COST      3.00000   R01       1.00000
    X01       R02       2.00000
    X02       COST      4.00000   R01       2.00000
RHS
    RHS1      R01       6.00000   R02       8.00000
BOUNDS
 UP BND1      X01       5.00000
ENDATA
```

| Field | Columns | Content | Notes |
|---|---|---|---|
| **Header** | 1–4 | Section Keyword (`ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, `ENDATA`) | Left-aligned, no leading whitespace |
| **Field 1** | 2–3 | Row type indicator (`N`, `G`, `L`, `E`) or Bound type (`LO`, `UP`, `FX`, `FR`, `MI`, `PL`, `BV`, `UI`) | Blank in `COLUMNS` and `RHS` |
| **Field 2** | 5–12 | Column name, RHS name, Range name, or Bound name | Max 8 alphanumeric characters |
| **Field 3** | 15–22 | Row name (or Column name in `BOUNDS`) | Max 8 characters |
| **Field 4** | 25–36 | Numerical value (IEEE floating-point representation) | Right-justified or standard float |
| **Field 5** | 40–47 | Optional: second row name | Max 8 characters |
| **Field 6** | 50–61 | Optional: second numerical value | Associated with Field 5 |

### 2.2 The Free Format (Modern Standard)
Modern industrial solvers (HiGHS, Gurobi, CPLEX) emit and parse Free Format MPS:
- **Whitespace Delimited**: Fields are separated by one or more spaces or tabs; character column positions are ignored.
- **Arbitrary Identifier Length**: Variable and constraint names can exceed 8 characters (e.g. `crude_tower_feed_rate_period_3`).
- **Quoted Identifiers**: Identifiers containing spaces must be enclosed in quotes (`"gas oil blend"`).
- **Comments**: Any line starting with an asterisk `*` in column 1 is ignored. Modern extensions also tolerate blank lines.
- **Case Sensitivity**: Section keywords are case-insensitive (`rows` = `ROWS`), but identifier names are case-preserving.

### 2.3 Auto-Detection Strategy
PipePye's parser implements an automatic format detector:
1. Scan non-comment lines. If line length exceeds 61 columns, or any token is not aligned with fixed column offsets (cols 1, 4, 14, 24, 39, 49), or identifiers contain more than 8 contiguous characters before whitespace, the parser switches to **Free Format**.
2. If column boundaries match and tokens align with the 8-character field grid, the parser uses **Fixed Format**.

---

## 3. Detailed Section Semantics & Edge Cases

### 3.1 `NAME` and `OBJSENSE`
- `NAME <instance_name>`: Declares problem identifier.
- `OBJSENSE <MIN|MAX>`: Modern extension specifying optimization direction.
  - **Standard Default**: If `OBJSENSE` is absent, standard MPS mandates **MINIMIZE**.
  - **Normalization**: If `MAXIMIZE` is detected, PipePye flips the signs of the objective vector ($c \leftarrow -c$) during ingestion and records `is_maximization = true`.

### 3.2 `ROWS`
Declares row identifiers and constraint sense:
- `N`: Non-constraining row.
  - The **first** `N` row encountered is treated as the **Objective function** (unless overridden by an explicit `OBJNAME` card).
  - Subsequent `N` rows are ignored or treated as non-binding diagnostic metrics.
- `E`: Equality constraint ($a_i^T x = b_i$).
- `L`: Less-than or equal constraint ($a_i^T x \le b_i$).
- `G`: Greater-than or equal constraint ($a_i^T x \ge b_i$).

### 3.3 `COLUMNS`
Specifies non-zero elements $A_{ij}$ and objective coefficients $c_j$.
- **Column Grouping**: Standard MPS groups all coefficients for column $j$ contiguously.
- **Robustness Policy**: PipePye's streaming parser accumulates entries into coordinate (COO) triplets `(row_idx, col_idx, value)` and performs a stable radix/bucket sort, tolerating **out-of-order/interleaved column entries** that occur in corrupt or generator-produced files.
- **Integer Markers**: Lines containing `'MARKER'` with `'INTORG'` switch subsequent variables to integer type ($x_j \in \mathbb{Z}$), while `'INTEND'` switches back to continuous variables.

### 3.4 `RHS`
Specifies constraint vector $b$.
- Follows the same two-pair layout as `COLUMNS`.
- If multiple distinct RHS vector names appear in the file, PipePye defaults to selecting the **first encountered RHS set**, logging a warning for subsequent sets.
- Any row in `ROWS` that never appears in `RHS` has an implicit default RHS of `b_i = 0.0`.

### 3.5 `RANGES`
Allows constraints to be bounded on both sides ($l_i \le a_i^T x \le u_i$) without creating separate rows.
Given row sense, RHS value $b_i$, and range entry $r_i$:

| Row Type | $r_i$ Sign | Effective Lower Bound ($l_{r,i}$) | Effective Upper Bound ($u_{r,i}$) |
|---|---|---|---|
| **`G`** | $r_i > 0$ | $b_i$ | $b_i + |r_i|$ |
| **`L`** | $r_i > 0$ | $b_i - |r_i|$ | $b_i$ |
| **`E`** | $r_i > 0$ | $b_i$ | $b_i + |r_i|$ |
| **`E`** | $r_i < 0$ | $b_i - |r_i|$ | $b_i$ |

### 3.6 `BOUNDS`
Specifies variable bounds $l_{c,j} \le x_j \le u_{c,j}$:

| Bound Type | Meaning | Lower Bound ($l_j$) | Upper Bound ($u_j$) |
|---|---|---|---|
| *(Omitted)* | **Default Non-negative** | $0.0$ | $+\infty$ |
| **`LO`** | Lower bound | $v$ | Preserves upper bound ($+\infty$ if unset) |
| **`UP`** | Upper bound | $0.0$ (standard default) | $v$ |
| **`FX`** | Fixed variable | $v$ | $v$ |
| **`FR`** | Free variable | $-\infty$ | $+\infty$ |
| **`MI`** | Negative infinity | $-\infty$ | Preserves upper bound ($0.0$ if unset) |
| **`PL`** | Positive infinity | Preserves lower bound ($0.0$ if unset) | $+\infty$ |
| **`BV`** | Binary variable | $0.0$ | $1.0$ (Integer) |
| **`UI`** | Integer upper bound | $0.0$ | $v$ (Integer) |
| **`LI`** | Integer lower bound | $v$ | $+\infty$ (Integer) |

---

## 4. Planned Mapping: MPS $\to$ Internal Representation

### 4.1 Internal Mathematical Formulation
PipePye represents all continuous problems in **General Bounded Form**:

$$\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & c^T x + c_0 \\
\text{subject to} \quad & l_r \le A x \le u_r \quad & (m \text{ constraints}) \\
& l_c \le x \le u_c \quad & (n \text{ variables})
\end{aligned}$$

Where:
- Row bounds $l_r, u_r \in [-\infty, +\infty]^m$ directly represent equality, inequality, and ranged constraints without introducing artificial slack variables.
- Column bounds $l_c, u_c \in [-\infty, +\infty]^n$ represent variable limits.
- Objective $c \in \mathbb{R}^n$, scalar offset $c_0 \in \mathbb{R}$.

### 4.2 C++20 Data Model (`include/pipepye/model/lp_model.hpp`)

```cpp
namespace pipepye::model {

enum class VariableType : uint8_t {
    Continuous = 0,
    Binary,
    Integer
};

struct LinearProgram {
    std::string name;
    bool is_maximization{false};
    scalar_t obj_offset{0.0};

    // Symbol Tables (bi-directional mapping: index <-> name)
    std::vector<std::string> col_names;
    std::vector<std::string> row_names;
    std::string obj_name;

    // Fast O(1) identifier lookup via string intern pool
    std::unordered_map<std::string, index_t> col_name_to_idx;
    std::unordered_map<std::string, index_t> row_name_to_idx;

    // Objective vector c
    std::vector<scalar_t> c;

    // Variable Bounds: l_c <= x <= u_c
    std::vector<scalar_t> col_lower;
    std::vector<scalar_t> col_upper;
    std::vector<VariableType> var_types;

    // Constraint Bounds: l_r <= A x <= u_r
    std::vector<scalar_t> row_lower;
    std::vector<scalar_t> row_upper;

    // Dual Sparse Representations of Constraint Matrix A
    // 1. Column-Compressed Sparse (CSC) for Simplex & Column Pricing
    std::vector<index_t> csc_col_ptr; // size: n + 1
    std::vector<index_t> csc_row_ind; // size: nnz
    std::vector<scalar_t> csc_values; // size: nnz

    // 2. Row-Compressed Sparse (CSR) for GPU SpMV, Presolve, and KKT verification
    std::vector<index_t> csr_row_ptr; // size: m + 1
    std::vector<index_t> csr_col_ind; // size: nnz
    std::vector<scalar_t> csr_values; // size: nnz

    [[nodiscard]] index_t num_cols() const noexcept { return static_cast<index_t>(col_names.size()); }
    [[nodiscard]] index_t num_rows() const noexcept { return static_cast<index_t>(row_names.size()); }
    [[nodiscard]] size_t num_nonzeros() const noexcept { return csc_values.size(); }
};

} // namespace pipepye::model
```

### 4.3 Ingestion Pipeline Flow

```text
Stream MPS File
      │
      ▼
Pass 1: Tokenize Header, Scan ROWS & RHS
      ├── Register row names in hash map
      ├── Categorize Row senses: L -> [-inf, b], G -> [b, +inf], E -> [b, b]
      └── Apply RANGES offsets to row_lower and row_upper
      │
      ▼
Pass 2: Parse COLUMNS & BOUNDS
      ├── Register column names in hash map
      ├── Parse coefficients: route objective row entries into vector c
      ├── Route constraint entries into COO triplet buffer: (row_idx, col_idx, value)
      └── Parse BOUNDS: overwrite default [0, +inf) bounds with explicit values
      │
      ▼
Pass 3: Matrix Assembly
      ├── Radix/Bucket sort COO buffer by col_idx, then row_idx
      ├── Build CSC representation in O(NNZ) time
      ├── Build CSR transpose in O(NNZ) time
      └── Normalize objective (if MAXIMIZE: c = -c, offset = -offset)
      │
      ▼
Pristine LPModel Ready for Presolve, Scaling & Verification
```
