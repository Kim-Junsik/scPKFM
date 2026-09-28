# scPKFM 데이터 설명

> **이 문서의 목적.** scPKFM이 쓰는 데이터 파일의 위치, 내부 구조, 조건 구성, 분할, 전처리 결과물(캐시)을 정리했다. 모델·학습 설명은 별도 문서 "scPKFM 기술 브리프"에 있다.
>
> 수치는 모두 파일을 직접 읽어 확인했다. 작성일 2026-09-17.

## 1. 파일 위치

### 1.1 저장소 위치 (세 곳 모두 같은 상대 경로)

| 머신 | 저장소 루트 |
|---|---|
| 로컬 PC (Windows) | `F:\paper\ICLR_PJ\scPKFM\` |
| 서버 `kang` | `/mnt/data/dev/scPKFM/` |
| 도커 컨테이너 | `/workspace/dev/ICLR_PR/scPKFM/` |

`data/`, `assets/*.h5ad`, `assets/kegg/`는 모두 `.gitignore`에 들어 있다. **git pull로는 옮겨지지 않으므로 머신 간에 직접 복사해야 한다.** 캐시는 원본이 있으면 `data_prepare.py`나 `run.sh`가 다시 만들고, KEGG 파일은 `scripts/download_kegg.py`로 다시 받을 수 있다.

### 1.2 파일 목록 (저장소 루트 기준 상대 경로)

**원본 데이터**

| 경로 | 크기 | 내용 |
|---|---|---|
| `data/norman/norman.h5ad` | 2.2 GB | Norman 원본 (log1p 정규화된 X) |
| `data/norman/split_results.pkl` | 9 KB | scDFM이 배포한 Norman 5 fold 분할 |
| `data/combosciplex/combosciplex.h5ad` | 775 MB | ComboSciPlex 원본 (정규화된 X + raw counts 층) |
| `data/combosciplex/combosciplex_lognorm_counts_median.h5ad` | 962 MB | counts를 중앙값 라이브러리 크기로 재정규화한 사본. **코드가 처음 실행될 때 자동으로 만든다** |

**KEGG pathway 스냅샷** (git에 포함되지 않음, `scripts/download_kegg.py`로 받음)

| 경로 | 내용 |
|---|---|
| `assets/kegg/pathway_list.tsv` | 사람 pathway 목록 (372줄) |
| `assets/kegg/pathway_gene.tsv` | pathway–유전자 멤버십 (39,574줄) |
| `assets/kegg/gene_list.tsv` | KEGG 유전자 ID → 유전자 심볼 (24,252줄) |
| `assets/kegg/manifest.json` | 받은 날짜(2026-09-15 기록)와 파일 크기 |

**학습용 캐시** (`assets/`, 자동 생성)

| 파일 | 크기 | 용도 | 있는 곳 |
|---|---|---|---|
| `norman_scanpy5000_additive_fold{0..4}.h5ad` | 약 137 MB | Table 1 | 로컬·서버 |
| `norman_scanpy5000_combinations_fold{0..4}.h5ad` | 약 137 MB | Table 2 | 서버 (이 PC에는 없음) |
| `norman_scanpy5000_additive_nval_fold0.h5ad` | 137 MB | Norman 개발 검증 | 로컬·서버 |
| `combosciplex_scanpy5000_additive_scdfm7_fold0.h5ad` | 50 MB | Table 3 | 로컬·서버 |
| `combosciplex_scanpy5000_additive_scdfm7val{0,1,2}_fold0.h5ad` | 약 49 MB | ComboSciPlex 개발 검증 fold | 로컬은 val0만, 서버는 전부 |
| `combosciplex_scanpy5000_additive_scdfm7val_fold0.h5ad` | 49 MB | 이전 검증 분할 (조합 2개) | 로컬·서버 |
| `norman_scanpy5000_fold1.h5ad` | 137 MB | 이전 이름 규칙의 캐시, 현재 쓰지 않음 | 로컬 |

**scPKFM이 쓰지 않는 파일**

| 경로 | 내용 |
|---|---|
| `data/controls.zip` (662 MB) | Virtual Cell Challenge 2026 검증 패널의 대조군 세포. context A/B/C, 대상 유전자 300개, context당 non-targeting 대조군 18,400세포, 유전자 18,533개. 현재 코드에서 읽지 않는다 |

## 2. Norman

### 2.1 개요

- K562 세포에서 CRISPRa로 유전자 1개 또는 2개를 활성화한 Perturb-seq 데이터 (Norman et al., 2019)
- 세포 84,986개, 유전자 19,264개
- 조건 227개: 대조군 `ctrl` 1개 + 단일 101개 + 조합 125개

### 2.2 파일 구조 (`norman.h5ad`)

| 항목 | 내용 |
|---|---|
| `X` | CSR 희소 행렬 84,986 × 19,264. **이미 log1p 정규화된 값**(0이 아닌 값 0.27–6.41). raw counts 층은 없다 |
| `obs['condition']` | 조건 이름. 조합 `AHR+FEV`, 단일 `AHR+ctrl`, 대조군 `ctrl` |
| `obs['condition_name']` | `K562_AHR+FEV_1+1` 형식의 긴 이름 |
| `obs['pert_type']` | `0` 대조군(7,275) / `1+0` 단일(45,342) / `1+1` 조합(32,369) |
| `obs['control']` | 대조군이면 1 |
| `obs['guide_identity']` | 가이드 조합 277종 |
| `obs['gemgroup']` | 시퀀싱 배치 8개 (배치당 약 1만 세포) |
| `obs` 기타 | `cell_type`(K562), `dose_val`, `UMI_count`, `n_genes`, `read_count`, `coverage` 등 |
| `var['gene_name']` | 유전자 심볼 |
| `uns` | `rank_genes_groups_cov_all`, `top_non_dropout_de_20`, `top_non_zero_de_20` 등 DE 유전자 정보. scPKFM은 쓰지 않는다 |

### 2.3 조건별 세포 수

- 대조군 7,275개
- 조건당 최소 46개, 중앙값 282개

### 2.4 분할 파일 (`split_results.pkl`)

- 길이 5인 리스트이고, 원소마다 `{'train': [...], 'test': [...]}`다. **조합 이름만** 들어 있다.
- 모든 fold가 학습 조합 88개, 테스트 조합 37개다. 5 fold는 서로 독립인 70/30 무작위 추출이라 fold끼리 테스트가 겹친다.
- 단일은 파일에 없고, additive 분할에서는 항상 학습에 쓴다.

| 표 | 분할 규칙 | 학습 | 테스트 |
|---|---|---|---|
| Table 1 (additive) | pkl 그대로 | 단일 101 + 조합 88 | 조합 37 |
| Table 2 (combinations) | 각 fold 테스트 조합의 앞 15개 + 그 유전자들의 단일을 테스트로 (scDFM 코드와 동일, 불러올 때 계산) | 나머지 전부 | 조합 15 + 단일 약 23 |
| Norman 개발 검증 | additive fold 0의 학습 조합 중 15개를 검증으로. 원래 테스트 37개는 학습에서도 제외 | 단일 101 + 조합 73 | 검증 조합 15 |

Norman 검증 조합 15개: PLK4+STIL, POU3F2+FOXL2, RHOXF2+ZBTB25, TMSB4X+BAK1, CDKN1C+CDKN1B, POU3F2+CBFA2T3, PRDM1+CBFA2T3, RHOXF2+SET, BCL2L11+TGFBR2, MAP2K3+SLC38A2, TBX3+TBX2, BCL2L11+BAK1, FOXF1+FOXL2, FOXL2+MEIS1, ZNF318+FOXL2

- Table 2에서는 fold마다 유전자 2–4개가 학습 조건에 한 번도 나오지 않는다.

| fold | 학습에 없는 유전자 |
|---|---|
| 0 | HOXC13, IER5L, ISL2 |
| 1 | CLDN6, KIF18B, KIF2C, S1PR2 |
| 2 | IER5L, RUNX1T1 |
| 3 | CLDN6, IER5L, S1PR2 |
| 4 | COL2A1, ISL2 |

### 2.5 KEGG와의 관계

섭동 대상 유전자 101개 중 53개는 필터를 통과한 KEGG pathway에 속하지 않는다. HOX, FOX, DLX, LHX, POU3F2 등 발생 전사인자 계열이다.

## 3. ComboSciPlex

### 3.1 개요

- A549 세포에 약물 1개 또는 2개를 처리한 sci-Plex 기반 스크리닝 (`obs['sample']` 값은 `sciPlex_theis`)
- 세포 63,378개, 유전자 27,518개
- 약물 17개, 조건 32개: 대조군 `control+control` 1개 + 단일 6개 + 조합 25개

### 3.2 파일 구조 (`combosciplex.h5ad`)

| 항목 | 내용 |
|---|---|
| `X` | CSC 희소 행렬 63,378 × 27,518. 세포당 10,000 counts로 정규화한 뒤 log1p (0이 아닌 값 0.44–5.67) |
| `layers['counts']` | 같은 크기의 **정수 raw counts** (1–73) |
| `obs['condition']` | `DrugA+DrugB`, 단일은 `control+Drug`, 대조군은 `control+control` |
| `obs['Drug1']`, `obs['Drug2']` | 두 약물 이름 (단일이면 한쪽이 `control`) |
| `obs['pathway1']`, `obs['pathway2']` | 약물 분류 라벨 (3.3 표) |
| `obs['split']` | 원본에 딸린 세포 단위 분할 `train` 49,653 / `test` 5,516 / `ood` 8,209. **Table 3에는 쓰지 않는다** (아래 참고) |
| `obs` 기타 | `total_counts`(중앙값 2,579), `Size_Factor`, `n_genes`, `pct_counts_mt`, `leiden`, `Well`, `RT_well`, `cell_type`(A549) |
| `var` | `gene_short_name`, `id`, `highly_variable`(5,000개 표시), `dispersions` 등 |
| `obsm` | `X_pca`, `X_umap` |

`obs['split']`의 `ood` 조건 5개(Dacinostat+Danusertib, Givinostat+Cediranib, Panobinostat+Alvespimycin, Panobinostat+SRT2104, SRT2104+Alvespimycin)는 scDFM의 테스트 7개와 **SRT2104+Alvespimycin 하나만** 겹친다. 그래서 scDFM 코드의 7개 목록을 쓴다.

### 3.3 약물 17개

"데이터 분류"는 `obs['pathway1/2']`에 적힌 값을 그대로 옮긴 것이다. **약리학적으로 정확하지 않은 라벨이 섞여 있다.** 예를 들어 SRT 계열은 실제로는 SIRT1 활성제이고, Cediranib·Crizotinib·Dasatinib·Sorafenib는 EGFR 억제제가 아닌 다중 키나아제 억제제다.

Table 3 학습 조건 24개 안에서 각 약물이 몇 번 나오는지도 함께 적었다.

| 약물 | 데이터 분류 | 학습 단일 | 학습 조합 | Table 3 테스트에 등장 |
|---|---|---|---|---|
| Panobinostat | HDAC inhibitor | 1 | 5 | 조합 4개 |
| Givinostat | HDAC inhibitor | 1 | 9 | |
| Dacinostat | HDAC inhibitor | **0** | 3 | **단일** |
| PCI-34051 | HDAC inhibitor | 0 | 3 | |
| SRT2104 | Sirtuin inhibitor | 1 | 2 | 조합 1개 |
| SRT3025 | Sirtuin inhibitor | 0 | 2 | |
| SRT1720 | Sirtuin inhibitor | 0 | **1** | 조합 1개 |
| Curcumin | Sirtuin inhibitor | 0 | **1** | 조합 1개 |
| Dasatinib | EGFR inhibitor | 1 | 3 | |
| Cediranib | EGFR inhibitor | 0 | 3 | |
| Crizotinib | EGFR inhibitor | 0 | **1** | 조합 1개 |
| Sorafenib | EGFR inhibitor | 0 | **1** | 조합 1개 |
| Alvespimycin | Protein folding & Protein degradation | **0** | 2 | **단일**, 조합 1개 |
| Tanespimycin | Protein folding & Protein degradation | 0 | 1 | |
| Pirarubicin | DNA damage & DNA repair | 0 | 1 | |
| Carmofur | DNA damage & DNA repair | 0 | 1 | |
| Danusertib | Cell cycle regulation | 0 | 1 | |

- 17개 중 **13개는 학습에서 단일로 한 번도 나오지 않는다.**
- 10개는 학습 조건 1–2개에만 나온다.

### 3.4 조건별 세포 수와 역할

| 조건 | 세포 수 | 원본 `split` | Table 3 | 개발 검증 fold |
|---|---|---|---|---|
| control+control | 1,451 | train/test | 대조군 | |
| control+Dasatinib | 2,343 | train/test | 학습 | |
| control+Givinostat | 1,682 | train/test | 학습 | |
| control+Panobinostat | 1,578 | train/test | 학습 | |
| control+SRT2104 | 2,756 | train/test | 학습 | |
| **control+Alvespimycin** | 758 | train/test | **테스트** | |
| **control+Dacinostat** | 1,869 | train/test | **테스트** | |
| **Panobinostat+Crizotinib** | 1,641 | train/test | **테스트** | |
| **Panobinostat+Curcumin** | 2,244 | train/test | **테스트** | |
| **Panobinostat+SRT1720** | 1,826 | train/test | **테스트** | |
| **Panobinostat+Sorafenib** | 2,013 | train/test | **테스트** | |
| **SRT2104+Alvespimycin** | 520 | ood | **테스트** | |
| Alvespimycin+Pirarubicin | 476 | train/test | 학습 | |
| Cediranib+PCI-34051 | 2,161 | train/test | 학습 | cv2 |
| Dacinostat+Danusertib | 1,939 | ood | 학습 | |
| Dacinostat+Dasatinib | 1,231 | train/test | 학습 | cv1 |
| Dacinostat+PCI-34051 | 3,298 | train/test | 학습 | cv2 |
| Givinostat+Carmofur | 2,692 | train/test | 학습 | |
| Givinostat+Cediranib | 2,783 | ood | 학습 | cv1 |
| Givinostat+Crizotinib | 2,662 | train/test | 학습 | |
| Givinostat+Curcumin | 2,736 | train/test | 학습 | |
| Givinostat+Dasatinib | 2,421 | train/test | 학습 | cv0 |
| Givinostat+SRT1720 | 2,260 | train/test | 학습 | |
| Givinostat+SRT2104 | 2,353 | train/test | 학습 | cv2 |
| Givinostat+Sorafenib | 2,734 | train/test | 학습 | |
| Givinostat+Tanespimycin | 1,310 | train/test | 학습 | |
| Panobinostat+Alvespimycin | 996 | ood | 학습 | cv0 |
| Panobinostat+Dasatinib | 1,955 | train/test | 학습 | cv2 |
| Panobinostat+PCI-34051 | 1,814 | train/test | 학습 | cv1 |
| Panobinostat+SRT2104 | 1,971 | ood | 학습 | cv0 |
| Panobinostat+SRT3025 | 1,889 | train/test | 학습 | cv1 |
| SRT3025+Cediranib | 3,016 | train/test | 학습 | cv0 |

- 개발 검증 fold(cv0–cv2)에서는 그 fold의 조합 4개를 학습에서 빼고 채점한다. Table 3 테스트 7개는 학습에서 계속 제외한다.
- **검증 fold에 단일 조건이 없다.** Table 3에서 오차가 가장 큰 단일 블록의 구조가 검증에 반영되지 않는다.

### 3.5 정규화

- 배포된 `X`는 세포당 10,000 counts 기준이고, scDFM은 counts를 **중앙값 라이브러리 크기(2,579)** 기준으로 다시 정규화한다.
- scPKFM도 같은 방식을 쓴다. `layers['counts']` → `sc.pp.normalize_total()`(target 생략 = 중앙값) → `sc.pp.log1p()`. 중앙값은 전체 세포로 계산한다.
- 결과는 `combosciplex_lognorm_counts_median.h5ad`로 저장된다(0이 아닌 값 0.15–5.95).
- 이 차이가 L2 척도를 크게 바꾼다. scDFM 테스트 7개에서 Control L2는 배포된 X로 8.25지만, 재정규화하면 5.33–5.36으로 논문 값(5.37)과 맞는다.

## 4. 학습용 캐시

### 4.1 만드는 과정

`run.sh`가 캐시가 없으면 `data_prepare.py`를 호출한다. 이 스크립트는 KEGG 스냅샷 확인, 분할 검증, 캐시 생성(`scripts/build_data.py`), 베이스라인 계산 순서로 진행한다.

파일 이름 규칙: `assets/{데이터셋}_scanpy{HVG 수}_{method}{split 태그}_fold{fold}.h5ad`

| split 태그 | 의미 |
|---|---|
| (없음) | Norman 기본 분할 |
| `_nval` | Norman 개발 검증 |
| `_scdfm7` | ComboSciPlex, scDFM 테스트 7개 |
| `_scdfm7val{0,1,2}` | ComboSciPlex 개발 검증 fold |
| `_scdfm7val` | ComboSciPlex 이전 검증 분할 (조합 2개) |

### 4.2 내용

| 항목 | 내용 |
|---|---|
| `X` | CSR, **모든 세포**(held-out 포함) × 선택된 유전자 |
| `obs` | `condition`, `cell_type` (ComboSciPlex는 `split`도) |
| `var` 인덱스 | 유전자 이름 |
| `uns['build_config']` | 캐시를 만든 설정 전체 (JSON) |
| `uns['gene_selection']` | 선택 유전자 수, 강제 포함한 대상 유전자 수 |
| `uns['perturbation_targets']` | 섭동 이름 목록 (Norman 유전자 101개, ComboSciPlex 약물 17개) |

### 4.3 유전자 선택

- scanpy `highly_variable_genes`(seurat flavour)로 5,000개를 고른다.
- **held-out 조건의 세포는 선택에서 뺀다.** 그래서 캐시가 fold마다 다르다.
- Norman은 섭동 대상 유전자를 강제로 포함해 fold마다 5,028–5,032개가 된다. ComboSciPlex는 약물이 유전자가 아니라서 5,000개 그대로다.

| 캐시 | 유전자 수 | 강제 포함 |
|---|---|---|
| Norman additive fold 0 / 1 / 2 / 3 / 4 | 5,030 / 5,032 / 5,032 / 5,030 / 5,029 | 30 / 32 / 32 / 30 / 29 |
| Norman 개발 검증 | 5,028 | 28 |
| ComboSciPlex (Table 3, 검증) | 5,000 | 0 |

### 4.4 held-out 세포 처리

캐시에는 held-out 세포도 들어 있지만, 다음 세 곳에서는 쓰지 않는다.
1. HVG 선택 (캐시를 만들 때 제외)
2. stage 1 인코더 학습 (학습 시 세포 목록으로 제외)
3. 잠재 표준화 통계 (같은 방식으로 제외)

held-out 세포는 **평가할 때만** 읽는다.

## 5. KEGG pathway prior

- `pathway_gene.tsv`의 멤버십을 모델 유전자 공간에 맞춰 `M_prior ∈ {0,1}^{K_p × G}`를 만든다.
- 필터: 질병 pathway(`hsa05*`) 제외, 모델 유전자 공간에서 크기 10–300인 pathway만 남긴다.
- prior가 없는 자유 토큰 101개를 더해 토큰 수 K를 정한다. **Norman은 fold마다 K ≈ 311–316, ComboSciPlex는 K = 258.**
- 유전자 심볼은 `gene_list.tsv`의 대표 심볼로 먼저 맞추고, 없으면 별칭으로 맞춘다.

## 6. 평가에서 데이터를 쓰는 방식

- **조건 평균:** 조건의 전체 세포로 유전자별 평균 발현을 계산한다.
- **평가 유전자:** 테스트 조건과 대조군 세포만으로 scanpy HVG 1,000개를 다시 고른다(scDFM 프로토콜). L2와 cell-eval 모두 이 유전자에서 계산한다.
- **척도 확인:** 대조군 평균을 예측으로 쓴 Control L2가 논문과 맞는지 확인했다. Norman Table 1 fold 1은 3.89 대 3.99(2.5%), ComboSciPlex는 5.36 대 5.37(0.2%)이다.
- **예측 입력:** 조건마다 대조군 세포 1,024개를 무작위로 뽑아 수송한다.
