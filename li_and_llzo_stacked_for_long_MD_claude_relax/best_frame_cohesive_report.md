# Best-Frame Cohesive Energy Report

Model: `mace_fps_split17_SaveIT0_256_candidate3.model`

For each structure the trajectory frame with the **lowest fmax** (maximum atomic force magnitude) was selected, saved as the relaxed CIF, and used for cohesive energy analysis.

```
E_coh/atom = (E_relaxed − Σ nᵢ·E_atom_i) / N
```

**Atomic references:**
- E_atom[La] = -958.077393 eV
- E_atom[Li] = -190.759033 eV
- E_atom[O] = -442.988892 eV
- E_atom[Zr] = -1380.181763 eV

| Rank | Structure | N | Best Frame | fmax (eV/Å) | E_relaxed (eV) | E_ref (eV) | E_coh (eV) | E_coh/atom (eV/atom) |
|------|-----------|---|------------|-------------|----------------|------------|------------|----------------------|
| 1 | Li_111_slab__longMDstack_v2__LLZO_110_Zr | 3030 | 1849 | 0.096712 | -1334699.000000 | -1317925.519043 | -16773.480957 | -5.53580230 |
| 2 | Li_111_slab__longMDstack_v2__LLZO_011_La_code71_sto | 3030 | 841 | 0.202989 | -1334685.500000 | -1317925.519043 | -16759.980957 | -5.53134685 |
| 3 | Li_111_slab__longMDstack_v2__LLZO_001_Zr_code93_sto | 3030 | 1342 | 0.150762 | -1334560.375000 | -1317925.519043 | -16634.855957 | -5.49005147 |
| 4 | Li_111_slab__longMDstack_v2__LLZO_110_Li_order17_off | 2886 | 1365 | 0.179468 | -1247894.750000 | -1232216.302246 | -15678.447754 | -5.43258758 |
| 5 | Li_111_slab__longMDstack_v2__LLZO_010_La_order0_off | 2934 | 1288 | 0.090442 | -1261215.000000 | -1245408.413574 | -15806.586426 | -5.38738460 |
| 6 | Li_110_slab__longMDstack_v2__LLZO_011_La_code71_sto | 2710 | 1992 | 0.047811 | -1085471.375000 | -1071901.216309 | -13570.158691 | -5.00743863 |
| 7 | Li_110_slab__longMDstack_v2__LLZO_110_Zr | 2710 | 672 | 0.057710 | -1085420.375000 | -1071901.216309 | -13519.158691 | -4.98861944 |
| 8 | Li_110_slab__longMDstack_v2__LLZO_001_Zr_code93_sto | 2710 | 931 | 0.048637 | -1085398.875000 | -1071901.216309 | -13497.658691 | -4.98068586 |
| 9 | Li_110_slab__longMDstack_v2__LLZO_110_Li_order17_off | 2602 | 858 | 0.048560 | -1020326.250000 | -1007619.303711 | -12706.946289 | -4.88353047 |
| 10 | Li_110_slab__longMDstack_v2__LLZO_010_La_order0_off | 2638 | 401 | 0.039763 | -1030369.812500 | -1017513.387207 | -12856.425293 | -4.87355015 |
| 11 | Li_100_slab__longMDstack_v2__LLZO_011_La_code71_sto | 1652 | 750 | 0.046487 | -599915.437500 | -592606.041016 | -7309.396484 | -4.42457414 |
| 12 | Li_100_slab__longMDstack_v2__LLZO_110_Zr | 1652 | 460 | 0.049328 | -599908.687500 | -592606.041016 | -7302.646484 | -4.42048819 |
| 13 | Li_100_slab__longMDstack_v2__LLZO_010_La_order0_off | 1622 | 202 | 0.078708 | -573528.750000 | -566556.680664 | -6972.069336 | -4.29843979 |
| 14 | Li_100_slab__longMDstack_v2__LLZO_110_Li_order17_off | 1604 | 220 | 0.049333 | -568502.875000 | -561609.638916 | -6893.236084 | -4.29752873 |

**★ Most stable:** Li_111_slab__longMDstack_v2__LLZO_110_Zr  
fmax = 0.096712 eV/Å  
E_coh/atom = -5.53580230 eV/atom  
E_coh = -16773.480957 eV
