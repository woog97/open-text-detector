# Final runs vs T2.1 / T2 / SPG

Recall at the in-domain threshold (1% FPR on sweep-eval dev human sentences); wide1 cells are recall / AUROC.

| Metric | final-eg2 | final-qwen35-4b | final-moe | T21 (3-seed mean) | T2 (3-seed mean) | SPG (3-seed mean) |
|---|---|---|---|---|---|---|
| Sweep test: small-edit recall @1% FPR | 0.340 | 0.507 | 0.540 | 0.449 | 0.476 | 0.629 |
| Sweep test: small-edit AUROC | 0.952 | 0.974 | 0.975 | 0.969 | 0.967 | 0.982 |
| Sweep dev: small-edit recall @1% FPR | 0.190 | 0.354 | 0.418 | 0.300 | 0.303 | 0.478 |
| Sweep test: paragraph recall @1% FPR | 0.696 | 0.864 | 0.904 | 0.874 | 0.874 | 0.925 |
| heldout-writers: small-edit recall (in-domain thr) | 0.329 | 0.611 | 0.699 | 0.550 | 0.602 | 0.465 |
| heldout-writers: all AI recall (in-domain thr) | 0.329 | 0.611 | 0.699 | 0.550 | 0.602 | 0.465 |
| heldout-writers: AUROC vs controls | 0.923 | 0.970 | 0.972 | 0.953 | 0.954 | 0.927 |
| heldout-writers: control FPR | 0.018 | 0.012 | 0.018 | 0.022 | 0.023 | 0.030 |
| t21-heldout-v2: small-edit recall (in-domain thr) | 0.305 | 0.720 | 0.793 | 0.500 | 0.569 | 0.271 |
| t21-heldout-v2: all AI recall (in-domain thr) | 0.700 | 0.857 | 0.904 | 0.826 | 0.867 | 0.303 |
| t21-heldout-v2: AUROC vs controls | 0.972 | 0.991 | 0.993 | 0.984 | 0.985 | 0.844 |
| t21-heldout-v2: control FPR | 0.010 | 0.009 | 0.012 | 0.011 | 0.012 | 0.012 |
| cross-model-s500: small-edit recall (in-domain thr) | — | — | — | — | — | — |
| cross-model-s500: all AI recall (in-domain thr) | 0.280 | 0.534 | 0.931 | 0.862 | 0.826 | 0.414 |
| cross-model-s500: AUROC vs controls | 0.804 | 0.970 | 0.998 | 0.989 | 0.978 | 0.895 |
| cross-model-s500: control FPR | 0.013 | 0.001 | 0.002 | 0.006 | 0.006 | 0.003 |
| fable-polish: small-edit recall (in-domain thr) | — | — | — | — | — | — |
| fable-polish: all AI recall (in-domain thr) | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |
| fable-polish: AUROC vs controls | — | — | — | — | — | — |
| fable-polish: control FPR | 0.004 | 0.013 | 0.019 | 0.007 | 0.008 | 0.007 |
| fable-humanizer: small-edit recall (in-domain thr) | — | — | — | — | — | — |
| fable-humanizer: all AI recall (in-domain thr) | 0.730 | 0.827 | 0.879 | 0.866 | 0.904 | 0.203 |
| fable-humanizer: AUROC vs controls | 0.975 | 0.997 | 0.996 | 0.990 | 0.989 | 0.821 |
| fable-humanizer: control FPR | 0.025 | 0.001 | 0.007 | 0.013 | 0.017 | 0.011 |
| fable-cowrite: small-edit recall (in-domain thr) | — | — | — | — | — | — |
| fable-cowrite: all AI recall (in-domain thr) | 0.083 | 0.224 | 0.268 | 0.141 | 0.165 | 0.049 |
| fable-cowrite: AUROC vs controls | 0.687 | 0.764 | 0.795 | 0.815 | 0.808 | 0.633 |
| fable-cowrite: control FPR | 0.004 | 0.005 | 0.008 | 0.005 | 0.004 | 0.005 |
| wide1-heldout: small-edit recall (in-domain thr) | 0.379 | 0.664 | 0.751 | 0.573 | — | — |
| wide1-heldout: all AI recall (in-domain thr) | 0.587 | 0.773 | 0.816 | 0.713 | — | — |
| wide1-heldout: AUROC vs controls | 0.937 | 0.977 | 0.976 | 0.960 | — | — |
| wide1-heldout: control FPR | 0.020 | 0.015 | 0.018 | 0.023 | — | — |

**wide1 held-out by writer (selected checkpoints):**

| Writer group | final-eg2 | final-qwen35-4b | final-moe |
|---|---|---|---|
| all | 0.587 / 0.937 | 0.773 / 0.977 | 0.816 / 0.976 |
| para | 0.756 / 0.963 | 0.861 / 0.983 | 0.869 / 0.983 |
| small | 0.379 / 0.906 | 0.664 / 0.969 | 0.751 / 0.968 |
| w:deepseek/deepseek-v4.1-flash | 0.611 / 0.943 | 0.812 / 0.982 | 0.842 / 0.982 |
| w:xiaomi/mimo-v2.6-flash | 0.570 / 0.932 | 0.717 / 0.967 | 0.761 / 0.964 |
| w:z-ai/glm-5.3-flash | 0.556 / 0.932 | 0.752 / 0.977 | 0.822 / 0.979 |
