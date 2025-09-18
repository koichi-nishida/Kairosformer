# Kairosformer v0

Kairosformer is a **plugin-style Transformer architecture for time-series forecasting**.  
It builds on insights from state-of-the-art models such as **Informer**, **Autoformer**, **PatchTST**, **iTransformer**, and **DTSFormer**, while aiming to overcome their trade-offs between **accuracy** and **efficiency**.

---

## Motivation

Time-series forecasting is essential in domains such as **energy management, finance, weather prediction, and healthcare**.  
Recent Transformer variants each excel in one dimension but struggle in another:

- **Informer** → Efficient (O(L log L)) with ProbSparse attention, but sometimes less accurate for long-horizon dependencies.  
- **Autoformer** → High accuracy with decomposition + autocorrelation, but slower due to heavier constant factors.  
- **PatchTST** → Patching boosts both performance and efficiency, but lacks strong multivariate correlation modeling.  
- **iTransformer** → Inverted attention learns variate-wise correlations effectively, but less tuned for temporal scale adaptivity.  
- **DTSFormer** → Integrates spatial + temporal diffusion, but increases complexity with cross-graph operations.

**Kairosformer’s goal is to unify these complementary strengths into a modular, plugin-style Transformer backbone.**

---

## Main Goal

**To design a Transformer architecture that achieves both the efficiency of Informer and the accuracy of Autoformer by introducing a flexible, plugin-based mechanism that can integrate decomposition, sparse attention, patching, and variate-wise attention as interchangeable modules.**

---

## Hypothesis

1. **Fusion improves trade-off**: Combining efficient attention (e.g., ProbSparse/patching) with decomposition (trend + seasonal separation) will yield both low computational cost and higher predictive accuracy.  
2. **Plugin-style modularity**: Designing the model as a collection of interchangeable modules (e.g., attention, decomposition, autocorrelation, patch embedding) allows task-specific adaptation without re-engineering the backbone.  
3. **Generalization**: Such modularity enables the model to generalize better across domains (finance, traffic, energy) by swapping or tuning only relevant plugins.  

---

## Architecture (Kairosformer v0)

- **Backbone**: Transformer encoder–decoder.  
- **Plugin Modules**:
  - **Attention Layer (swap)**: ProbSparse, full, linear, or diffusion-based attention.  
  - **Decomposition Layer (optional)**: Seasonal-trend separation like Autoformer.  
  - **Autocorrelation Plugin**: Sub-series similarity discovery for periodic signals.  
  - **Patch Plugin**: PatchTST-style local subseries tokens for efficiency.  
  - **Variate Plugin**: iTransformer-style inverted attention for multivariate signals.  

---

## Current Status (v0)

- Implemented core training/evaluation loop (based on Autoformer codebase).  
- Verified on **ETTh1** benchmark: Kairosformer v0 already shows **higher accuracy than Informer** and **better efficiency than Autoformer**.  
- Modular hooks for swapping plugins are in place.  

---

## Next Steps

- Conduct **systematic ablations**: test each plugin module independently.  
- Benchmark on ETTh2, ETTm1, ETTm2, Traffic, Electricity, Weather.  
- Explore **self-supervised pretraining** (PatchTST-style) for transferability.  
- Optimize runtime profiling for real-time deployment.  

---

## References

- Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*
- Wu et al., *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*
- Nie et al., *PatchTST: A Time Series is Worth 64 Words*
- Liu et al., *iTransformer: Inverted Transformers are Effective for Time Series Forecasting*
- Zhu et al., *DTSFormer: Decoupled Temporal-Spatial Diffusion Transformer*