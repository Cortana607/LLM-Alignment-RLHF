# LLM-Alignment-RLHF
A complete pipeline for aligning Small Language Models using SFT, PPO, DPO, and GRPO.

This project implements a complete end-to-end alignment pipeline for Large Language Models (LLMs). Using **SmolLM2-1.7B** as a base, I implemented and compared multiple state-of-the-art alignment algorithms to improve model safety and reasoning capabilities.

## 🚀 Key Features
- **Supervised Fine-Tuning (SFT):** Initial instruction-tuning using masked cross-entropy loss.
- **Reward Modeling (RM):** Trained a Llama-3.2-1B judge using Margin Ranking Loss.
- **PPO Implementation:** Online Reinforcement Learning with Generalized Advantage Estimation (GAE).
- **Direct Preference Optimization (DPO):** Offline alignment achieving the highest performance gain.
- **GRPO (Group Relative Policy Optimization):** DeepSeek-style group-based alignment without a critic model.
- **RLVR (Verifiable Rewards):** Teaching mathematical reasoning using GSM8K and binary verifiable feedback.

## 📊 Final Results
The models were evaluated on a held-out validation set for "Harmlessness" using the trained Reward Model as a judge.

| Method | Avg Reward Score | Gain vs SFT |
| :--- | :--- | :--- |
| **SFT (Baseline)** | 0.2329 | 0.0000 |
| **PPO** | -0.0579 | -0.2908 |
| **DPO** | **0.6840** | **+0.4511** |
| **GRPO** | 0.1811 | -0.0518 |
| **RLVR (Math)** | 0.1385 | -0.0944 |

*Note: DPO emerged as the most stable and effective method for preference alignment in this hardware-constrained environment.*

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, Hugging Face Transformers
- **Optimization:** PEFT (LoRA), BitsAndBytes, Accelerate
- **Hardware:** Trained on a single NVIDIA T4 GPU (15GB VRAM) using bfloat16 precision.

## 📝 Qualitative Example
**Prompt:** "Can I write answers on my hand?"
- **SFT Response:** "No... you can't write on your own body."
- **DPO Response:** "No... you can't even write on your shirt." (Shows more restrictive/safe behavior).
