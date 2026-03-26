# Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor

Official implementation of the paper: **"Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor"**.
Accepted by **ICASSP 2026**.

📄 **Paper (arXiv):** [https://arxiv.org/abs/2510.07909](https://arxiv.org/abs/2510.07909)

## 📌 Overview

Bloodroot is a novel **Watermark-as-Trigger** framework that repurposes audio watermarking as a stealthy and robust backdoor trigger for speech systems.

* **Bloodroot-FT:** Uses AudioSeal to embed watermark triggers without fine-tuning.


* **Bloodroot:** Utilizes **Adversarial LoRA fine-tuning** to enhance the trigger's perceptual quality (PESQ/STOI) and robustness against defenses like filtering and pruning.



## 🚀 Key Features

* **High Stealthiness:** Achieves significant PESQ improvements (up to 32.5% in SR) compared to traditional audio backdoors.


* **Strong Robustness:** Remains effective under 6th-order Butterworth low-pass filtering and model pruning where conventional triggers fail.


* **Versatile Tasks:** Validated on Speech Recognition (SC-10, SC-30) and Speaker Identification (VoxCeleb).



## 📂 Repository Structure

(Code coming soon)





## 📖 Planned Usage

1. **Data Poisoning:** Generate poisoned datasets with controllable poisoning rate and poison level.


2. **LoRA Fine-tuning:** Optimize the watermark generator for better imperceptibility and robustness.


3. **Defense Evaluation:** Test the attack success rate (ASR) against spectral filtering and model pruning.



## 🎓 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{chen2025bloodroot,
  title={Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor},
  author={Chen, Kuan-Yu and Lin, Yi-Cheng and Li, Jeng-Lin and Ding, Jian-Jiun},
  journal={arXiv preprint arXiv:2510.07909},
  year={2025},
  url={https://arxiv.org/abs/2510.07909}
}

```
