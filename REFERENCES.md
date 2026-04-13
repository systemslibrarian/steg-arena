# References

## Primary Architecture

**HiDDeN: Hiding Data With Deep Networks**
Zhu, J., Kaplan, R., Johnson, J., & Fei-Fei, L. (2018).
*Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 657–672.
→ Core architecture this project is based on. Joint encoder/decoder/discriminator training.

**SteganoGAN: High Capacity Image Steganography with GANs**
Zhang, K., Ji, J., Zhang, Y., & Lyu, M. (2019). *arXiv:1901.03892*.
→ GAN-based steganography with dense and residual encoder variants.

## Steganalysis (Detection Side)

**Deep Learning for Steganalysis via Convolutional Neural Networks**
Qian, Y., Dong, J., Wang, W., & Tan, T. (2015). *Proceedings of SPIE*, 9409.
→ Early CNN steganalysis work showing neural detectors outperform hand-crafted features.

**Ensemble Classifiers for Steganalysis of Digital Media**
Kodovský, J., Fridrich, J., & Holub, V. (2012).
*IEEE Transactions on Information Forensics and Security*, 7(2), 432–444.
→ Classical ML steganalysis baseline; the Warden is its neural analogue.

## Adversarial Steganography

**Generating Steganographic Images via Adversarial Training**
Baluja, S. (2017). *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
→ Pioneering work using neural networks to learn imperceptible image hiding.

**Invisible Steganography via Generative Adversarial Networks**
Tan, S., Zhang, G., Li, B., & Huang, J. (2021).
*Multimedia Tools and Applications*, 80, 2017–2043.

## Foundational

**Generative Adversarial Nets**
Goodfellow, I. et al. (2014). *NeurIPS*, 27.
→ The adversarial training paradigm that motivates the encoder/warden competition.

**Image Quality Assessment: From Error Visibility to Structural Similarity**
Wang, Z., Bovik, A.C., Sheikh, H.R., & Simoncelli, E.P. (2004).
*IEEE Transactions on Image Processing*, 13(4), 600–612.
→ Defines SSIM — the imperceptibility metric used in this project.

## Recommended Datasets

| Dataset   | Images | Notes                           | URL |
|-----------|--------|---------------------------------|-----|
| BOSS Base | 10,000 | Standard steganalysis benchmark | http://agents.fel.cvut.cz/boss/ |
| BOWS2     | 10,000 | Common pairing with BOSS        | http://bows2.ec-lille.fr/ |
| ALASKA2   | 80,000 | Kaggle competition dataset      | https://www.kaggle.com/c/alaska2-image-steganalysis |
| ImageNet  | 1.2M+  | Large-scale, requires subset    | https://www.image-net.org/ |

---

*"Whatever you do, do it all for the glory of God." — 1 Corinthians 10:31*
