# Frozen Gen11 baseline

- Checkpoint: `models/frozen/gen11/best.pt`
- SHA-256: `31832286c6493e1bbff77c3357b6787fd17d60e52a5a572483b70d2077fee536`
- Size: 22,585,924 bytes
- Source: `pv-long-run` generation-11 champion
- Original path: `experiments/pv-long-run/generation-11/candidate/best.pt`
- Architecture: `twixt-resnet-policy-value` v2; 32 channels, 4 residual blocks, 256 value hidden units; 22 input channels, encoding v1
- Board: Mini 10x10
- Intended usage: frozen evaluation opponent, browser Human vs AI opponent, bootstrap reference
- Git storage: ordinary Git blob; no Git LFS configuration is present in this repository

Evaluate a candidate with paired seeds and both colors:

```sh
python3 -m twixt_ai.evaluation.frozen_gen11_cli --candidate candidate.pt --games 2 --simulations 64 --output evaluation.json
```
