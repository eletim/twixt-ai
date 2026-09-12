# Mini Twixt training inspection

Source: `experiments/issue-128/matched-1k/report.json`
Source SHA-256: `ade70e9f07f28e46123ab72b3ef0df04746a0e7e7f0e8f725aafc9ac5d7757f0`
Run status: **completed**
Probe set: `mini-fixed-positions-v1`

## Exact run configuration

```json
{
  "artifact_uri": "s3://twixt-ai/issue-128/matched-1k",
  "batch_size": 128,
  "board": {
    "height": 10,
    "width": 10
  },
  "dataset_window": 1,
  "device": "cuda",
  "epochs": 20,
  "evaluation_games": 40,
  "evaluation_seed": 1289000,
  "evaluation_simulations": 20,
  "games_per_generation": 1000,
  "generations": 1,
  "inference_batch_size": 8,
  "inference_max_wait_seconds": 0.002,
  "learning_rate": 0.001,
  "promotion_rule": "candidate wins / all paired evaluation games >= promotion_win_rate; draws remain in the denominator",
  "promotion_win_rate": 0.55,
  "rollout_limit": 4,
  "seed": 1281000,
  "selection_metric": "value",
  "selfplay_exploration": 0.7,
  "selfplay_progressive_widening_constant": 3.0,
  "selfplay_progressive_widening_exponent": 0.5,
  "selfplay_simulations": 64,
  "shard_size": 5000,
  "validation_fraction": 0.1,
  "weight_decay": 0.0001,
  "workers": 8
}
```

## Checkpoint lineage

| Role(s) | Recorded path | SHA-256 | Verification |
| --- | --- | --- | --- |
| initial champion | `experiments/issue-125/generation-3/generation-0001/candidate/best.pt` | `aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9` | verified |
| generation 1 candidate, final champion | `experiments/issue-128/matched-1k/generation-0001/candidate/best.pt` | `5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2` | verified |

```json
[
  {
    "candidate_sha256": "5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2",
    "champion_sha256": "5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2",
    "decision": "promoted",
    "generation": 1,
    "parent_sha256": "aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9"
  }
]
```

## Generation overview

| Gen | Status | Self-play games | Games/hour | Dataset examples | Train loss (first→last) | Validation loss (first→last) | Candidate vs parent | Champion change | Decision | Search budgets |
| ---: | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | completed | 1000 | 8152.7 | 25535 | 3.345→3.109 | 3.275→3.248 | 75.0% | updated to candidate | promoted | self-play 64 sims; evaluation 20 sims; rollout 4 |

## Fixed-opponent evaluation results

| Gen | Opponent | Candidate W-L-D | Candidate win rate | Games | Paired role swaps | Seed |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 1 | matched non-neural MCTS | 32-4-4 | 80.0% | 40 | yes | 1289000 |
| 1 | heuristic search | 6-34-0 | 15.0% | 40 | yes | 1289000 |

## Training loss components

| Gen | Train total | Train policy | Train value | Validation total | Validation policy | Validation value | Best validation epoch/loss |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.345→3.109 | 2.942→2.834 | 0.404→0.275 | 3.275→3.248 | 2.933→2.931 | 0.341→0.317 | 9 / 3.218 |

## Scaling evidence

| Gen | Total / self-play / dataset / training / evaluation seconds | Training examples/s | Retained bytes (self-play / dataset / training / evaluation) | Self-play / dataset / candidate / evaluation SHA-256 |
| ---: | --- | ---: | --- | --- |
| 1 | 526.938 / 441.574 / 49.748 / 18.631 / 16.947 | 74869.5 | 544426771 (232485992 / 311208285 / 669400 / 63094) | `bd05a6321a64afb890f42d9c5dcdf2cf39a83d2429f93cde3391e1510528eca8` / `9d79b587e041f044084d8f14933b98baca31b9fbc0c799b358322e6160d5d6f7` / `5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2` / `d7272a450dda05d04454dd8115eb73ed402b3945dd8c84caf4915d7f2fd86896` |

## Artifact identities

| Gen | Role | Path | SHA-256 |
| ---: | --- | --- | --- |
| 1 | teacher | `experiments/issue-125/generation-3/generation-0001/candidate/best.pt` | `aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9` |
| 1 | self-play summary | `selfplay/summary.json` | `bd05a6321a64afb890f42d9c5dcdf2cf39a83d2429f93cde3391e1510528eca8` |
| 1 | dataset manifest | `dataset/manifest.json` | `9d79b587e041f044084d8f14933b98baca31b9fbc0c799b358322e6160d5d6f7` |
| 1 | dataset shard (train) | `dataset/train/shard-00000.jsonl` | `c9bc5463efa1c8e462694741f4b9f9abe2ea7038de2d624581681df3cd8edffc` |
| 1 | dataset shard (train) | `dataset/train/shard-00001.jsonl` | `4e2a382ba6259e136a44a86dae861fb0527f2f6bd27607ec4b17fffbce610330` |
| 1 | dataset shard (train) | `dataset/train/shard-00002.jsonl` | `6b5a93ad326d8c46378db4a77469e686b253ea84c1fbd905f0eb44fdb3dfb972` |
| 1 | dataset shard (train) | `dataset/train/shard-00003.jsonl` | `400e3b64f73134a1c1963c39a4fd220aceda9535b8f195e81bfc7b8ec37a577a` |
| 1 | dataset shard (train) | `dataset/train/shard-00004.jsonl` | `8ac740c6146a17966dc9643d44b032e68eb86ce04638f5fb5877fa53e88abb8d` |
| 1 | dataset shard (validation) | `dataset/validation/shard-00000.jsonl` | `e4a5ee89afadfe4e74670008da8e5a996df358a1fac540316862ee0540169b09` |
| 1 | evaluation: parent champion | `experiments/issue-128/matched-1k/generation-0001/evaluation.json` | `d7272a450dda05d04454dd8115eb73ed402b3945dd8c84caf4915d7f2fd86896` |
| 1 | evaluation: matched non-neural MCTS | `experiments/issue-128/matched-1k/generation-0001/matched-non-neural-mcts.json` | `90807dce1f482edcf462d7ecfded7b58e42c612c75d9e04485c50ec39a2139d0` |
| 1 | evaluation: heuristic search | `experiments/issue-128/matched-1k/generation-0001/heuristic-search.json` | `bbd9de4ecafe091172c0455147c502292b21c0d3a03cf58518408d41642cad2d` |

## Retention manifests

| Gen | External URI | Inventory complete | Storage attested | Pruning ready | Category | Files | Bytes |
| ---: | --- | --- | --- | --- | --- | ---: | ---: |
| 1 | `file:///home/eletim/twixt-ai-artifacts/issue-128/matched-1k` | yes | yes | yes | dataset | 7 | 311208285 |
| 1 | `file:///home/eletim/twixt-ai-artifacts/issue-128/matched-1k` | yes | yes | yes | evaluation | 3 | 63094 |
| 1 | `file:///home/eletim/twixt-ai-artifacts/issue-128/matched-1k` | yes | yes | yes | selfplay | 1001 | 232485992 |
| 1 | `file:///home/eletim/twixt-ai-artifacts/issue-128/matched-1k` | yes | yes | yes | training | 4 | 669400 |

### Retained object inventory

| Gen | Category | Path | Bytes | SHA-256 |
| ---: | --- | --- | ---: | --- |
| 1 | dataset | `dataset/manifest.json` | 2095 | `9d79b587e041f044084d8f14933b98baca31b9fbc0c799b358322e6160d5d6f7` |
| 1 | dataset | `dataset/train/shard-00000.jsonl` | 61350883 | `c9bc5463efa1c8e462694741f4b9f9abe2ea7038de2d624581681df3cd8edffc` |
| 1 | dataset | `dataset/train/shard-00001.jsonl` | 60888038 | `4e2a382ba6259e136a44a86dae861fb0527f2f6bd27607ec4b17fffbce610330` |
| 1 | dataset | `dataset/train/shard-00002.jsonl` | 60611475 | `6b5a93ad326d8c46378db4a77469e686b253ea84c1fbd905f0eb44fdb3dfb972` |
| 1 | dataset | `dataset/train/shard-00003.jsonl` | 61303817 | `400e3b64f73134a1c1963c39a4fd220aceda9535b8f195e81bfc7b8ec37a577a` |
| 1 | dataset | `dataset/train/shard-00004.jsonl` | 33745011 | `8ac740c6146a17966dc9643d44b032e68eb86ce04638f5fb5877fa53e88abb8d` |
| 1 | dataset | `dataset/validation/shard-00000.jsonl` | 33306966 | `e4a5ee89afadfe4e74670008da8e5a996df358a1fac540316862ee0540169b09` |
| 1 | evaluation | `evaluation.json` | 16239 | `d7272a450dda05d04454dd8115eb73ed402b3945dd8c84caf4915d7f2fd86896` |
| 1 | evaluation | `matched-non-neural-mcts.json` | 23873 | `90807dce1f482edcf462d7ecfded7b58e42c612c75d9e04485c50ec39a2139d0` |
| 1 | evaluation | `heuristic-search.json` | 22982 | `bbd9de4ecafe091172c0455147c502292b21c0d3a03cf58518408d41642cad2d` |
| 1 | selfplay | `selfplay/games/game-000000.json` | 126623 | `5d9979c1a724969c7163a1bcf477855edff7eeb8c0fef31e8f067851719585bc` |
| 1 | selfplay | `selfplay/games/game-000001.json` | 468564 | `cfbedea192e93064cb6e5dcdd9557851a17ae2a38089a0975dbec880c7f40e6f` |
| 1 | selfplay | `selfplay/games/game-000002.json` | 155074 | `f6def9c9f5afb343e771d5e19af56d1e5b928e985d026d5be32fb0fa6e417a19` |
| 1 | selfplay | `selfplay/games/game-000003.json` | 570788 | `51148c69f6f4d361e93507598fe9c5794aac20df76b3d745c6e947170966118e` |
| 1 | selfplay | `selfplay/games/game-000004.json` | 343523 | `4db4ad2c3996dae3b1fffd11d0da33c9f2e7d0694185932b36d9a9cadbce83da` |
| 1 | selfplay | `selfplay/games/game-000005.json` | 126457 | `cd0990260053b39442e9b2af6b1e3b41f397415d02fb1c7737e008c16268f6a5` |
| 1 | selfplay | `selfplay/games/game-000006.json` | 218882 | `6e515aa5bc715d3fdac65943f2e45cc4cd84bd07189c9d480fdf4c41bd436ce8` |
| 1 | selfplay | `selfplay/games/game-000007.json` | 164563 | `67c3505e30c4b78efbd7cb34172db1723ad608a13c9073402a87288eacd41642` |
| 1 | selfplay | `selfplay/games/game-000008.json` | 154861 | `be2a1fd60ccac555e38aaa560e457f20d4c4516367c371cb0fef50278b0dde29` |
| 1 | selfplay | `selfplay/games/game-000009.json` | 145917 | `bba457322025fbe83690f1effa3b5d44f0af88d24270fe3788dea8d3c88d9b6d` |
| 1 | selfplay | `selfplay/games/game-000010.json` | 155307 | `b333312056d2a9a8690d29ee5471346be27489c93248c7ddfadfbb469c9ed38d` |
| 1 | selfplay | `selfplay/games/game-000011.json` | 457051 | `8b1fc34eeebd72038faf3f0fcad1f6707cfb98dc8ee1a2ecab0e86cde6f175d0` |
| 1 | selfplay | `selfplay/games/game-000012.json` | 164984 | `730792c28984b004d4b51e68c6bdfd5202ed718244fd9c4df0c6eb5654006f17` |
| 1 | selfplay | `selfplay/games/game-000013.json` | 173584 | `24b3f1277cf381df052a915f9ae69f5193a51ed531a7469a861d80fbfd39c9f3` |
| 1 | selfplay | `selfplay/games/game-000014.json` | 225970 | `ebf2db08d6c7292a3d675ce1ea23decae522c28720f3a199e8e70746ad0baf4c` |
| 1 | selfplay | `selfplay/games/game-000015.json` | 315307 | `272987104206b3e35f2cb83ea098db34b8b7213185c282c936a96160ab748b28` |
| 1 | selfplay | `selfplay/games/game-000016.json` | 154916 | `dd8eba442136a0032acd2b1db048357126a9a98b773841773bdd1d55bfc399f9` |
| 1 | selfplay | `selfplay/games/game-000017.json` | 268486 | `f278a339dd49c91c591c49b8e15bccb335b5e1ef38eabfa90e73bb08ecb020ca` |
| 1 | selfplay | `selfplay/games/game-000018.json` | 154914 | `a78b249fd3274e73dfb397e4bec7f4bcaddae55a8316b582a281e2bfa3650c82` |
| 1 | selfplay | `selfplay/games/game-000019.json` | 298938 | `fdb8039067363395c84b2a8b151a45fc9c81598197d8421f0a9960220b49f37b` |
| 1 | selfplay | `selfplay/games/game-000020.json` | 218224 | `c0d4b054ac7a1756e3bee6bc36c12b5ae1bc8b1664b8616300af0c8380384724` |
| 1 | selfplay | `selfplay/games/game-000021.json` | 155084 | `259615bef91dd5f39a7cf2f7d2231d1fc61d5b0fe0349cfc5685a646d4f3fe2d` |
| 1 | selfplay | `selfplay/games/game-000022.json` | 566129 | `521e9530f1b796972fecc771bd4e55e2679a49e0f8240ae961877d7e71118287` |
| 1 | selfplay | `selfplay/games/game-000023.json` | 436791 | `5d34ac1a31c40486b482603683c6c1616e617c3b781ac140b7d622b6a7dc7263` |
| 1 | selfplay | `selfplay/games/game-000024.json` | 126603 | `76fd73401afc086741d93fbabf3a60721fbd147330921f914a4f9af28d10f4c1` |
| 1 | selfplay | `selfplay/games/game-000025.json` | 145822 | `76667f36d056f532dca466d3b1cac64c1676c9edbe6cc6b1c3902e1beeb71bfb` |
| 1 | selfplay | `selfplay/games/game-000026.json` | 155153 | `19c316db02ca548e48547ff4e05ff184c0b937fad9652ca6278a4418b7451bbb` |
| 1 | selfplay | `selfplay/games/game-000027.json` | 155066 | `74ca8ee5980438a276fdf9e72db87d1d6da7d88f2bfb72ade809c0e735bb7720` |
| 1 | selfplay | `selfplay/games/game-000028.json` | 382536 | `866af4ddd28b85dcf17d8c76311085152b48510b1d83ad6792083b012d244170` |
| 1 | selfplay | `selfplay/games/game-000029.json` | 126633 | `5513e94ec81d550011d73f962233d6f0d401d5c5846c81fa884a0e522c71d41a` |
| 1 | selfplay | `selfplay/games/game-000030.json` | 200619 | `443504a46456a4ad0f4829e308e6467e128c0b2694c11ecf332b70eace62538e` |
| 1 | selfplay | `selfplay/games/game-000031.json` | 284670 | `69d9653097f08fc16131e47245d3b2e57b9a1609a44bafabd578368d89575dc8` |
| 1 | selfplay | `selfplay/games/game-000032.json` | 126627 | `d3fa5ae52cc693ba1849803a19bc3ea0abbab398b0ee07b423bfa25853663978` |
| 1 | selfplay | `selfplay/games/game-000033.json` | 252224 | `36ea0fde3c1058a6be5081a52f58b63c166a7a9faece327ed68c950afd74578f` |
| 1 | selfplay | `selfplay/games/game-000034.json` | 164859 | `3822c00fd562840fd62475ad2b8256bc8bfe982bddb60df382951b670e019dbd` |
| 1 | selfplay | `selfplay/games/game-000035.json` | 225812 | `c80192f68c2ca0137f23bef9c95d1a568bee467fe1179f06bccec626f403ccad` |
| 1 | selfplay | `selfplay/games/game-000036.json` | 126494 | `5917f0cb9745dcc3f8b176bd0b93486da0b47f5b5949c50fa8de18eff1089474` |
| 1 | selfplay | `selfplay/games/game-000037.json` | 145716 | `7e1807e6cf1a1aa2e0576dd2611043d6d4ccd7d10b9c116f7b6fed84405a66ed` |
| 1 | selfplay | `selfplay/games/game-000038.json` | 225614 | `f8d62161b5041cb52972811031272b71b1e3e903971028ea8d55a826cef63ad0` |
| 1 | selfplay | `selfplay/games/game-000039.json` | 182726 | `d5788e4b21696d3be4a282908f8d11cb93844ab68127753d8a5af5ec8ddbc40b` |
| 1 | selfplay | `selfplay/games/game-000040.json` | 155455 | `65af3188a1fed589e9ac4b7b2c56d27dc17f0a25c5d1556019e0af4e82ddc20a` |
| 1 | selfplay | `selfplay/games/game-000041.json` | 267351 | `75031fe40451ce06d7f2da292e8a370508c0c05189c50fccf53edb601babc17e` |
| 1 | selfplay | `selfplay/games/game-000042.json` | 209196 | `6a06e5d61b0be55f5c445e63e2cfd09ceaeaa88b06e7a15d961bb73c9aaae486` |
| 1 | selfplay | `selfplay/games/game-000043.json` | 164766 | `f76a3c393de0f0f0ada92a954213f7090f68277d79dce2ffebf8f6d7036cddba` |
| 1 | selfplay | `selfplay/games/game-000044.json` | 236308 | `13aa60783aed44d55ce608ea2cdea0981b6b75e63658498e09f5678b3531e119` |
| 1 | selfplay | `selfplay/games/game-000045.json` | 155068 | `d89e342ecba65a25211e96e47fe293c658266912983ba0414ae6d8579fdf8935` |
| 1 | selfplay | `selfplay/games/game-000046.json` | 164656 | `a28a3c5149870f5cd3e2258d9ddd3de89b6b134e2d8818fc8317171f2b153b14` |
| 1 | selfplay | `selfplay/games/game-000047.json` | 357100 | `9ccbb445f516d4e7ae30f5d216696888445536210362280719566af066cdbafe` |
| 1 | selfplay | `selfplay/games/game-000048.json` | 311802 | `d00bedac874e43e7f8802b97440f5ea8014f8cad55f8c8e4fcbb57beb3216b3c` |
| 1 | selfplay | `selfplay/games/game-000049.json` | 145807 | `776c493d66fd495e1692591c696e8177715258bc256fa71f09c640c83e97a8b2` |
| 1 | selfplay | `selfplay/games/game-000050.json` | 218160 | `4735d908e83c28d30a6f7ef4e6f3eb2fd87677c83ad03a9415783f44c968f976` |
| 1 | selfplay | `selfplay/games/game-000051.json` | 126291 | `ae334a222ea933d85f48f5569675b87dc16226b9a67b346da5003930bfcd9120` |
| 1 | selfplay | `selfplay/games/game-000052.json` | 365653 | `a4e200cf2060f2d5cc25d1820343536a14a5c08af833f8a0b762b46afa495d27` |
| 1 | selfplay | `selfplay/games/game-000053.json` | 201359 | `83c90077558d0905e66ce4c053a70abfeb7545360c96efe01ca05f8318999b2e` |
| 1 | selfplay | `selfplay/games/game-000054.json` | 209261 | `357ba2ef90e92a7560d35ee526e649d30e25e3fa952d94ca137710ad7b64e639` |
| 1 | selfplay | `selfplay/games/game-000055.json` | 209230 | `b6ecfea87f0cd9ab5ad9d5532922cd44af3d59a9331142023c602a191311cf01` |
| 1 | selfplay | `selfplay/games/game-000056.json` | 173269 | `d74c655c7109407e0aec195ebecd5e6ccc17f717b160465e81d7df9a3ade73c4` |
| 1 | selfplay | `selfplay/games/game-000057.json` | 357102 | `5d9dc0085e2df5beef4d57213fd4a10a2ff4878b08c144486da20178e2e9d948` |
| 1 | selfplay | `selfplay/games/game-000058.json` | 126596 | `e43a254e85c965f0017ef4f983d185e695493c65aef90611f126db0f7460f52c` |
| 1 | selfplay | `selfplay/games/game-000059.json` | 201179 | `86f2e3df36b245bfe4348ac97a1a07a0c5bdc42765d12d19391d952ec20e7b91` |
| 1 | selfplay | `selfplay/games/game-000060.json` | 251092 | `ffc95d22b8d6c0073fa726350ec7a15a07600f1a565d223c0ccc852d26430545` |
| 1 | selfplay | `selfplay/games/game-000061.json` | 439096 | `9920d59adfa6e0ac08bbb7ac4d0b96356f2eb2ee4e19d89acb2fd95b40f50e81` |
| 1 | selfplay | `selfplay/games/game-000062.json` | 226114 | `1e5369910eb5976cb0d25aaa06c97b7494fb29af39f66ec74fad077f61a234f6` |
| 1 | selfplay | `selfplay/games/game-000063.json` | 397022 | `a139b7a655a94d3130ecfbd1b995bdde7f456a988254575ee98aaece2791fc2e` |
| 1 | selfplay | `selfplay/games/game-000064.json` | 369329 | `ded4919fa1dcdbdcc1d1e75b422acb718b71da7e0b77200a565d2145069ca487` |
| 1 | selfplay | `selfplay/games/game-000065.json` | 155248 | `a3b19ce4b1cc891670c8a530bc9cce14229ef32af9942db9e079cea1e0e99b34` |
| 1 | selfplay | `selfplay/games/game-000066.json` | 126520 | `f4adcd6afcfd99805ae8972b1f2f963d4067efa0108dd08889d89a9c7fc8901b` |
| 1 | selfplay | `selfplay/games/game-000067.json` | 173596 | `e6fbd1feea8abd620715ec162cbe1c59f506e1fb563a6e559afc0b6ee42dcf6f` |
| 1 | selfplay | `selfplay/games/game-000068.json` | 342792 | `28261ccaeaabc9d8d9c6b5807646fd915f9ebc2d52c3a4cd3c675468ee6d96a7` |
| 1 | selfplay | `selfplay/games/game-000069.json` | 126641 | `5bfc8ccaecb5d585e4b12f23d87737b3314e02746367604412f5821ad15404b6` |
| 1 | selfplay | `selfplay/games/game-000070.json` | 145796 | `12f20fe222893f9a562b39659a91e9cf7c715dd6f0289530044d617036a97e6b` |
| 1 | selfplay | `selfplay/games/game-000071.json` | 145567 | `53b7c7339de5acb911ea66d60f7f3e0e39f503c34cf7609bb4f064c6e0b1bbc6` |
| 1 | selfplay | `selfplay/games/game-000072.json` | 565883 | `1078513d93ff3ff37932999871b68bd1c0b7a2104d9899476c12104077c201c9` |
| 1 | selfplay | `selfplay/games/game-000073.json` | 252100 | `c8e020f4d7de025008171ac475d34a0bae4bea0ba320fafbc0b750453b398339` |
| 1 | selfplay | `selfplay/games/game-000074.json` | 116335 | `b1ca5057515b7091a1e16aeda7a826064162d76ff10ab473eb5d2996cea5378e` |
| 1 | selfplay | `selfplay/games/game-000075.json` | 154961 | `8ae59750c06b1a88478de85ed0c6ca1e0c8d7c985cf75c8da3ee4569a157de2c` |
| 1 | selfplay | `selfplay/games/game-000076.json` | 155077 | `f1ee7d6085a21e1948ea3894a435cdb5524468de0fa0204691d3499eea8cc999` |
| 1 | selfplay | `selfplay/games/game-000077.json` | 173712 | `a38e74ee651ad0a3583a6b76d3615abffd7eff50cb4e351736d7a8dc64d669d2` |
| 1 | selfplay | `selfplay/games/game-000078.json` | 225633 | `8d12f06c646bd3dda9d19014c95d0ebaf7beb3a2dd2d3cec753c352b10e2e58e` |
| 1 | selfplay | `selfplay/games/game-000079.json` | 183426 | `9aed07486936cc50ba7fd18ae363e593cb3e4e82d3ae4dd4e50e0eb103a0544a` |
| 1 | selfplay | `selfplay/games/game-000080.json` | 226822 | `6f6fb7c9e932a449095789177b694fbe6cebeb6dfb58b6624554026e43f1cae8` |
| 1 | selfplay | `selfplay/games/game-000081.json` | 299661 | `b32095c650d47c189c92e46c1f4dcbbf5eb070d9e6e4adb8a1a36be9c3daa400` |
| 1 | selfplay | `selfplay/games/game-000082.json` | 470332 | `e6fbd0a2b413bcc81cb97aece6dc08f473ddea062d9bb95bd0a27785ee27e1ab` |
| 1 | selfplay | `selfplay/games/game-000083.json` | 126483 | `af7552fe10d77ed730a47c04bc237ee81dc2f730187493c7c0488d1a2798b952` |
| 1 | selfplay | `selfplay/games/game-000084.json` | 500372 | `fa3e9e7f1fd7d281ce133b5ef4f546798903c769cb1a7bd688f646ebf40f63c8` |
| 1 | selfplay | `selfplay/games/game-000085.json` | 267713 | `83f94d4940666e97f28e580ce92597e330284fe014abf952a5c7810f2a40020e` |
| 1 | selfplay | `selfplay/games/game-000086.json` | 326448 | `f3d39125fa851d7e2f9d20b2b6ef1b0d7021b3874c1f51d238b2fa699eba56a0` |
| 1 | selfplay | `selfplay/games/game-000087.json` | 155105 | `b09b9132c43c8953a132ce1f2aae532ae7bb8fe11a74681de730280f627cf62a` |
| 1 | selfplay | `selfplay/games/game-000088.json` | 173596 | `e46bd19ea23d3b4173a609e966b6861327e77cd0e7ba6348494f5ca7c7083c4a` |
| 1 | selfplay | `selfplay/games/game-000089.json` | 299302 | `83204b6cab77ec807c826ca64d84c2a42f4d224d1f33f2760fdf337e58b8640f` |
| 1 | selfplay | `selfplay/games/game-000090.json` | 191578 | `9d55556fc5ed5c87f880897eb139cfd3e0bc9da836bd0bd7282a62d2f1f73412` |
| 1 | selfplay | `selfplay/games/game-000091.json` | 164758 | `be1404a2173ed3e91f9822c5f064d5e176285e96c9e931f11ad17bbd7564119c` |
| 1 | selfplay | `selfplay/games/game-000092.json` | 173609 | `bf79eb76d3d8365c86a03c22c92814c68287a8356844ed35f637e3e84dfcb6b5` |
| 1 | selfplay | `selfplay/games/game-000093.json` | 356431 | `f80ee44865e70fbd1a1d020b53daaf2936bedc1a1998fb3cfa523fb239ca44a2` |
| 1 | selfplay | `selfplay/games/game-000094.json` | 200317 | `c1489be468cfba6bea22bcc83a7eda1ac1b738b419d826573b31771bf0f80816` |
| 1 | selfplay | `selfplay/games/game-000095.json` | 154910 | `3f7b57ed60dfd17cd02e0f99eb3e2e3e390183e6fdce2e1e84e62f216b71a3e2` |
| 1 | selfplay | `selfplay/games/game-000096.json` | 154980 | `09837aefc0a6f8372eaeb889b04d2600dec66eceb4970fb571f36fc7302db726` |
| 1 | selfplay | `selfplay/games/game-000097.json` | 135756 | `d652708346291b589fb5b2eacdc9fb28be07d9d494304cc0ad1f2805986068a7` |
| 1 | selfplay | `selfplay/games/game-000098.json` | 432644 | `da975cb316017fae5c525bcea37f64347f90cd4fc8aff12e5df9a90d24909d2d` |
| 1 | selfplay | `selfplay/games/game-000099.json` | 126461 | `1d0f64e11a4bd4cc9676375177400e52d30e82fb94b1bb0165b0c0e94839e51b` |
| 1 | selfplay | `selfplay/games/game-000100.json` | 183470 | `6e9f9f4c99f15370ce5b91186ee42b78264800424922fb2560bcf58107556ff4` |
| 1 | selfplay | `selfplay/games/game-000101.json` | 258618 | `6befb3ad58c5a5702dd1feab2481b584f8ba2f5e41859648a82b2b6467dd0b40` |
| 1 | selfplay | `selfplay/games/game-000102.json` | 182624 | `678b87178c1aec53d17a563c7ec1169301d5ae95e742b6028f71bc77b19518bd` |
| 1 | selfplay | `selfplay/games/game-000103.json` | 145740 | `9465397299fef7288112346c2d20df79c9a5105879b8c10eb7b4057e4e9ac335` |
| 1 | selfplay | `selfplay/games/game-000104.json` | 182662 | `0c33e27684742d51d6611d3a44e9353b2d02f1933569b32370e1364d175b963d` |
| 1 | selfplay | `selfplay/games/game-000105.json` | 155092 | `71ea7b8077d7ecdca59074441681977ce19fc83b9b33574167fec843e3c52c34` |
| 1 | selfplay | `selfplay/games/game-000106.json` | 145845 | `1250f771edd80d084a6bb214c7773c60bf1198570defcfd1e25867f8ab96cb52` |
| 1 | selfplay | `selfplay/games/game-000107.json` | 155097 | `2b64ed83434bee2009eecae470a758ad820eb67fcaf325be6b4305458c6c55f6` |
| 1 | selfplay | `selfplay/games/game-000108.json` | 560881 | `ad60142fddac329ad37f1a987a584eca23fc2aa05b62c84d3b71cba7448d834c` |
| 1 | selfplay | `selfplay/games/game-000109.json` | 392892 | `53f8d6c5c3cbb5532325bc3afbf37007bb3693f9c37e690323e4867ca14cbef3` |
| 1 | selfplay | `selfplay/games/game-000110.json` | 145918 | `290833554f71b3370322ba09acc5b772be3442bade5698c1b90432bd3082c9bc` |
| 1 | selfplay | `selfplay/games/game-000111.json` | 380370 | `682cabd9c9bbeb702b9510a6f1a2e114c1dafbeea6027d5d09bc77aa6c3ce5a5` |
| 1 | selfplay | `selfplay/games/game-000112.json` | 155087 | `28339fe67160c198e0e9fd873f0743639f5c5dd04c07756bc0da412b3578dc6c` |
| 1 | selfplay | `selfplay/games/game-000113.json` | 126456 | `2a6a5d3dc0c4f3a3869bca27e291c8e5d97a72b840de511caf9794219ad11661` |
| 1 | selfplay | `selfplay/games/game-000114.json` | 384485 | `9fdfbeb4718545d78112249085e9ed12d34c8bd5fc2c92b1b3807eeef496be8d` |
| 1 | selfplay | `selfplay/games/game-000115.json` | 155446 | `b4fe687f266887b6a138fa59bdf0e55eaf4c5a94abf261078ce97152ee7d5e50` |
| 1 | selfplay | `selfplay/games/game-000116.json` | 268635 | `28f9a741346b4acfc26f7419d45c8d837896339dff5158a4d228e34e5f7c1664` |
| 1 | selfplay | `selfplay/games/game-000117.json` | 352581 | `cf9f26e5749b248c178d32e56be17ad9a8dcd74c96696f79fefd110607616df0` |
| 1 | selfplay | `selfplay/games/game-000118.json` | 145935 | `48aa156305f6bd9f7afba7855fb6f553005bbd54242d2bc782eeb06b1c906df1` |
| 1 | selfplay | `selfplay/games/game-000119.json` | 145828 | `0a9cec0680ddb8b40596b1c1490fadcf0baa4782a44a62043f173c187d317bb1` |
| 1 | selfplay | `selfplay/games/game-000120.json` | 201277 | `316f0c43b7f1c8dd8e51ea4c6754dcd85540145ebbd64f81793c028368d4719b` |
| 1 | selfplay | `selfplay/games/game-000121.json` | 154833 | `9557f986504d829bafa9534adb7b981fb697b67d56bd48a463327607c2f12bf1` |
| 1 | selfplay | `selfplay/games/game-000122.json` | 298428 | `5a63e7fa594b278f7b4569752f4402ed9fa281227701eb1f8fa179ad26abe163` |
| 1 | selfplay | `selfplay/games/game-000123.json` | 126654 | `d384d892c3608e79dc380a66ca6d98e48f3ec24e662bb50cec68e947ab44d8c1` |
| 1 | selfplay | `selfplay/games/game-000124.json` | 164569 | `c12fbe1b0d05623bd25c6acd6f947834f9fe3e64585acf0a5623ef8cb48d0dd4` |
| 1 | selfplay | `selfplay/games/game-000125.json` | 266708 | `0cc3a5da836872773804fe94dc80cb5a1aef243b7745b01fb952048bbf3b8f7e` |
| 1 | selfplay | `selfplay/games/game-000126.json` | 319206 | `1c220f5a5fef964efc8f86cb12fa8fdd4464729baadfe41918bb03e29d9003a4` |
| 1 | selfplay | `selfplay/games/game-000127.json` | 452595 | `b5d1a5a9c75b2b7d2452c0ee64793ea922ab9ffa9eba7c4ce12d56e58986e051` |
| 1 | selfplay | `selfplay/games/game-000128.json` | 182667 | `6e90431c07d855f5278a61d2c43c97826e0ed8aaa65a7da766ccb171268e0350` |
| 1 | selfplay | `selfplay/games/game-000129.json` | 145804 | `36c6bba555c5ba1ad172de704987e53fd77504144e1bafd339105472046c02c8` |
| 1 | selfplay | `selfplay/games/game-000130.json` | 515942 | `5ae7c50f591cb0ad7c28e37c04824bd50be7404467f33940d96d0eeb0978c335` |
| 1 | selfplay | `selfplay/games/game-000131.json` | 217625 | `ddc76cd8cc4b0c91dc49d62ebc48b2f6522e5ab3414c3ca3529e5fa39d372733` |
| 1 | selfplay | `selfplay/games/game-000132.json` | 126608 | `5e5306c2b05ff26705ed3792c6a82ff8d07d6556d11519b8188f280bc19bda18` |
| 1 | selfplay | `selfplay/games/game-000133.json` | 441009 | `7f5cf18a8284e6d6cea2c022448c261f266ff6e40996ca3d7c8390b4715176a1` |
| 1 | selfplay | `selfplay/games/game-000134.json` | 155086 | `303f21ca9a99c2a1c4423c24112764facfaa050d7cfc03ccdd3a4ef45574bf23` |
| 1 | selfplay | `selfplay/games/game-000135.json` | 201540 | `267c7b20a5d8585f690c1261568f623f0e70a4ceb21b98eb52e603aaf34dfa77` |
| 1 | selfplay | `selfplay/games/game-000136.json` | 358045 | `614408c0ec10e3611adfef9dd4a9464c1d5de50a2951d9f95d5a4bfd66f06341` |
| 1 | selfplay | `selfplay/games/game-000137.json` | 182855 | `c7b0ec47bd862be821f1a2aa6eed8bfb9827cfeae826a1e3c0e6712229df7ae5` |
| 1 | selfplay | `selfplay/games/game-000138.json` | 126608 | `bebccf63828b9f4df9a2b96b16b82e1fcfee983562bc4000526c4b2b2e2519e0` |
| 1 | selfplay | `selfplay/games/game-000139.json` | 164484 | `37e0d3778c8d9d988d4af6a2ae8ff0ada9a6de017a32ef920499b5e28980ef8f` |
| 1 | selfplay | `selfplay/games/game-000140.json` | 173593 | `fa92900fd2c8c88ff30af52c02395f3b66373cbd2b629d7c8902d06eecffeeb9` |
| 1 | selfplay | `selfplay/games/game-000141.json` | 209584 | `335957402c37f30dbae21be242d9eff27eb8807753a69d62cffe0ff734dfa3f3` |
| 1 | selfplay | `selfplay/games/game-000142.json` | 314165 | `a26f0a732cb2c39d9361472074c8148ad282f0bc0e4878541dbdf093ab44497c` |
| 1 | selfplay | `selfplay/games/game-000143.json` | 369073 | `d5d37f639818d8133166842444025e2cbb474d63456930b0fa06e770420b6154` |
| 1 | selfplay | `selfplay/games/game-000144.json` | 155093 | `22211d9afab0e2e40a62a5e4a00c7585b671db9a96ac522f3eb29a252a6154fe` |
| 1 | selfplay | `selfplay/games/game-000145.json` | 225629 | `9d18b62398a7951a2155ce7686c055e3723fb281b15ffe3a6edfd6ce12807f0c` |
| 1 | selfplay | `selfplay/games/game-000146.json` | 200773 | `847ad97cc56bcb89878f12ed839aa43bbff4c4dccccd8a17150d6266f90cca17` |
| 1 | selfplay | `selfplay/games/game-000147.json` | 126466 | `2c728a945e8c6716e190e7262f31a45bf6552a1ecf29cae6bff577a1af8f36d5` |
| 1 | selfplay | `selfplay/games/game-000148.json` | 173443 | `722373cfd7dea76e886edb188f3805e2c5b6cce94bfdd99dca513cc009df89d3` |
| 1 | selfplay | `selfplay/games/game-000149.json` | 155097 | `a1b75b129030e057008a00b3e63ec2c9bd9facfce2bde22b15f509f99a94b1d5` |
| 1 | selfplay | `selfplay/games/game-000150.json` | 299663 | `836d257c16e3d4fcc7dbfa24a612be015359a489064701c2dd935ad91f1ba86d` |
| 1 | selfplay | `selfplay/games/game-000151.json` | 312863 | `29826ac94b9857987ed85f276e448e9c983a3991ea2d4d7e05509bdd08a78598` |
| 1 | selfplay | `selfplay/games/game-000152.json` | 556009 | `a270382f76b10a2adcd6bddddcc4e41042d8388d5a99573229b4b02bae02eb9c` |
| 1 | selfplay | `selfplay/games/game-000153.json` | 126837 | `7173c2d8552d038185125df2808c946d31b6e1786c319fc69a4abf4dde619321` |
| 1 | selfplay | `selfplay/games/game-000154.json` | 164710 | `cef48efe8896565142a4af64ebfe14aedaf4f5d7acd0b59a6fb26747d70e0f82` |
| 1 | selfplay | `selfplay/games/game-000155.json` | 301044 | `63889fb7afbba69b71f1088f93dbba430d270d03d887e63e103d4c8d1623c304` |
| 1 | selfplay | `selfplay/games/game-000156.json` | 154918 | `208af9a3cb18ac8c18836ac430254dc33cee80605575baee5358c7d63f7fde45` |
| 1 | selfplay | `selfplay/games/game-000157.json` | 225622 | `d8de5ead53298698067643df27d6dbd2df00745a2cbd60d2ebd42650a9d9e635` |
| 1 | selfplay | `selfplay/games/game-000158.json` | 225630 | `c15da99fb41f8e4ee890864cd9dd9b7ac3b0c83e2680121dc8fa04855a41c280` |
| 1 | selfplay | `selfplay/games/game-000159.json` | 342716 | `73394305444609df9a836099bc2f052fe3fae9efb37a0607a3862f2d0e3d7b9b` |
| 1 | selfplay | `selfplay/games/game-000160.json` | 200317 | `84c0de9d84646597ef3d2bb26c8d72060851d2df0b10170ee58b0bb4547ac1e7` |
| 1 | selfplay | `selfplay/games/game-000161.json` | 548979 | `b7ab8349bad8b2d9aab44002db82cb6189d94a90286b79e4d2d431b152750a30` |
| 1 | selfplay | `selfplay/games/game-000162.json` | 235357 | `69d6644ed8671e1443b374b5fba1bd393c073b7dd66940efafa2171556fe75c1` |
| 1 | selfplay | `selfplay/games/game-000163.json` | 452682 | `259e6a5f440b0cc1f035579d4dbe4f5f0a6c53c009f24c45fa7b4cde974e4940` |
| 1 | selfplay | `selfplay/games/game-000164.json` | 259252 | `181a6dca38646fb08d1e109a897602e919bfd82c09d52e5dbcdfdbd31d98dcc1` |
| 1 | selfplay | `selfplay/games/game-000165.json` | 360243 | `35485415024d1b0fd2a7d1bcdea88536d127899aae8121b5ea40b5f0de429081` |
| 1 | selfplay | `selfplay/games/game-000166.json` | 355549 | `6f24eb9ed385d9454ba357f5412c45abef81a55840344cce1b76526ff2263d47` |
| 1 | selfplay | `selfplay/games/game-000167.json` | 164780 | `73f8abe4625a18db1ffb40449638a4ea182225225038d4a9b5cfbdc389449990` |
| 1 | selfplay | `selfplay/games/game-000168.json` | 209486 | `3defef84e986c2de7b0f9a782be66af4d03d76ed5d8991abbee06bad974d238a` |
| 1 | selfplay | `selfplay/games/game-000169.json` | 191314 | `ce65830c09f088abdc8479067474f4ee286cce5c8e183ca5fd2d890b7f9b2058` |
| 1 | selfplay | `selfplay/games/game-000170.json` | 126614 | `c6378991c63704c2d4a1dfb37725665b2780d132a0fef99272a041e1acc69a8f` |
| 1 | selfplay | `selfplay/games/game-000171.json` | 126510 | `88f17811e7edc6f405da8bc9808fdf7244e6d7c74ed252e2db361be56dee3c50` |
| 1 | selfplay | `selfplay/games/game-000172.json` | 155127 | `c4b643d91c83147627371c7bfc99b540945c580e4548e7a9f71e3535b234722e` |
| 1 | selfplay | `selfplay/games/game-000173.json` | 218868 | `1815b1baa50c80747ea0c08dca5c9cf908470d64b8427d19e721eea6001adb45` |
| 1 | selfplay | `selfplay/games/game-000174.json` | 329079 | `b8ed03537cef025019efa3aabe49953e8146b199b934404bc0e9e6395e8b5664` |
| 1 | selfplay | `selfplay/games/game-000175.json` | 564110 | `0985b06a0a37e46814dde8c1b9422f7970708c1728c5b14e7fdd1549896ea515` |
| 1 | selfplay | `selfplay/games/game-000176.json` | 155018 | `755b7a82fa2a9e0234bb6928257a986157d809479235116fee92cd43b64046e2` |
| 1 | selfplay | `selfplay/games/game-000177.json` | 146058 | `358d56350f99becbcb6c224cf5c32f5b17d80156a008747dfaac229bf5f197a4` |
| 1 | selfplay | `selfplay/games/game-000178.json` | 200899 | `8f2f34ee769a34ab20453b737951403b1ebd3d4a913a84d4d42dd76d065deefd` |
| 1 | selfplay | `selfplay/games/game-000179.json` | 234469 | `605f2d8c48d0a98da96544c2317641b4c34cd991b8b60146ed8e9144054ab6ef` |
| 1 | selfplay | `selfplay/games/game-000180.json` | 299912 | `83df14d0369b3574c41e6a5de4ce21816a9232bdbdcfece1a7d564bd396e5af3` |
| 1 | selfplay | `selfplay/games/game-000181.json` | 381036 | `044d5289006cb0f23c71251bbadc4f2aed2581b4249c1193d64fcc96db9c0161` |
| 1 | selfplay | `selfplay/games/game-000182.json` | 417301 | `e3256a254ddaf1c81cf169b213ece15582c2c24fc9c20a4905aec38c114f0bca` |
| 1 | selfplay | `selfplay/games/game-000183.json` | 557741 | `3c19dc0fab4e6acf61edc10b6c4316a2e28aa8ef83c66786685987aae674354d` |
| 1 | selfplay | `selfplay/games/game-000184.json` | 173782 | `48640c05ec5489e2f0d987c97abb11d52d4701e50509af541ff3ce74834b50f7` |
| 1 | selfplay | `selfplay/games/game-000185.json` | 182672 | `c2db3db30e7683b68ca6d84f5f1f699d20bf38a10f696a7c36e0f4bbe4105883` |
| 1 | selfplay | `selfplay/games/game-000186.json` | 154994 | `0badb99d40ebfe77d137dc6b4f2440446dc354efbce8d9834bc5efe29ae2a16e` |
| 1 | selfplay | `selfplay/games/game-000187.json` | 314329 | `10d114ef64314f45ea118d909a317ac6b5f7c32ae039bbe966c20e9a74c38d11` |
| 1 | selfplay | `selfplay/games/game-000188.json` | 155107 | `da4975de008d7237d75fc95ceec22f2d11118363a1ffbd9890aeb0b666e65988` |
| 1 | selfplay | `selfplay/games/game-000189.json` | 242297 | `62b691b9e7326ddcd134f8850cb905b7b17724ab28f1fbd2e3eb134faf800a15` |
| 1 | selfplay | `selfplay/games/game-000190.json` | 268822 | `843e40657910d46f14d2db834f56b4bdb11e4265ed639c3bf5bfcaf021fc4556` |
| 1 | selfplay | `selfplay/games/game-000191.json` | 284268 | `c4b16f5f324e568676af481280afeb23739b5dd40e6dad90b78043f1b549ade4` |
| 1 | selfplay | `selfplay/games/game-000192.json` | 154917 | `f1e4edfb1218bfb3ca5428d43e171964d4e44a54737758abbe86550c345a4bdd` |
| 1 | selfplay | `selfplay/games/game-000193.json` | 191827 | `3fae25920e067eb7dcbed675298e6de51bed18b1c8330e7728474a2ae1cf5548` |
| 1 | selfplay | `selfplay/games/game-000194.json` | 191336 | `707db25f0ff15a303dbfdf1222f8f91376393a24ec2744e420cd2f4b6b8f1f5f` |
| 1 | selfplay | `selfplay/games/game-000195.json` | 126603 | `ecf32aeec5e3b070e868e55d5cc0954fd5b7207c78042b4bca378abb9966523b` |
| 1 | selfplay | `selfplay/games/game-000196.json` | 283022 | `44eb8b05be5dc75734be8a768c6ca72d265d3a07969363f9ac03f49c2b450bbc` |
| 1 | selfplay | `selfplay/games/game-000197.json` | 126612 | `72faa76b4d99f709707ed05f81af8dbb00bfa7d24d30a6f2cdd5480ebe4bbef5` |
| 1 | selfplay | `selfplay/games/game-000198.json` | 116346 | `fb469b8a5ef777b10bd9925968192aa13d4ed695fd3ff2b347393311974c7b77` |
| 1 | selfplay | `selfplay/games/game-000199.json` | 327437 | `7636c696d39fdcb04d8a132706464b8c1c671ca856e1d82f2c4023f439e605ff` |
| 1 | selfplay | `selfplay/games/game-000200.json` | 568647 | `0aa4de2a266b61dd910ef79668b9bfdc974fd5865190980e0374f8a040921ca0` |
| 1 | selfplay | `selfplay/games/game-000201.json` | 173450 | `ab3085b53a120cd2f083f88485f24ddb8b2ea740100e73aecff83fa757fd33f2` |
| 1 | selfplay | `selfplay/games/game-000202.json` | 191297 | `580f2d29857aa95b166f086080b38710f78d4c001d718b207f79708dca25c6ad` |
| 1 | selfplay | `selfplay/games/game-000203.json` | 126636 | `70ac38233f41b203df7ddcbd9c6ba612dbb99e1dffa8f023d3b93bf096741ff7` |
| 1 | selfplay | `selfplay/games/game-000204.json` | 164613 | `0033ba1517a7eb7b97b57a25e406e354c1595761dc21785693205e5e61e62194` |
| 1 | selfplay | `selfplay/games/game-000205.json` | 367110 | `5b4d606be73b4398ff64ebf24a510c1b529b47e0e3dcb03fdb1d7e753361318e` |
| 1 | selfplay | `selfplay/games/game-000206.json` | 155099 | `6da356c2eb3e081977a99019a37964f71359b0b2416c8879806b6010dad8598d` |
| 1 | selfplay | `selfplay/games/game-000207.json` | 252797 | `bda5c23d4d6349b697df9de7b242c5fbf807e8183320c1467d52c62bfd517d12` |
| 1 | selfplay | `selfplay/games/game-000208.json` | 164203 | `2e654a89bec84744cd89daccc1d97fa947aa65af4ab7ec56453cd91039a6417a` |
| 1 | selfplay | `selfplay/games/game-000209.json` | 437930 | `82e968fc87eb153906f7580d34e0198faafcab0840c820dca16f23ec4aad87ce` |
| 1 | selfplay | `selfplay/games/game-000210.json` | 200886 | `38cb592c2cda0243e9d6f881460cdc9d0788fb5a0610e13a96cb08adc7cec442` |
| 1 | selfplay | `selfplay/games/game-000211.json` | 200384 | `4f36dc1860b4cf1f4e17db00667f42c24196856507acf3f24c46fcc465afe869` |
| 1 | selfplay | `selfplay/games/game-000212.json` | 392572 | `45c15f1509b8c8c32beb6f70482527d2816c99f28811bd485091fb1beaca818e` |
| 1 | selfplay | `selfplay/games/game-000213.json` | 182787 | `d44c9128615ac923831e20a94d42c6c32ce8d1df3dd09fb0fd3f85f64967a5ee` |
| 1 | selfplay | `selfplay/games/game-000214.json` | 173377 | `1f350f057e7beffba5d3323494839b706e0407909d07e6588a8f6c11111837fc` |
| 1 | selfplay | `selfplay/games/game-000215.json` | 164864 | `ab4b55baf7cb03d2c8668dbd8a683b922d2536b04cdfc2edbe1d33b2d2d22c42` |
| 1 | selfplay | `selfplay/games/game-000216.json` | 330007 | `08923842cc9ccdcac7a8368eb0b404fbebc30cc4dcdbc1b7b79567b386c2014f` |
| 1 | selfplay | `selfplay/games/game-000217.json` | 218395 | `0e6768eee136364a4577fb5e235f38945a1c213ab4fda9a57be3c28555fca364` |
| 1 | selfplay | `selfplay/games/game-000218.json` | 234957 | `bdb54433e4635b081d422a1ee8eb75c5f73a585fc81f1e75b62a6da4da025750` |
| 1 | selfplay | `selfplay/games/game-000219.json` | 154943 | `2ebeb0ba8f1bee49c86e502470f0d55ae3696006fcf0aaafee8e50b68cabf2a3` |
| 1 | selfplay | `selfplay/games/game-000220.json` | 258958 | `5f42f9081bd2d53cc43901aeb268e201f3c7bcf6cc813c3dfa477b8f3bac820a` |
| 1 | selfplay | `selfplay/games/game-000221.json` | 155442 | `08eb8b793491502d8a164377f12af9d41958aabebead8d1c34285e2339826102` |
| 1 | selfplay | `selfplay/games/game-000222.json` | 154910 | `423f95f89fdcaf4502c90ec83c686d89202b02d024255df55496752bc3edad5b` |
| 1 | selfplay | `selfplay/games/game-000223.json` | 432726 | `b7e85c016645f07817a5305f70a906dce19af3abc226fca5c2909f0d0cec272f` |
| 1 | selfplay | `selfplay/games/game-000224.json` | 126630 | `11f867ffd4d5437990eb2d752cbe41d019108e6270072fdc75385d4401c59ed5` |
| 1 | selfplay | `selfplay/games/game-000225.json` | 301373 | `621259e0df65275b82443d7b00120353faee730605769ff4e50bc00c394ddbfd` |
| 1 | selfplay | `selfplay/games/game-000226.json` | 146070 | `75abb200c81b9d00877cd867bea3e80397f3f6dd2bb1832a6c5a3d317caee94d` |
| 1 | selfplay | `selfplay/games/game-000227.json` | 423996 | `41ad4a2a22e2311030ffaba13a8a0b240924f099732881e599ee22faf782b6ae` |
| 1 | selfplay | `selfplay/games/game-000228.json` | 154981 | `363511c69788e3ddade9ec6f399eaafefb65e85b680bc8adbf868a943532be0c` |
| 1 | selfplay | `selfplay/games/game-000229.json` | 244525 | `8ab1b872fdc195a1ee81b5f0f4cbfd7dfaf26cd30acfa339b4342456692f5bcb` |
| 1 | selfplay | `selfplay/games/game-000230.json` | 308164 | `6845df08060d62cdaef2deda7987179e3a1ba1d88ca122b23f47c4458923a844` |
| 1 | selfplay | `selfplay/games/game-000231.json` | 173442 | `ca43b66111cc6e82deff1cd3eaa034f16d7d464cf1b815f5b0bb6ff9308554d6` |
| 1 | selfplay | `selfplay/games/game-000232.json` | 126690 | `840a537126068706a66be404043cc7188bdc0d60ae2d3ddf9baa53b76fe01e93` |
| 1 | selfplay | `selfplay/games/game-000233.json` | 235569 | `812771325ba57b999cd627cda0700b4ab580c3e6f14f8d7156a2ecf685fd9400` |
| 1 | selfplay | `selfplay/games/game-000234.json` | 392842 | `1508fcce14c1d7a93773d4462206126bc76f49e4503ff880c32b4ff5138bf40e` |
| 1 | selfplay | `selfplay/games/game-000235.json` | 164261 | `828294818f6c29df59e4d5a636ac7fc2bcee58a131d1254b5178585b4efe4b74` |
| 1 | selfplay | `selfplay/games/game-000236.json` | 384115 | `5165182b3a5b8f1218ae442fc8a1eb044ff7eb14d0553f11a9b49917c0c5e23a` |
| 1 | selfplay | `selfplay/games/game-000237.json` | 259087 | `6ad8cc0089123798144817a0146d86d0adae89c3bdf525fdd1a1001477154194` |
| 1 | selfplay | `selfplay/games/game-000238.json` | 358360 | `6c9b0a4f6d0cada705b87d563fa396a16244d4b111951104b078c31841b60a86` |
| 1 | selfplay | `selfplay/games/game-000239.json` | 155159 | `cb59c57bd8e0c709508aa16072ce07d540481c07c2cc3faff3cdee418e0010e1` |
| 1 | selfplay | `selfplay/games/game-000240.json` | 235100 | `973ceb99d1a7adcb8b9c4819d31ee53b8ef7714010403d79be7ee12135a0c706` |
| 1 | selfplay | `selfplay/games/game-000241.json` | 126586 | `48b6f09aaaabc4a2f78710547c49e1836f29089a7ab4dd36cf9dd711f39cdc27` |
| 1 | selfplay | `selfplay/games/game-000242.json` | 164899 | `db38a01bb27dcc9976044fa56a348142c353bc7b4bc34229feed9d13f150255d` |
| 1 | selfplay | `selfplay/games/game-000243.json` | 465944 | `a4ab5fc4a65a7cf5ab2119ce923d61cfae5ca6b1bae64dca84c10321ca7578ac` |
| 1 | selfplay | `selfplay/games/game-000244.json` | 126478 | `1ea576d4a807b7269cdced8cb96ac16854689e8653131251849f0d2f6115f129` |
| 1 | selfplay | `selfplay/games/game-000245.json` | 458216 | `9573ceb8c554238458242c3602a7a58ab9649023156341db0b93f0ca44bc7023` |
| 1 | selfplay | `selfplay/games/game-000246.json` | 155088 | `b3d1932812f7f7366e62263d94d67a5410dfd08c0137136409d3c028376cc234` |
| 1 | selfplay | `selfplay/games/game-000247.json` | 266768 | `4efa613be67aad247c6c79d971fa9112d02859631915ce94be64be569088ddc1` |
| 1 | selfplay | `selfplay/games/game-000248.json` | 126501 | `0d877128c1f9bdeaae035bc0bd710b3254eba792a1572a6081054986b0a4f367` |
| 1 | selfplay | `selfplay/games/game-000249.json` | 145732 | `a1f833b1d01c4aa90d84a95bee09f0fde0a94488c3907b90d5e2753501781e63` |
| 1 | selfplay | `selfplay/games/game-000250.json` | 173456 | `dd9216dd1881747e2f5b8307954098a58bf25a77c04935da03dde40c564df869` |
| 1 | selfplay | `selfplay/games/game-000251.json` | 442424 | `c456a8e451af6085966e1cfd8cc0ea17cfe510c794d629a10569b5d68f219136` |
| 1 | selfplay | `selfplay/games/game-000252.json` | 146012 | `365877fd15b2428c2dca4b47b05ee10e276d0da08388042583ffa5fac4fed77e` |
| 1 | selfplay | `selfplay/games/game-000253.json` | 145997 | `f794a69638ae856822c7765cc03ce3efc35f65a0498c9eeef358501b82e7482a` |
| 1 | selfplay | `selfplay/games/game-000254.json` | 369042 | `e595c21264ac4ac3cb54d50b0c34c1719ff017d73e46b8169fd380988921473a` |
| 1 | selfplay | `selfplay/games/game-000255.json` | 126622 | `c2af2721eae5ba5dc168aba8b123b3ef4f677dce997e39e310d0591f58ab9609` |
| 1 | selfplay | `selfplay/games/game-000256.json` | 126694 | `167c7c408fb418494ba9deb36cf30d7586609784e1c2eeb7c3ca4cc1e218b575` |
| 1 | selfplay | `selfplay/games/game-000257.json` | 145841 | `eaef3d06b8ead07fd14b762c9e81c849062856e7052a824a0d54b4bdcd976bb2` |
| 1 | selfplay | `selfplay/games/game-000258.json` | 225602 | `1cec29bb8025246a612992bd8900a20be7be74cedd698d06e9bff103ed366981` |
| 1 | selfplay | `selfplay/games/game-000259.json` | 252270 | `a36fd3b2090a664bf6b98f7ec21730e1dd6a9e95bef29eeaa3b066195bb05833` |
| 1 | selfplay | `selfplay/games/game-000260.json` | 164414 | `ed5ec75f67be879615230035f8d8f6b17772e90cf67e1a303fa568531eb36308` |
| 1 | selfplay | `selfplay/games/game-000261.json` | 234901 | `6c099d6086038393d6c0b655b190b5371b4e32955abf5980bcbca1316347406a` |
| 1 | selfplay | `selfplay/games/game-000262.json` | 164653 | `a53377f9e038adc3583598b6cf7ebec25326c072747ab22e41b967cbcab1b3fa` |
| 1 | selfplay | `selfplay/games/game-000263.json` | 200570 | `a58ae54aa6b4107f5f757dbf7f8351602280b378675bf9d3d9931d215ece12af` |
| 1 | selfplay | `selfplay/games/game-000264.json` | 226057 | `ea21c003be8fd701afbc0326995e1361c8222d757442307290e0a5cd8bb2f23b` |
| 1 | selfplay | `selfplay/games/game-000265.json` | 218239 | `95cf7a2bbfbee4e26c205a51b90e8a7627522cc21817bca69c8aaeb036b69fb2` |
| 1 | selfplay | `selfplay/games/game-000266.json` | 164718 | `3fb5b9c3f446e4d4f7dafb04627565ab903eb29af1c237c6f6ef363e440cab4f` |
| 1 | selfplay | `selfplay/games/game-000267.json` | 155314 | `b65110781d05e85f0f8948ad58baa12089e2449c73cc6f9f09fa1f0fcc042a47` |
| 1 | selfplay | `selfplay/games/game-000268.json` | 369200 | `df71ef3e7abb3cea09196c4b58dbd7ba6b279daf01c4d1770fa96ea271b7ba11` |
| 1 | selfplay | `selfplay/games/game-000269.json` | 226643 | `421e9799e26e3a6a880d02f07ef283a1d89704c63eefa87effb1a200c20a3221` |
| 1 | selfplay | `selfplay/games/game-000270.json` | 164340 | `d30901c9ee06ae49a4f714c76a6d27a850b53a57cea0d6aa9b4bca40e72ed633` |
| 1 | selfplay | `selfplay/games/game-000271.json` | 182600 | `2f46f20365dfa1a5275622a9203e3f32e5473050f16985998594c4f9256a2391` |
| 1 | selfplay | `selfplay/games/game-000272.json` | 225607 | `e37aa49c5f37e6f3c4c2474e6c2a3686f12599df9b133f37382474d73f5339c7` |
| 1 | selfplay | `selfplay/games/game-000273.json` | 558665 | `d2df050ea8c8bd72d850e97ba54f61fd64b431a34df6719fd8c7605acdbf36d5` |
| 1 | selfplay | `selfplay/games/game-000274.json` | 146142 | `c8f1d131e91cc1b6e5acb3c3b94832788bd1a1c7a291ed311fee53b33edd830f` |
| 1 | selfplay | `selfplay/games/game-000275.json` | 251695 | `3f602abea1d851e77423a8ac591b6562f8b5760a726e1ac72953252231d8345d` |
| 1 | selfplay | `selfplay/games/game-000276.json` | 155103 | `47cd3b284ea62f2a9ddb59f6ea8198d45e1ff67ad5699e6a3e0eac3ff8be9bbc` |
| 1 | selfplay | `selfplay/games/game-000277.json` | 369808 | `815545d51866a630e7bad918385573ed585aeff547008c7c0d0efbafa8ebf8e8` |
| 1 | selfplay | `selfplay/games/game-000278.json` | 354322 | `72b03b98f25605fe6861ea38373405b312abc293146efd9a99328464a87626b7` |
| 1 | selfplay | `selfplay/games/game-000279.json` | 226616 | `5aa6e0c6c30175ca32e84876e6555c52d3372e7bb3fd43e9d897cc295a9c2cb2` |
| 1 | selfplay | `selfplay/games/game-000280.json` | 154972 | `758377989182f342684b46c2cc061ff4c843929710543854d1a39864bafe2825` |
| 1 | selfplay | `selfplay/games/game-000281.json` | 155081 | `931a701908a1b5decc699d6a6e81f05b160b8cb3e4cc61adc39e127230da6823` |
| 1 | selfplay | `selfplay/games/game-000282.json` | 218628 | `224f8ea84306fa4436a7c2a8b0e231168b6f8100672132aecbc2482d58c2500c` |
| 1 | selfplay | `selfplay/games/game-000283.json` | 145901 | `cb623d467edfe7efce04bc04ccb78f8d45481f9f6ec6c5fa1809e69cbcb9018d` |
| 1 | selfplay | `selfplay/games/game-000284.json` | 314533 | `a12207a938f183cb3605c8b07d519804eb4cfd7adce0867c058bc0ae57bb402e` |
| 1 | selfplay | `selfplay/games/game-000285.json` | 251962 | `708145f046161b7c94946c2a1f08cdc5fdc6b66ded9eec8aebfd8be78f1f8b35` |
| 1 | selfplay | `selfplay/games/game-000286.json` | 234786 | `69398c77c981ccd0630788b5d58ba4f61ad3d7d77416e2b0c81e9a519539fc58` |
| 1 | selfplay | `selfplay/games/game-000287.json` | 252677 | `d758ef7388787048382774f0f2976e0e832a13e48f5cdb188eb9ed8612946df6` |
| 1 | selfplay | `selfplay/games/game-000288.json` | 284358 | `9d5e151e4b9e397b120c3a754ab869e1a84a9c52dc662f28e4cc82334d8aecc1` |
| 1 | selfplay | `selfplay/games/game-000289.json` | 328028 | `c769130f6ed064daa90f170c1f3d078ed33444db01d4c72e0d3d1f98b3e51e69` |
| 1 | selfplay | `selfplay/games/game-000290.json` | 154883 | `75c49add6a40098e080dfeb1a61510ae9c9caddbf2d2f5d2c9e0aab8acd503ea` |
| 1 | selfplay | `selfplay/games/game-000291.json` | 297774 | `31c02c64a7d280592e3fb5ade4ca04255b324819e65e7d5c3265fa687033c425` |
| 1 | selfplay | `selfplay/games/game-000292.json` | 259105 | `d7b8254ba1e0e45be7f1dd5db846cbe9906262be48b2c656d38138487b754cfd` |
| 1 | selfplay | `selfplay/games/game-000293.json` | 389797 | `ceb72c223044f2bc6fdaad36ec18d679ce387fe58e5182539151afd73799bd53` |
| 1 | selfplay | `selfplay/games/game-000294.json` | 145706 | `16d6a60fa5eda9a5fb27e2d99777c359ce0e59e62691a3e0c60d2713ef21f9ef` |
| 1 | selfplay | `selfplay/games/game-000295.json` | 258578 | `debed901b24f275a42729540069a9e7887ca4c6ac7c6b3333cadd7afdef63263` |
| 1 | selfplay | `selfplay/games/game-000296.json` | 251604 | `b1297bd81b752fa2076cd2fdf182a516f1758b1dfc361dc935c99b73514b2598` |
| 1 | selfplay | `selfplay/games/game-000297.json` | 154904 | `69a902a3cd6d94393598853d75e45193716dc8195a525214acb5637d0e16da84` |
| 1 | selfplay | `selfplay/games/game-000298.json` | 200274 | `a29ced9e0ffc627e6911c5499bfadb6120051056937fc16184afc18ecfae80c7` |
| 1 | selfplay | `selfplay/games/game-000299.json` | 126739 | `8e7cac8785af74e8d72c066e2c546b3ab199aa754d1be177ca2a2d55103f5928` |
| 1 | selfplay | `selfplay/games/game-000300.json` | 329160 | `3fe5211261835843a33e05a53526ed82bcb235bcffcfc138e17b95c9ab71cf61` |
| 1 | selfplay | `selfplay/games/game-000301.json` | 173959 | `308faeab28d2789540caa1647bc8fcb2e7ff6eaeb7384c3194517a3efba0a91b` |
| 1 | selfplay | `selfplay/games/game-000302.json` | 225650 | `d7e60c8fee80cc3e68383e7a81beda7808eae297f730b020aff8d4bbfa9640b9` |
| 1 | selfplay | `selfplay/games/game-000303.json` | 145903 | `d3b30b3fa29824de865f7f6b74b0763ca27e334c010b24f2e7f3a3d21fb03267` |
| 1 | selfplay | `selfplay/games/game-000304.json` | 243359 | `e25c807ea300aa40633b02594b0b2814aea8e9331f5d81a369cad49c422bb8b2` |
| 1 | selfplay | `selfplay/games/game-000305.json` | 173605 | `22d322a9189cb49a98f98017bcac7c97f8334b670de0005410a1c7af47b25729` |
| 1 | selfplay | `selfplay/games/game-000306.json` | 200375 | `f2c0075ff695d5e94d5f7db4ede2dec6f07444aefc997b96344c7a96385400ca` |
| 1 | selfplay | `selfplay/games/game-000307.json` | 164525 | `5d0aa11dd1e0dd9e0216d759453204b1cdb6f45c3dc47b89929085540cb17115` |
| 1 | selfplay | `selfplay/games/game-000308.json` | 209445 | `2f70b39e0f0ae4be7a5545c05657c9d1305e7df3758d29c357ce326a86a1ee8d` |
| 1 | selfplay | `selfplay/games/game-000309.json` | 243015 | `91127c71512045c718342c127a8d551d71dccfc8cac22139f3ae26f3111d291d` |
| 1 | selfplay | `selfplay/games/game-000310.json` | 183385 | `f0c747793ae109e3310a40b849c0f32be85a825ddd357fc778eb90243162d1e3` |
| 1 | selfplay | `selfplay/games/game-000311.json` | 154899 | `27562c0d93fbd1fe4125cb92df431a1d2983512a6434f089d4ee96f41dcc9bf7` |
| 1 | selfplay | `selfplay/games/game-000312.json` | 252419 | `a92d4e239be53ee3924d61b12312ad6c23398d84b3142e3c7e0edeff0b2919ce` |
| 1 | selfplay | `selfplay/games/game-000313.json` | 250867 | `353b602f739b9be53faa711095dce3ff1db3451e0e1e91f06f2ada4ddbcd1112` |
| 1 | selfplay | `selfplay/games/game-000314.json` | 313320 | `5708127e46a8ec6fb358382dfcb41413f1e5b41e75c16ca9dbe823080f56979f` |
| 1 | selfplay | `selfplay/games/game-000315.json` | 155317 | `5f68dd8324008287d45041badc809b95788da7a28229222e5a92e02ae457ecab` |
| 1 | selfplay | `selfplay/games/game-000316.json` | 155139 | `7f61dc1d9aa0283452083c2f9adc9057242b5d09fe4400c7ec894eab6f3dfa8a` |
| 1 | selfplay | `selfplay/games/game-000317.json` | 155118 | `df072599ae6256124ddc1d30f5c39192cb1b303ed8b35bd6a2a346ec6ba19a35` |
| 1 | selfplay | `selfplay/games/game-000318.json` | 558565 | `3fbeb440df1c7353cf3f67669f29286a41b0853945f6f143c705e11c238d9cac` |
| 1 | selfplay | `selfplay/games/game-000319.json` | 226670 | `75823a67156429829965da252cceec3846e73a3bb469ed39db13c8b409562858` |
| 1 | selfplay | `selfplay/games/game-000320.json` | 573209 | `43b0cd8b709a19471947ddabe067fa8eb3f9931375c2797be86a0789af3bfbcb` |
| 1 | selfplay | `selfplay/games/game-000321.json` | 146056 | `b87188e7f7783290ae048369bd089c14e2c04120bbcf5846f19dd40a0cfc2f5d` |
| 1 | selfplay | `selfplay/games/game-000322.json` | 225783 | `0cab8a828903e8ee75870f544f066828486fad5627f963566643a6c65b22a08f` |
| 1 | selfplay | `selfplay/games/game-000323.json` | 154960 | `fc43c6af71489b0920c859722e1021bf96b2ab0f11324ae344b2bcdd3530faab` |
| 1 | selfplay | `selfplay/games/game-000324.json` | 291645 | `9e9d45d557572600d860a0f28a0a6206cdc4990064ef3d3348d767cc07574de9` |
| 1 | selfplay | `selfplay/games/game-000325.json` | 164698 | `437409dba60db4f9d9ccab00b0da4d314054d1613b21e185c38e99d6b041e024` |
| 1 | selfplay | `selfplay/games/game-000326.json` | 574174 | `f9a69cb527085fce483020a75c186ab4c845bb0e67e2396145ed3181a3956f6e` |
| 1 | selfplay | `selfplay/games/game-000327.json` | 155068 | `918bdd24169b7130255a6eed09f5bf380e98a2e6c677ce23f95f8d1f259f0c1d` |
| 1 | selfplay | `selfplay/games/game-000328.json` | 369096 | `c92a06d290e35dbe5b06069146bcc1fbf485a5bc6a253d06c54602035c93f394` |
| 1 | selfplay | `selfplay/games/game-000329.json` | 164675 | `4607343fa9b3152384f0a4836c885ae8f418918266d6cac463ae1b3becb630ed` |
| 1 | selfplay | `selfplay/games/game-000330.json` | 252973 | `0356ced334ebd30d160798d9cb40bcdabb39db5f180fa173cbd6b732b1b749d3` |
| 1 | selfplay | `selfplay/games/game-000331.json` | 200811 | `2a52142c20a94ac0dd51c8883e9911638bb3a7a1d0366005f888246be3db9bbc` |
| 1 | selfplay | `selfplay/games/game-000332.json` | 414224 | `fd6a1402e6efeb94730950a454367ade216975cf1c29f59732c114e751228794` |
| 1 | selfplay | `selfplay/games/game-000333.json` | 145706 | `8e8601e8a0f0de1f0a6ab0e689db967abfca00046e7f22279dc7896879bbd084` |
| 1 | selfplay | `selfplay/games/game-000334.json` | 164452 | `55a5a390aa343119f0c6437b0b37bf43a9c03d35d874e07668d3ff7c42fcde80` |
| 1 | selfplay | `selfplay/games/game-000335.json` | 154929 | `c51eb63126fdc0a061b42e53e37d72d09ea0542c33a8fea5fdd417225fa6d771` |
| 1 | selfplay | `selfplay/games/game-000336.json` | 155450 | `16219ea50418b4f7f21510d3a63f07abeb7e92430127d0331258f4de629adb04` |
| 1 | selfplay | `selfplay/games/game-000337.json` | 183640 | `6ca7f69a0931a0beffcf909d8187c16900e84255e98e02e23d152dfaf69f80d1` |
| 1 | selfplay | `selfplay/games/game-000338.json` | 145933 | `cfbf25912e28e4e8cd70248eaed4259625ab2956b90cccdb82c15eed1b67ceae` |
| 1 | selfplay | `selfplay/games/game-000339.json` | 173558 | `7951730f4c3f3c50a6d4a4a8d0405509fcd97e1b48e710c4b91b342e158e6b0a` |
| 1 | selfplay | `selfplay/games/game-000340.json` | 164644 | `15bc4f0dc82804f7f3609c488460da798a42c9c5db42a209833f1c38e536969f` |
| 1 | selfplay | `selfplay/games/game-000341.json` | 373626 | `775deb7c9aded4b54bd1f71625a219c76c98f2fcebcd2620a2f43e15e37ee847` |
| 1 | selfplay | `selfplay/games/game-000342.json` | 154949 | `e96ca94815fa4b17b29b682d16a1999acf92729fbdd5c2c1768c3ebcde35d71b` |
| 1 | selfplay | `selfplay/games/game-000343.json` | 218369 | `83b3022e15ed8a49ddb526d7f2f754d30dc029531b81e5ceb00d42ea5bbc6a24` |
| 1 | selfplay | `selfplay/games/game-000344.json` | 164678 | `8c2473c17e18d093f0bb790b8de42ac731a07296cdd773e158f7da8f1b939f23` |
| 1 | selfplay | `selfplay/games/game-000345.json` | 164284 | `3e78a69180cda10c56259bf31ee38e0d324533bb1ddfad69fcbd51fb6c12d48e` |
| 1 | selfplay | `selfplay/games/game-000346.json` | 146046 | `7ec8fe13a61d8d8dad74430f43ea23d8a8923861aeb507a056ae87ddf9a61464` |
| 1 | selfplay | `selfplay/games/game-000347.json` | 200413 | `19625ce10fd53987830beb1a099ee631a8cc54837ce12789ff21f341ef39b4f0` |
| 1 | selfplay | `selfplay/games/game-000348.json` | 415533 | `5da5b7b9e8104a424f21d2dc239393f124ffb2696a0eb4fa3ad96e9f7a4f97c1` |
| 1 | selfplay | `selfplay/games/game-000349.json` | 145977 | `bc2c0024920fc9498a5b9f5bc64d19c85cbc3d097161014c86e3af5505634c31` |
| 1 | selfplay | `selfplay/games/game-000350.json` | 126578 | `a9e1925c4d617cfe6a8c5d189ece9179049d873e2cd36dc18d43a9cbb22002f6` |
| 1 | selfplay | `selfplay/games/game-000351.json` | 328128 | `50fe5bf77edc31db27354941b650710e1169f8b6733ec24bd457e089a054cf2d` |
| 1 | selfplay | `selfplay/games/game-000352.json` | 380995 | `d47e0d00a5dc790c86034695f6acde4027909fc4ee2e03824b6497f4eb743f0d` |
| 1 | selfplay | `selfplay/games/game-000353.json` | 155132 | `1eccb80a6b580d46569aef78f590640327273948642d74c33e7c37436695d491` |
| 1 | selfplay | `selfplay/games/game-000354.json` | 164570 | `35d08613639513b0a1f71564151887ecd79723e46bf8fac9e3bed1167c54f823` |
| 1 | selfplay | `selfplay/games/game-000355.json` | 225644 | `463dac4578bc5e83c2af15332ec5989476a0b78e640ba127753427d62b0f5011` |
| 1 | selfplay | `selfplay/games/game-000356.json` | 164680 | `f98b8377c4d330d4c3a8d7299f34371b464116520961f22b40559ca3e130417a` |
| 1 | selfplay | `selfplay/games/game-000357.json` | 225616 | `f613292155c2b96a22625432c2daa1390357265d1a80bbe3cd031d22446bccff` |
| 1 | selfplay | `selfplay/games/game-000358.json` | 126595 | `bdc51c1110b9daae9fe0bf44717dc18540517311acc4ac27bb5160d66e4b6be6` |
| 1 | selfplay | `selfplay/games/game-000359.json` | 460269 | `1c46573519095d24dc083eda6af21a07bab82b33cd51eebca77b0fdc3ecb947c` |
| 1 | selfplay | `selfplay/games/game-000360.json` | 182646 | `9b3317d0b068e927c6c835b55470d1a740cf66ace095f485d8d1a0f84f4ed078` |
| 1 | selfplay | `selfplay/games/game-000361.json` | 379105 | `2e8b975fbc070d6e13c12e9dfd30029a4111e3428c3d4e96eb37c7a68968eff4` |
| 1 | selfplay | `selfplay/games/game-000362.json` | 201164 | `f4c330d072e8cc07636cf3952eae13273224995890577f8419c59ec6cf623b6c` |
| 1 | selfplay | `selfplay/games/game-000363.json` | 276699 | `a61b6de1d42d73594bd5e32051a175b8d6e037e47a5be4706e43d74becd150b3` |
| 1 | selfplay | `selfplay/games/game-000364.json` | 126512 | `826c55a6995641d629275425e1e486b8c5afdca648752bf184c6295c36222f79` |
| 1 | selfplay | `selfplay/games/game-000365.json` | 154914 | `c02016822c2061d7c6834dc25d68f01b61f74a05893993197c82703f2de86d21` |
| 1 | selfplay | `selfplay/games/game-000366.json` | 528294 | `a8cc9c2a521c3f48ca88d0d03bff881e85f6115582a1cec561a19d95920f373b` |
| 1 | selfplay | `selfplay/games/game-000367.json` | 183330 | `aacf2b3e54da6b769fe8297f9f01ac2fea80fcdd4396e6413eff922b386c1956` |
| 1 | selfplay | `selfplay/games/game-000368.json` | 164321 | `922dff6946eb6e35a68b763a622012ab7299bd805a1f624e05ae3c0b40e24369` |
| 1 | selfplay | `selfplay/games/game-000369.json` | 268879 | `69fbc52ab8a3535fa9fcf9db2ae73196b50b93bf51e4cfd023b331afccf474a9` |
| 1 | selfplay | `selfplay/games/game-000370.json` | 155081 | `ef8f4cc950bb3b2c67fcb246d220e2d75bac741cbb750832dfa3dc63f85f43d3` |
| 1 | selfplay | `selfplay/games/game-000371.json` | 284088 | `dbb119f47977a5ff200826de12db9b27187803d0f73bb67770c099cb6e22f9ed` |
| 1 | selfplay | `selfplay/games/game-000372.json` | 154930 | `ef0fdd4809cd0bc28b800febdef4c4b9fbc63c197b840f17c40ed85fe3d9d72d` |
| 1 | selfplay | `selfplay/games/game-000373.json` | 405955 | `d12f735637080520532a771d235c6ce31a813809dc72d3082d4333ca4a60db6c` |
| 1 | selfplay | `selfplay/games/game-000374.json` | 182744 | `23b8ef78032470fffde96a8f437c1b69ed451926e7fbfec31f81394796fb3612` |
| 1 | selfplay | `selfplay/games/game-000375.json` | 268736 | `e504458136f6f50f16c2b63d08a72b4ce58b065917d5b97d35b50eaf58d036a3` |
| 1 | selfplay | `selfplay/games/game-000376.json` | 284980 | `b7ab7d3140fe2a9a3e11ce3b9d60c74bbff9115001ab5d4a601406b348d97a5f` |
| 1 | selfplay | `selfplay/games/game-000377.json` | 126606 | `c0847e6e49b2cd8c777aa95a3f2f135b4d0822cba1b0ae373f6c40b8f7110584` |
| 1 | selfplay | `selfplay/games/game-000378.json` | 145832 | `58be7327376a5494ee065993aaa979cbd0fed457e0c0cf48a94cbfa7b4804f7d` |
| 1 | selfplay | `selfplay/games/game-000379.json` | 564722 | `3776846ba790e87f71c0bbb8e2d40a4be8d02a5a334b7271dbb23ad6e6781e7b` |
| 1 | selfplay | `selfplay/games/game-000380.json` | 164679 | `3a36f1fe1635e04c211c83a7e8a0a2777a179fd22f69ede4a56ae3c218640066` |
| 1 | selfplay | `selfplay/games/game-000381.json` | 126689 | `68b7ebd6a2bac70ed16bb2dbab1e7de84a99d1335b0dded9b0b4735b2c2f30b0` |
| 1 | selfplay | `selfplay/games/game-000382.json` | 284307 | `03d1ec8d4eeb9de3041a0c73c09e97fb062790efb442ff79bee0e37ce94ed9b6` |
| 1 | selfplay | `selfplay/games/game-000383.json` | 182470 | `90453e065dead3cccf2842a0bb7a69d6785c8475e5e4a4ae7908740ed569e4b1` |
| 1 | selfplay | `selfplay/games/game-000384.json` | 268518 | `cd4edf844fde46bc9fa36ee61ef1f1fdaf8411201bad1603a2f3f11623f63d9b` |
| 1 | selfplay | `selfplay/games/game-000385.json` | 126651 | `deca2d650316513cd314d057a3f6e08de43dafb2c63302e64ba88b4aa214dc3a` |
| 1 | selfplay | `selfplay/games/game-000386.json` | 276149 | `31215393c6c293d93d07972a0e32605a8ee35a71c06f055d193b71339ef3c2d2` |
| 1 | selfplay | `selfplay/games/game-000387.json` | 154905 | `9567cdafcc84a2f2087f0265540ac4b6fdab07dad8af68f77e37e6d9576c162c` |
| 1 | selfplay | `selfplay/games/game-000388.json` | 183414 | `db73412d2c43c90b6fde014e0a6a5151cdc42ed0da9a2e1540d21138d00dd68d` |
| 1 | selfplay | `selfplay/games/game-000389.json` | 183256 | `cb55a93770d3e66e261e1902795cb01906776b8ff7acadbaf8c06574c6aaf65a` |
| 1 | selfplay | `selfplay/games/game-000390.json` | 183048 | `ccecb03f4af030b7d7adae8cab0a34df53dc0b9f89c773c5271dfab36d2ca691` |
| 1 | selfplay | `selfplay/games/game-000391.json` | 380737 | `9fcc68fe5faeccd24d2fe4c3d4626607a615501aa8670b6939a96a9efe585859` |
| 1 | selfplay | `selfplay/games/game-000392.json` | 200855 | `484f4a643ecb667dd8e399afd766138dfdd8d2d6111bb817a99acdb8755b58e7` |
| 1 | selfplay | `selfplay/games/game-000393.json` | 252377 | `9fcc8e9c7e928ddee04e9b3cf1c67413876cfb75bf9fb76b3c26e30a6e10d036` |
| 1 | selfplay | `selfplay/games/game-000394.json` | 209208 | `7326b8033b5d48baf0ab580a43a5d119a09a05e54571cbde502accc2225b4ea0` |
| 1 | selfplay | `selfplay/games/game-000395.json` | 183472 | `97c1a504c34fbcc502d1ebf391bd67bb5bf9ce8ca2041e3d0d36a005c0de061e` |
| 1 | selfplay | `selfplay/games/game-000396.json` | 164937 | `2ace0089250fb2c5a996e70f063ae69d0b5e732a89ca4bc568368f5a966294ab` |
| 1 | selfplay | `selfplay/games/game-000397.json` | 253064 | `100a05d6962094e2163b119eeda868500957752a8764724d54afe6fa6416fd07` |
| 1 | selfplay | `selfplay/games/game-000398.json` | 126582 | `b08aaad6ba956de85fe6dc18ba44fcec48ad96e9055502f9d2daed68b721ffb6` |
| 1 | selfplay | `selfplay/games/game-000399.json` | 126652 | `b54486eb6dbb045f5d98e469b484f01c2218152adac6259f35aaea5ac4274036` |
| 1 | selfplay | `selfplay/games/game-000400.json` | 208793 | `752bde0b34771e9a4dbd6ad91956202cdfd70e8794aeb9173bb7bc50471a36d7` |
| 1 | selfplay | `selfplay/games/game-000401.json` | 252262 | `3ccf0369f509d6a18ca0a59564fa5cf6c5e4062b6fb9ef2b0742d4226d8bcc89` |
| 1 | selfplay | `selfplay/games/game-000402.json` | 208657 | `125c81c1249c8d0e7d088c7c91b6bc9b1d8ee7a71d047194f4db834fafff8714` |
| 1 | selfplay | `selfplay/games/game-000403.json` | 566230 | `bc49c8eff03b7bee57c914957aeb6b84c43a59f3da8b118545fbcb3cff170d27` |
| 1 | selfplay | `selfplay/games/game-000404.json` | 145823 | `a6e648dfa9ff7e58168160dc1854d8dc216d6ec4e2c79238bb3b2cadab1a5119` |
| 1 | selfplay | `selfplay/games/game-000405.json` | 154892 | `8d1d9148bbdaf31027cdb9bed278de9ea02984e41c8fe7d248e1c4f168e7827c` |
| 1 | selfplay | `selfplay/games/game-000406.json` | 164927 | `094ebaef71ff92d45a4c72ea870974591a2b57d2d303b19ce1f6c83c6e6b1dc0` |
| 1 | selfplay | `selfplay/games/game-000407.json` | 341132 | `e7023b3077b92e8387152d3ca6838d80287f1e32c49df481c361a818ad93cda0` |
| 1 | selfplay | `selfplay/games/game-000408.json` | 218259 | `6e1dec325bd6b147c9d07541090717f0a6a594d9aa230fe2505d9000bdde8c5a` |
| 1 | selfplay | `selfplay/games/game-000409.json` | 381049 | `434b24dcf1fac0992db3c1847daefb968e0c6fe9c7a0944583de43f747db9860` |
| 1 | selfplay | `selfplay/games/game-000410.json` | 251140 | `e56417cc07e9fd92903f2cd1241febafca1859bdc7bdae07438ac021974338ff` |
| 1 | selfplay | `selfplay/games/game-000411.json` | 173416 | `774105db3231bfb2e6464577addb97e29636030c018809e4841272657e9fccb5` |
| 1 | selfplay | `selfplay/games/game-000412.json` | 183308 | `7060339c9929532cd38b8c937f4d17b0b277217d1594e6695ac480f913de6a72` |
| 1 | selfplay | `selfplay/games/game-000413.json` | 284068 | `c64d42fe9f44876fe4e4c67fae1d598a79769491a9d407200ad0f2e1ff12d1f3` |
| 1 | selfplay | `selfplay/games/game-000414.json` | 426475 | `9b90e1830295cf605f38761070f2c04f3c15831d048502b65637284ecfeb019d` |
| 1 | selfplay | `selfplay/games/game-000415.json` | 191483 | `937e9193e820a11dddcc9a10f568541be1e1182360bad34a64cc32f11af027c3` |
| 1 | selfplay | `selfplay/games/game-000416.json` | 146265 | `ee3ba52ac6df926c3bfebfc3c07945ee80c5630bcb9b5604605e4560d74a9bf5` |
| 1 | selfplay | `selfplay/games/game-000417.json` | 155086 | `b2297dcba85d6519b03c0a4130ca28b215652c1526d5b02ef3d3c21c21f2acfe` |
| 1 | selfplay | `selfplay/games/game-000418.json` | 126678 | `cd0e3f1bd82223b06d707fdd7bba195ebe4eb783d9da8f3d47b1971e4f400edc` |
| 1 | selfplay | `selfplay/games/game-000419.json` | 154967 | `a857a499acd71c41985cf328248cbf53b2bb3b6fcbbb859b549bd771d7c271c3` |
| 1 | selfplay | `selfplay/games/game-000420.json` | 329392 | `97e31514bc1089307f91273f53108f5daa979e4f9015b9275a246ea651b15655` |
| 1 | selfplay | `selfplay/games/game-000421.json` | 145975 | `960a5929659729537a8cc5fd2eaac9fed1b9f60bffd2b7243d4c8c9d52d27d8f` |
| 1 | selfplay | `selfplay/games/game-000422.json` | 155080 | `1b56c5bdeeddef04257988b063eeee27c9f65213f93a5cc0c8f0c556fcad77ec` |
| 1 | selfplay | `selfplay/games/game-000423.json` | 200792 | `fee0b0f75d36e04055be9a428828624dd67626ce92d4c6bb3f943219ce674e6c` |
| 1 | selfplay | `selfplay/games/game-000424.json` | 234522 | `ab941cd130c7262cea9b73c57e5024ef6ae49b77cfa1117e949bb74c392e4fa9` |
| 1 | selfplay | `selfplay/games/game-000425.json` | 126608 | `4bcb70ca91a959a5c77fb009c5d17f9191a085e5820603fe0a3393b63a8def0c` |
| 1 | selfplay | `selfplay/games/game-000426.json` | 182673 | `a54c13aac06328f2d50bd66b9e7c77d013d6100d7a8d950fb66eb6660fe60989` |
| 1 | selfplay | `selfplay/games/game-000427.json` | 155301 | `44d8ffc2f07b7aa836954c257b3b7b017f84508b1ac83f8b4afe91c980f12570` |
| 1 | selfplay | `selfplay/games/game-000428.json` | 558324 | `bea6a350127f1c06789dfced2e0581774f94efda90b95b49d794e732042b5b05` |
| 1 | selfplay | `selfplay/games/game-000429.json` | 164680 | `6faa666587c9d161d6947f80a56b25daebef67f156d9453a04ffd7b7dcc5711f` |
| 1 | selfplay | `selfplay/games/game-000430.json` | 242409 | `f64cbe220a73382ec7cf5239fbcc0539829e8289eb8a7718bab4d8bf8d96a66b` |
| 1 | selfplay | `selfplay/games/game-000431.json` | 218951 | `f0bae4f07d2aa295bea8acb3553c1493f674f3c6ccaa06c148202227063314be` |
| 1 | selfplay | `selfplay/games/game-000432.json` | 380289 | `55fc19ad95e014a673c78694b8d7679c3b18cbeb5471c8199f2264a2872c8949` |
| 1 | selfplay | `selfplay/games/game-000433.json` | 285984 | `7b94566ddc2d3e707d333d3002db88a85b04a6e03d762ccc03f1fb1675d0c3c7` |
| 1 | selfplay | `selfplay/games/game-000434.json` | 291646 | `cea13d3df3a0eaf18b5b6a7cb49d3ba23730d8cd51f9c371b918b3bb31922f6e` |
| 1 | selfplay | `selfplay/games/game-000435.json` | 146083 | `2af62af6360598e2161186493189363cd35fbdf8c224bd1a24d281c765ed0cc7` |
| 1 | selfplay | `selfplay/games/game-000436.json` | 164565 | `1b2c5919cec61bb7c1f4c0cc682f9c69095c2a300cfdfe47a88216d40e76238a` |
| 1 | selfplay | `selfplay/games/game-000437.json` | 155302 | `38398ea139e4d90366fbb66562a8529922b56b4bad762b8aaf731a792c2203d2` |
| 1 | selfplay | `selfplay/games/game-000438.json` | 431331 | `19f03b7dc864154174bcc9786e7e08139df555c56eface75853f6dff96e1eb98` |
| 1 | selfplay | `selfplay/games/game-000439.json` | 155099 | `d4b46446793d79f31a30fce3bbca04c9caf9fb448deb336f3b622c5ac84254d2` |
| 1 | selfplay | `selfplay/games/game-000440.json` | 182598 | `2f7b1cc39e34dddaa5dfb678f3699f67e3a0f3ba84afcca89d2b9032ae909f26` |
| 1 | selfplay | `selfplay/games/game-000441.json` | 126631 | `fc3fdbcbb1b9e1eae48ebe7ebcbdc3c459f7a51a315f74cd26927219d42564ec` |
| 1 | selfplay | `selfplay/games/game-000442.json` | 465925 | `b295afb16810e727ed6c72d3952198903aadd8fd9f2628bab000a4bfc762919b` |
| 1 | selfplay | `selfplay/games/game-000443.json` | 268545 | `d5ebfaca74e939e1adad77d20376832327f86a34bbc51ef09b269f7759191327` |
| 1 | selfplay | `selfplay/games/game-000444.json` | 164401 | `0e6fe0aeacb45e38a734aef6abe03c68779888aa893db1363fc04e312b16daff` |
| 1 | selfplay | `selfplay/games/game-000445.json` | 164508 | `0a77bd479b90800d183eca32b21f75cc0c5d03385a57a9f774fe4f0aa5d01517` |
| 1 | selfplay | `selfplay/games/game-000446.json` | 292274 | `92696048a6521ef1d798b5fb7e354b8df726e9e3dd2003e415f9075b3a284610` |
| 1 | selfplay | `selfplay/games/game-000447.json` | 173525 | `ab3d8a4b1f30d5cf1c7c2f730b1098ce6551d92ce35d0cd18df4ca2c90320770` |
| 1 | selfplay | `selfplay/games/game-000448.json` | 173568 | `be612b34271b3976ba74b2456761b5ea0d63d5a51a068b07af4167f3e216af89` |
| 1 | selfplay | `selfplay/games/game-000449.json` | 493978 | `a97a9832b889aa488309cfcebdf35c19f850e373b53d2faf6df6cb8d0af48bda` |
| 1 | selfplay | `selfplay/games/game-000450.json` | 566765 | `f8994a7062fdf1e93a833dc29bfc1ba8aaa21f5977eae3d54c852bcec7e20bea` |
| 1 | selfplay | `selfplay/games/game-000451.json` | 415957 | `291a537501bd29d5a63383416ffbfe018c9d62912ea2993a28108f59ca387521` |
| 1 | selfplay | `selfplay/games/game-000452.json` | 268464 | `873a13d0e995127f362a79b8e48024699e9afac00f731eaf35957d91ef3c0a4b` |
| 1 | selfplay | `selfplay/games/game-000453.json` | 218025 | `0e2b367dd86a0230d236cbdf31ebbc7765339560e32cf7a906a69e5c836e9293` |
| 1 | selfplay | `selfplay/games/game-000454.json` | 209230 | `1a2dfd2454eab880926fcb0c7ab583646d10925b46e413221ee349089837ece5` |
| 1 | selfplay | `selfplay/games/game-000455.json` | 183496 | `ab6d5554748f4057e2fb8d548b3a1d465d4e98180cbead1944dd1ac97dd9d1dd` |
| 1 | selfplay | `selfplay/games/game-000456.json` | 200312 | `e34fa87575f718a58799e2e92c7131a8384eff62c99be8f9b25a63c22fb61963` |
| 1 | selfplay | `selfplay/games/game-000457.json` | 126475 | `e13d59131344185f6138cc00e2814c890d334d856ce52f2a3240d92abbcb1cdf` |
| 1 | selfplay | `selfplay/games/game-000458.json` | 155242 | `f262c874bf31d4e0de5980adbaecb7e38324a87a486b01161f801a69280ce473` |
| 1 | selfplay | `selfplay/games/game-000459.json` | 521671 | `1b4fc658453d57d34e2e86dfbb19da1bd53cf44f0171c17620863f6646b3dab2` |
| 1 | selfplay | `selfplay/games/game-000460.json` | 164792 | `e679e8556069720f5e2dedaa59057d4d2ea4f42f0a9a09db6dd3ed9c31627181` |
| 1 | selfplay | `selfplay/games/game-000461.json` | 226140 | `db0ff0283ee2c52b789a46d578043b7051e2ab278586000df44c1ab5de126180` |
| 1 | selfplay | `selfplay/games/game-000462.json` | 164529 | `f397fe5b709c8f5eb43c5dc090a99cf1db614a7ae26f20dd162c15958afb4294` |
| 1 | selfplay | `selfplay/games/game-000463.json` | 145814 | `baf1900707a061ea4648b767b897aad8c823c209221ef24dadc27d1ab1c9b5c9` |
| 1 | selfplay | `selfplay/games/game-000464.json` | 225821 | `6a7652c279cb8a9504af252897e98791ad668d7bb8db4b876409547321865342` |
| 1 | selfplay | `selfplay/games/game-000465.json` | 200910 | `8102174ac5dab441498fb3af3e689009e4bdb8035d58f9e5a3c54b16d94c7513` |
| 1 | selfplay | `selfplay/games/game-000466.json` | 126468 | `9f919ddaf243837c00bbfe7dfeb489f2a2ab91c24fcbbbcd68239b38b391c70b` |
| 1 | selfplay | `selfplay/games/game-000467.json` | 290975 | `ab3d1229df4fd704ab672530418197e0db6869dde050bac3d23f90d809d53dc4` |
| 1 | selfplay | `selfplay/games/game-000468.json` | 126623 | `d04994681456f75ea658407853676972b034ba083d6d15b24472ecf38aaa2238` |
| 1 | selfplay | `selfplay/games/game-000469.json` | 276159 | `27c34163bd325d6f040c5c78fe121b8c234fb2af289606e90c076086c66aea6a` |
| 1 | selfplay | `selfplay/games/game-000470.json` | 126614 | `936e45c7b0b4ce6ef225ebc6334eaf331205f9ecb9a33792bea4c496060f559d` |
| 1 | selfplay | `selfplay/games/game-000471.json` | 331671 | `9363915197deb69c754316307ec7e4953fc1f8945ff0935ec4f7a785da69c163` |
| 1 | selfplay | `selfplay/games/game-000472.json` | 145801 | `2905b8fe12345745e3409f3ebf18c311ea03020ebdf0111bc43ad268102d9a0e` |
| 1 | selfplay | `selfplay/games/game-000473.json` | 146043 | `e7f7ad5cb6f4e584bcc01d37e30bbad9d139e28cdeea7bd6589ba2420067bb7a` |
| 1 | selfplay | `selfplay/games/game-000474.json` | 154935 | `eb4e8fe33513167c054d79bc9004cdd6baeca71cb25b1f2793e49309c68be443` |
| 1 | selfplay | `selfplay/games/game-000475.json` | 154902 | `6ef528794ff67e1eee0b4593b8440d943fdd1cd1f13a9090ce80ad12c882731f` |
| 1 | selfplay | `selfplay/games/game-000476.json` | 126507 | `f70aabfa4ac9214f3775b3a0658d11f3cf83f7e88a26d9634462e116ed46910a` |
| 1 | selfplay | `selfplay/games/game-000477.json` | 145887 | `0a4591a2040343efe8927c15bb60fccce0d3e9b129be099ed182a88b41d7f830` |
| 1 | selfplay | `selfplay/games/game-000478.json` | 154952 | `a00256933cd59fd9a11e9097097ad235b1aaef17e4ad2b1eb38475de812d6863` |
| 1 | selfplay | `selfplay/games/game-000479.json` | 126637 | `b7fece2375aa1694a5f376ead0e81b26a55d8b9da15a8f1f9b9e2557528f69c7` |
| 1 | selfplay | `selfplay/games/game-000480.json` | 126443 | `bbe72885a5c79496514d3f381443a83a94a56bceb6788831ca3f3ea85afae953` |
| 1 | selfplay | `selfplay/games/game-000481.json` | 173459 | `a1ebc0586c4ce51287f5f62eebdd3ed777537697f9f108fe71aa49f69b84dc21` |
| 1 | selfplay | `selfplay/games/game-000482.json` | 145918 | `cb008fe0751ae02dff9edebfb714b1db0ad65ff1e238d6d394149434ec1306e2` |
| 1 | selfplay | `selfplay/games/game-000483.json` | 126570 | `077ccbfa1f8418108a2896d796308c9abfb794fd37acc116b9c332b96851e52e` |
| 1 | selfplay | `selfplay/games/game-000484.json` | 404260 | `473187a836a78b8fdc01c7ec61bc49937e5b90c96afb0f4fd397a6681cbd981f` |
| 1 | selfplay | `selfplay/games/game-000485.json` | 154879 | `83718ad0bef65331a829f94827bf1b6b8cd20f550d85c6be9c8743f2a018d554` |
| 1 | selfplay | `selfplay/games/game-000486.json` | 155182 | `c05aca171d753c26cbedc3b63f326f981cfa472f1889882baf2260eb8a7c66f8` |
| 1 | selfplay | `selfplay/games/game-000487.json` | 314531 | `49a9bba351cfe08e980603ea2359efeee69d61688148103c878004e9d4150e99` |
| 1 | selfplay | `selfplay/games/game-000488.json` | 284240 | `5853a0fcbfd020269828fa8ded2200be4a355a11a90cde1b4249fd313d753050` |
| 1 | selfplay | `selfplay/games/game-000489.json` | 155126 | `a28adf86b51146b0e4bca6d3879a19f5cbe0dab56b8421dee77897eb937943a8` |
| 1 | selfplay | `selfplay/games/game-000490.json` | 173454 | `1e31d3eb30b22d91ba00315a387fded9c825a3d9e8b42ead85af94950aba0a9e` |
| 1 | selfplay | `selfplay/games/game-000491.json` | 234978 | `8a948c41f43633aa37064c252aea627adc8b600c705e8d28b6a7ae655211552a` |
| 1 | selfplay | `selfplay/games/game-000492.json` | 145764 | `00829bf5bb45e36d8be8ba798393566b03c7cdf8b568be8e69a017347cfc594a` |
| 1 | selfplay | `selfplay/games/game-000493.json` | 578134 | `3eb9e585402ad3f593e220e6fbacd62af2d9cb06f2cdebcd5a96963de67d8420` |
| 1 | selfplay | `selfplay/games/game-000494.json` | 126632 | `9d0a22f9abd10a2ad7a76ca31e4454458eaf2c0c5daa979bd75161a18f4adf31` |
| 1 | selfplay | `selfplay/games/game-000495.json` | 145643 | `05ec060d8080e03bb43b2a47215f5a32c1f784cfd649fdb390988300a7e484ba` |
| 1 | selfplay | `selfplay/games/game-000496.json` | 499283 | `36295091eefd217b67eca6204aa8e764dc62eb6530d7beae706171656e859591` |
| 1 | selfplay | `selfplay/games/game-000497.json` | 173654 | `3da7af472bda090ac2e4437264c468cf9b8e7c79bafc251fde7d90f31e691d13` |
| 1 | selfplay | `selfplay/games/game-000498.json` | 200659 | `f1f1e1408554b98197033825af6d8ed36616a6006bbbd83f1da66cb7e094ab24` |
| 1 | selfplay | `selfplay/games/game-000499.json` | 173441 | `b488cf5488cdc49b2aa01d9da8741a02651cacee2cbb3c7e9e4b822d9b183420` |
| 1 | selfplay | `selfplay/games/game-000500.json` | 252374 | `73da58a0ca8c8448188e0288c4acd060bcb90b8c6058603d72d3e657cce43dd7` |
| 1 | selfplay | `selfplay/games/game-000501.json` | 164608 | `ef976654da3287b3c0dbfa5ade8fa65e8ede2463326ede52001ab40fe2311f44` |
| 1 | selfplay | `selfplay/games/game-000502.json` | 155454 | `294412290170e2e498946f98c6a28d5f6a12d35089f27b56a7e2b2dfee9f4075` |
| 1 | selfplay | `selfplay/games/game-000503.json` | 226064 | `8f01cf38972a47370c9c92374fc62dec1718e3b4c6e6a45cedba63b7a7257a82` |
| 1 | selfplay | `selfplay/games/game-000504.json` | 146039 | `874567b20d7b914e4e602d2b3e314ee61412fa7ceb4cf24cd32aa1e94b6978ef` |
| 1 | selfplay | `selfplay/games/game-000505.json` | 355680 | `f48d2c257e24f1344391676e57f14c7669fdcb2fcd8806d05e0769a7c09fdf7c` |
| 1 | selfplay | `selfplay/games/game-000506.json` | 126721 | `57bc62f49861ac60e7a8d521dcd7b5f755768a164dc879daf28ce7dfb85b9255` |
| 1 | selfplay | `selfplay/games/game-000507.json` | 126603 | `b6d995e567282228a58c535923da1b1c6129d0daab499ca48342ec2ea41996a8` |
| 1 | selfplay | `selfplay/games/game-000508.json` | 154977 | `fd9c8ca8534d36ede586588463b3cabe4a09ea7f0901a658b7d581003da8d6a0` |
| 1 | selfplay | `selfplay/games/game-000509.json` | 326110 | `533ecdf7db176bea5eb04ddd9bc4fb469d1104bbb3313851a050676687db299f` |
| 1 | selfplay | `selfplay/games/game-000510.json` | 225783 | `6982cc4d8d9b2e4dce535e266a57ab4307fb58b4e629b3d54b992e68b1c2f76e` |
| 1 | selfplay | `selfplay/games/game-000511.json` | 252125 | `1456c8bd11cebfadec3f89ed3b51b1ae8ce5232221dc496a0f7d305692654a70` |
| 1 | selfplay | `selfplay/games/game-000512.json` | 154928 | `18ebf32cf50bdbd5fbbf301b8562ccbb26d1cc96f1c5d43753975210e9bcdd8f` |
| 1 | selfplay | `selfplay/games/game-000513.json` | 251802 | `2bf6451cf22bef181c8f8c6a7193f413a64a0316372f9674bcf3981d0c26ca83` |
| 1 | selfplay | `selfplay/games/game-000514.json` | 126595 | `bbb084c17b0182fa3511505663289d5e82e66b203b6f3eefb568d8402c66d83b` |
| 1 | selfplay | `selfplay/games/game-000515.json` | 572631 | `09f415d08e358cd80d451e500162dcd4f3bff659e258545666a31ca2809f856d` |
| 1 | selfplay | `selfplay/games/game-000516.json` | 164546 | `800290becb4ae6e08028bc66d21cabf27782613be865668362dd410a7efc1d2b` |
| 1 | selfplay | `selfplay/games/game-000517.json` | 183396 | `290619a332f627b25a2149f46e5d0cb406b56c6df6ec87269ddab9e18b00cbfe` |
| 1 | selfplay | `selfplay/games/game-000518.json` | 425301 | `5758847e60b46cb8d9066a2c4621f380954398b23f6d23613bc3fdcf33a5df75` |
| 1 | selfplay | `selfplay/games/game-000519.json` | 368689 | `35dad337a8c1ec7a63093ec8e8a81e09c35d30d58fdcc7767ec2b4072daf86fc` |
| 1 | selfplay | `selfplay/games/game-000520.json` | 145834 | `01e59c88f6aa4a42094e22eb223c53c201bc2b11e3682700c2c86faa2624d7d1` |
| 1 | selfplay | `selfplay/games/game-000521.json` | 234812 | `5be619d3ea8ee6b6545aa2b0ed6551254475c8f8f0a2f7a2ee906ac62eb940f7` |
| 1 | selfplay | `selfplay/games/game-000522.json` | 126578 | `1cbfeae2b92c8b66062ec3f2649b9e41fd8f4a8bacdc3a495b4f2731718da967` |
| 1 | selfplay | `selfplay/games/game-000523.json` | 209834 | `9ecfcd22e577d343f8c6ab1a948ea945230dad5118e2eee6baca12e12cad10ed` |
| 1 | selfplay | `selfplay/games/game-000524.json` | 218508 | `178ac50b5b015d268381702e9ec316ea9c6e681a51442b0aa9b2bb643147baf4` |
| 1 | selfplay | `selfplay/games/game-000525.json` | 209313 | `ed405f75f1cca2c9a63bdee2d608f36268a6c0f56b0c0671f22cafcf99afd5e4` |
| 1 | selfplay | `selfplay/games/game-000526.json` | 275541 | `156d56cff914e569087380b9e4f5bd75afbb8b09e02286634427a90fc31f5cad` |
| 1 | selfplay | `selfplay/games/game-000527.json` | 406764 | `503b070752582ccade4a9d204865dc55cebdd87ea6cd6373325336a93be4dae0` |
| 1 | selfplay | `selfplay/games/game-000528.json` | 155149 | `974da3e0d2945be959065942362ddc7ec7985847ee066a03525ddc55fdc51404` |
| 1 | selfplay | `selfplay/games/game-000529.json` | 145765 | `6a11f6e0fd073aa783dfee385a14ca0dfa4959c10dd3757e218a626659a5aec9` |
| 1 | selfplay | `selfplay/games/game-000530.json` | 135978 | `5e8900348e8cf46c31341e95366ce78c3d14e25d7348ee1fc6e1ddead0a61f4b` |
| 1 | selfplay | `selfplay/games/game-000531.json` | 126586 | `e2132867f2e964dd4395336741850ca84f6c6ba219541f5912efe1955f1dda78` |
| 1 | selfplay | `selfplay/games/game-000532.json` | 218331 | `4088de30b03208ba34193451e0a1b9619e0b642c72464e4d6e2c00eed36d8849` |
| 1 | selfplay | `selfplay/games/game-000533.json` | 164808 | `eff61870b40ca0afa9a409e6284531aae6f6f5bc202220c343ca7ef57efd2dce` |
| 1 | selfplay | `selfplay/games/game-000534.json` | 126645 | `fbd1c192a200689023e439d70ddc572d30fd0d2a6896f4a75a345d21cfce0fb9` |
| 1 | selfplay | `selfplay/games/game-000535.json` | 126616 | `95bde3316672b1721d92ea9615745d5598d3a62c50c9240b7556b73d33c2b919` |
| 1 | selfplay | `selfplay/games/game-000536.json` | 155089 | `50a24015ee9a4235ec28050c6ff000484a13587c8d27776f5997460c1d2c32e5` |
| 1 | selfplay | `selfplay/games/game-000537.json` | 218818 | `05d56dc959ac3fd65f3bf3196b1e4534a593860da253c3a57c22f09108219b53` |
| 1 | selfplay | `selfplay/games/game-000538.json` | 572386 | `4d89872358f5c1f3be77644e4a8019c44fbed6bd5eec93f1d5500c8df043f5cd` |
| 1 | selfplay | `selfplay/games/game-000539.json` | 154853 | `523fa26c912b668921c172e62f8348d400cd432261387c9afda069ba26883084` |
| 1 | selfplay | `selfplay/games/game-000540.json` | 164669 | `1e28175c938c94ce2c7ed1c733fc8961c818214011662aeb80cdeee4d5cf53ea` |
| 1 | selfplay | `selfplay/games/game-000541.json` | 283694 | `e723fa066ef9d41d593a339dc1a0852cd05bb1fa4257c161eac27ff168dad8cc` |
| 1 | selfplay | `selfplay/games/game-000542.json` | 200773 | `ee952dff3c658a53a9a765d5b62074dd7da295019648fe716dde40e151e69dd3` |
| 1 | selfplay | `selfplay/games/game-000543.json` | 551088 | `d769be257056369cefe5a090c38d82e4c44e47b56bef9e5117edce68e339f8e7` |
| 1 | selfplay | `selfplay/games/game-000544.json` | 225798 | `a9fa2f6f1ddfc7d7dd652069f339e5a2f88d1b40e981ef893bb6f7bca439efa1` |
| 1 | selfplay | `selfplay/games/game-000545.json` | 259581 | `f2ccb04938343a41a38039887b20da46e123694c5de307396e3ad41f80f1ca40` |
| 1 | selfplay | `selfplay/games/game-000546.json` | 235566 | `f992b6e082bdc8feb73aa482fc956ff5d68546927904c58f0d70452502ae0112` |
| 1 | selfplay | `selfplay/games/game-000547.json` | 342262 | `b883fdba29bd96844050efdc69e2b8bab7f936006e5279187fa229b795b3f32d` |
| 1 | selfplay | `selfplay/games/game-000548.json` | 390541 | `159839dba2fbfbce474479202f3c4659ed0b12ceee31bed1ea33d9052a031af4` |
| 1 | selfplay | `selfplay/games/game-000549.json` | 164510 | `9956267c59d3d438b3eeedc14e2ff5a8fbc5b2f63d39fff820f21dab3e7bcb27` |
| 1 | selfplay | `selfplay/games/game-000550.json` | 191597 | `4d2b8ebab0423d94750ed1ebc2bce1e73a765c888deffabb28e3020756ae149a` |
| 1 | selfplay | `selfplay/games/game-000551.json` | 379758 | `caa14d8da37186869d19797286daa6b85d2e934b55279456da663f738df7be4d` |
| 1 | selfplay | `selfplay/games/game-000552.json` | 557355 | `08ad9c62b7088e0a2a5e3fbc75c31f363bf10f9fe571319ffea4e44c13286c9c` |
| 1 | selfplay | `selfplay/games/game-000553.json` | 191343 | `d8e41de48602a1284adee26315998a8645179574020de8f831036bc509123a9b` |
| 1 | selfplay | `selfplay/games/game-000554.json` | 174121 | `a058bb03d63410e4c462d7993a2953e01f20bbe40d535f458bc6281a5daf7211` |
| 1 | selfplay | `selfplay/games/game-000555.json` | 218716 | `98096d0bf12b3b1c8e94e1581c10437d34ccfa966acf557c7d2baf2f94eeafaf` |
| 1 | selfplay | `selfplay/games/game-000556.json` | 174075 | `596bfcdd7c18981769b8e172acbed7dc51bd46b51fba250010751eda93c9b500` |
| 1 | selfplay | `selfplay/games/game-000557.json` | 299503 | `2dbec8adb9dd63d16b5101afacd9bd023630291df8d8cad28525c4b5cab9ae58` |
| 1 | selfplay | `selfplay/games/game-000558.json` | 299395 | `1c096e2097fd0ffb8dbf0cdcae31d8fb73dd1cb7b62ac72989f82fbbb346f062` |
| 1 | selfplay | `selfplay/games/game-000559.json` | 154962 | `a769d3295c3ba8dc81d5093687681531b3277e71d907c8b6558d7c5ac7342bfc` |
| 1 | selfplay | `selfplay/games/game-000560.json` | 164800 | `495b93b3d4e080f5a560830c11cc98ea32c0b6be7d5467ef231a629baa7a8634` |
| 1 | selfplay | `selfplay/games/game-000561.json` | 191341 | `c277df1970915ac900dfbc72ef8547662b481aca0e002870e53669291726e38c` |
| 1 | selfplay | `selfplay/games/game-000562.json` | 164313 | `0ba28be10882d195ad53401631f3dc56b2d1e399089265402892765051af4349` |
| 1 | selfplay | `selfplay/games/game-000563.json` | 183762 | `b6a35c5776a947dadf3bab595572ea0a465baa65d50a429a2cfef2a318f140b5` |
| 1 | selfplay | `selfplay/games/game-000564.json` | 145726 | `ee4ec20ad4cdfbde41d51721ce5e8bf3674ef8af367db93780ca2429d1df82b3` |
| 1 | selfplay | `selfplay/games/game-000565.json` | 155463 | `f2f4ad187798ecd0e3d76ed40d7807c6257a7918f9954c36820a34077822117a` |
| 1 | selfplay | `selfplay/games/game-000566.json` | 173448 | `e549ba84fb00fd34ea5f7df2afe5c7e99dc1926db654d9c12e7cf75a383a09e7` |
| 1 | selfplay | `selfplay/games/game-000567.json` | 208948 | `f0ba67c78299399a9fb21629345943508e88f27e8ac648f9d484def2c274065c` |
| 1 | selfplay | `selfplay/games/game-000568.json` | 146013 | `083960499774c91292fd1eb1b9e2b13787d0a418c72bbeb3c451484099f6fe32` |
| 1 | selfplay | `selfplay/games/game-000569.json` | 155449 | `b326b33655dce99ef3066d0c796dfd185d1800623f717cae1a72afa9f65061e2` |
| 1 | selfplay | `selfplay/games/game-000570.json` | 182810 | `9ede7450e118a4e55e9a3beccbbc856143858173d5392f27ed94ba2884b63bee` |
| 1 | selfplay | `selfplay/games/game-000571.json` | 474394 | `fdcec9757abf856a194eaef728b4b1a5e254b08b69d62a28d4352684782e317e` |
| 1 | selfplay | `selfplay/games/game-000572.json` | 145737 | `2c80585e81875ac575597c67b948e58b889de07395c81ef488ae6ea714ecf880` |
| 1 | selfplay | `selfplay/games/game-000573.json` | 516035 | `7f1642e415c7ad3712a096598d2b9229eccd58c3ed4437c1837eaee21d23aa49` |
| 1 | selfplay | `selfplay/games/game-000574.json` | 145858 | `470ef8f21f54feb49f9819a03e3d0afbd29f5910b523490fc799e671160c0be3` |
| 1 | selfplay | `selfplay/games/game-000575.json` | 252143 | `f9be16dac87db255c2a5e546d028ba0e128e602d82ed590533e00b502f8ec88e` |
| 1 | selfplay | `selfplay/games/game-000576.json` | 154889 | `06388b069b039a3acb57db85f50baffef18079753846074758be0cee3df8bc97` |
| 1 | selfplay | `selfplay/games/game-000577.json` | 164507 | `d03a4534456ba3ab742aadb24189f625cb31108121b5f450a2453ed1f473f489` |
| 1 | selfplay | `selfplay/games/game-000578.json` | 267407 | `a2cab1ebcb105bbdaf67d7d32ab7ed06b1e7ef3dcf11d5e156d27e31ca4775fe` |
| 1 | selfplay | `selfplay/games/game-000579.json` | 370201 | `4f74d4332becc02aa4184fa5bfc789611d40410962de4d26aae8e0c5fd24b1fc` |
| 1 | selfplay | `selfplay/games/game-000580.json` | 457986 | `3b24a03e423e4745dbd6ade5c4907a76bec390ad923a51c49f3e3f59cb1b79b5` |
| 1 | selfplay | `selfplay/games/game-000581.json` | 173614 | `f5e2d591b9a981163931cfca69e2d6455ffb007aeb9d7105890f4299faca8be5` |
| 1 | selfplay | `selfplay/games/game-000582.json` | 173719 | `9c27dd6bdddb8165a0330c855fc866a7578e1d81626f0df1ed3b83d2955f0e9b` |
| 1 | selfplay | `selfplay/games/game-000583.json` | 173309 | `5809334a4f660d21a0ffef51bd64a633b1ed5b68c60023929faae351f103a9e5` |
| 1 | selfplay | `selfplay/games/game-000584.json` | 145917 | `2b0426e8ee595ccc5085e7d6c34f40ffb4fd477f1f6d44943a2081ff208c782a` |
| 1 | selfplay | `selfplay/games/game-000585.json` | 183000 | `ba3952a988b6b5a31ee1294088b67f6c109033638a7faf117d9979ecdc94a817` |
| 1 | selfplay | `selfplay/games/game-000586.json` | 226337 | `fb6ef005c099f98de9a2ef771c3a02cbfeaed239ecf015919125cd68d524c14b` |
| 1 | selfplay | `selfplay/games/game-000587.json` | 191344 | `e526f88c3ccca6b4f2767126de83e4397a1fc0af63b2eeb25f79885231c807ff` |
| 1 | selfplay | `selfplay/games/game-000588.json` | 154948 | `03b7fcc662e0441249d4d4f940f6cb5d6d73d6fa7759fdc3cb71e36385a49680` |
| 1 | selfplay | `selfplay/games/game-000589.json` | 226487 | `9b5bea229ad4302e8a42c3f429934fd0a5a936cb4dca55910ec45d8b7785bce2` |
| 1 | selfplay | `selfplay/games/game-000590.json` | 173457 | `d105a18cec361f24059dc1420727b4719f1e9a4a28b8aa3e9b27e6741d6cbfb1` |
| 1 | selfplay | `selfplay/games/game-000591.json` | 201501 | `83eff6add5e899eac10af833bd7b2ba9a8bbdc8863fb0b6b21f0ee21341db3ac` |
| 1 | selfplay | `selfplay/games/game-000592.json` | 126636 | `53ced7ceb14e354999c0bc707eb63cc3238dc8ee2882aabbf1880394b39089c4` |
| 1 | selfplay | `selfplay/games/game-000593.json` | 154949 | `75d195785507983237910d90f2f54b7fae3f9508d307fed6f16ce65b0de475fc` |
| 1 | selfplay | `selfplay/games/game-000594.json` | 242774 | `8f8388a90fdd2076227519307ac3f848ab5bf15f64b86e52a007d6695d0e8649` |
| 1 | selfplay | `selfplay/games/game-000595.json` | 234986 | `85ea69f0a3a11f91601dde4bdda234e55b2d0787a753292f42b32761fdcb1339` |
| 1 | selfplay | `selfplay/games/game-000596.json` | 234893 | `66ac53225a96e657312f041bd61cdc79cffb3beb0f0e0255b730cfc5506527fe` |
| 1 | selfplay | `selfplay/games/game-000597.json` | 284960 | `6394ee230482b93c7b4115b6a038828c96028c53d15dd2aad9db146ca85f35bb` |
| 1 | selfplay | `selfplay/games/game-000598.json` | 155120 | `3c8608c1069e3c9d13719d7e6a5745d667b88af1b531cd463c6f244879de7105` |
| 1 | selfplay | `selfplay/games/game-000599.json` | 145856 | `2eb20cc5ea01714929f36a78baff6fc97d2388d232da8d696f08b154d97d16c5` |
| 1 | selfplay | `selfplay/games/game-000600.json` | 342582 | `0533ded777fd2bfc32c7f133220df7e8353fee2878e8992c150e349182dee594` |
| 1 | selfplay | `selfplay/games/game-000601.json` | 380364 | `b2333b33ddfdd203cc8e995d56bc14c06b27ff19ed84e1f6b64ecb3afbe2a2ef` |
| 1 | selfplay | `selfplay/games/game-000602.json` | 145882 | `c1ff28eb2068234545deaa7fcb2cbc01a91b03af872f8562b793cf136fcfdbce` |
| 1 | selfplay | `selfplay/games/game-000603.json` | 225629 | `2dee5c35b13db7e0a9a164f8833d52fff2d2a34c87b6b2fb56c6ac49d540237a` |
| 1 | selfplay | `selfplay/games/game-000604.json` | 268770 | `d8a556670d46c12366e488973697cc6f3e7cecfc947282d13e8db4546169fbcc` |
| 1 | selfplay | `selfplay/games/game-000605.json` | 183016 | `26d6591bcc9dad6e923898e9d353d4f4440efe4a8f351ad692a78503e16ed004` |
| 1 | selfplay | `selfplay/games/game-000606.json` | 145768 | `c393ded70a93518e40b561aae736e74c296a9e71265986029144c0c534348b0a` |
| 1 | selfplay | `selfplay/games/game-000607.json` | 251548 | `16747d61a4968e5c4a8d6d397db6ac46ff98c3ab61b96be106d88654536c1e44` |
| 1 | selfplay | `selfplay/games/game-000608.json` | 328636 | `68d273250257197795eb5c1bcc04119fa6a9e7db4ea9b8468d90559117944433` |
| 1 | selfplay | `selfplay/games/game-000609.json` | 235508 | `e818afd83eed0b8e58412d1695309e762e89f18f7368493cbd76397a8195f92d` |
| 1 | selfplay | `selfplay/games/game-000610.json` | 126441 | `ba194f2df9d6261b4591ece36754b3ddc9f1612fb3b0584e7cf31e24c2d08968` |
| 1 | selfplay | `selfplay/games/game-000611.json` | 201329 | `124dc2041799e3747abdd330837dc850d03449de0ce43955a53a32f5b62f09dc` |
| 1 | selfplay | `selfplay/games/game-000612.json` | 126666 | `3cf244f9ddb13a3016455455d0ffa74e38952b88164251bda541ef4c471bb274` |
| 1 | selfplay | `selfplay/games/game-000613.json` | 234721 | `a016a9869525d4ec02543d4cd0998094c74ea71e2d8b35d624c0a95db6da77cd` |
| 1 | selfplay | `selfplay/games/game-000614.json` | 126462 | `7be1217618e32262b7750146ce215d0fb4fa18a9f3bfdb7095cbd6bb9f9825b4` |
| 1 | selfplay | `selfplay/games/game-000615.json` | 173632 | `a4bf351e4c5e9621dac5e4ecdaa7dfa5216d5ae65e44a74f0fa164be3f77a226` |
| 1 | selfplay | `selfplay/games/game-000616.json` | 155092 | `9736e391154e5d92a972fc75c41bc01d26368558b13c7b1f86ea77561297e32b` |
| 1 | selfplay | `selfplay/games/game-000617.json` | 284856 | `b82cea0c20a17eae33f8f7a452c9e7bab7a67ce8f011d444f786be918ca9683a` |
| 1 | selfplay | `selfplay/games/game-000618.json` | 350442 | `79c1a0d8831c0cc5385554a5e10d4e569baa6829c61ac9245fbed28fd1e37677` |
| 1 | selfplay | `selfplay/games/game-000619.json` | 314295 | `ed6f25f910cfb937aa42c7d84b0535a2eb621323db33c3ed6a861cdc44bf83ff` |
| 1 | selfplay | `selfplay/games/game-000620.json` | 182732 | `4b1885e4cf5bb7ccc6d116e64b0b29bf9fa317c398294def3714476328bcab81` |
| 1 | selfplay | `selfplay/games/game-000621.json` | 259599 | `87e93d6281862998a89cd7c8167d50834e501721b789f9481ca0b34ce07c874c` |
| 1 | selfplay | `selfplay/games/game-000622.json` | 342450 | `0108a0426f4da37f3a923a34fc12da70716b3798860442574291f25cfb14b162` |
| 1 | selfplay | `selfplay/games/game-000623.json` | 268781 | `5f976ef941fa9ff8f6f20f6cfcfd904a37f88f741da3a607f1032971248b4f63` |
| 1 | selfplay | `selfplay/games/game-000624.json` | 164354 | `ce558942c56dd7d0cbb9113e647bfa21e1c87df75f0a7218c68b73a8458f9fb2` |
| 1 | selfplay | `selfplay/games/game-000625.json` | 289726 | `5e4fb22862d5ef6fb1cb68481df145614ca50db3557793f4bb3aff97321ee086` |
| 1 | selfplay | `selfplay/games/game-000626.json` | 243482 | `2e6a5d27ddb64c468dd2da3ea655c1eb3af7b5d68f22c3ea8b79cea25fe841e3` |
| 1 | selfplay | `selfplay/games/game-000627.json` | 235617 | `f8bd483e6e2ab8c3e9302859953a0877fa04759d2d7e288eaceb4589c4e8d8c2` |
| 1 | selfplay | `selfplay/games/game-000628.json` | 252236 | `b764fc9b3c311221cedb692e0deafc0f355d49f8986b048a6941415653881000` |
| 1 | selfplay | `selfplay/games/game-000629.json` | 415263 | `aa262b014cd0499c90aa4ad393f6aa78c6ab5629bb960c711fcbf41a345fc36b` |
| 1 | selfplay | `selfplay/games/game-000630.json` | 154946 | `c072932d9fbb4608032a6c5afe26669b04536eebebc18557b85b731685464b88` |
| 1 | selfplay | `selfplay/games/game-000631.json` | 154905 | `ca7fb3367162644ad8b1ac22799eae27946eb65e52884be231e577299ddaeb00` |
| 1 | selfplay | `selfplay/games/game-000632.json` | 225648 | `8f0b78e1473dbe54e985774772a2c31d8e12fb4b9df078ebe4347d430c0c2466` |
| 1 | selfplay | `selfplay/games/game-000633.json` | 191685 | `404747bf07e17a9dd72006c2c47baa83a2515ee368a161cc0828c705f4f65091` |
| 1 | selfplay | `selfplay/games/game-000634.json` | 182517 | `8a4a35752b78a5e1d6aa4da3bcf217a80e3e4316e9c1d86fac085ea459343ef6` |
| 1 | selfplay | `selfplay/games/game-000635.json` | 155101 | `364345bc541defe46607f3728e19053d3ab65ca530c8bb172088fe75e32fcb3f` |
| 1 | selfplay | `selfplay/games/game-000636.json` | 146028 | `a42ad69e3b2b9e3d72ec99b85fcb60b76904f22c4dc141a609a206928532a021` |
| 1 | selfplay | `selfplay/games/game-000637.json` | 146059 | `4bfeabbbea68d082c4209029e262d36d38a3fc76475c3ed3f39623bcbb8cb53a` |
| 1 | selfplay | `selfplay/games/game-000638.json` | 165127 | `60f2a3e211a654307d84023b1a8d284b09ca17679d48c86fa5ef769257fe8365` |
| 1 | selfplay | `selfplay/games/game-000639.json` | 145856 | `2a8b9538a2707873d3e48df748afd9642f71d37927701a4021bd38bb056808dc` |
| 1 | selfplay | `selfplay/games/game-000640.json` | 154900 | `ad80ba444494511aa0ee0e4ec2131346ef939983695434c077becf7d9b0e4c5f` |
| 1 | selfplay | `selfplay/games/game-000641.json` | 208875 | `0e6ba69dd67707480dfdad5976f77199fd93adffead4f1a799471cdb5d539806` |
| 1 | selfplay | `selfplay/games/game-000642.json` | 296734 | `cf54ed6d52181048dcb9304ddebc7fc02f8e5ae9d1e12a5d891ddc5e310c538c` |
| 1 | selfplay | `selfplay/games/game-000643.json` | 284016 | `044257db510d07bf4c2d2f6ecda9fb7784b1b2a89fcf52ef4176ab10e5d05a66` |
| 1 | selfplay | `selfplay/games/game-000644.json` | 235497 | `21b133257afb8bf54c4d03bb437a1b16afb23dd97b3e5a5277978bca677fa1e0` |
| 1 | selfplay | `selfplay/games/game-000645.json` | 164734 | `805041553633e623614cc0a3975c6d037bde61338d9349bf6af43548ca015daf` |
| 1 | selfplay | `selfplay/games/game-000646.json` | 126593 | `dc44507417e5d772d535c9fa2abf01444d5696401851e8d925dc6a8f15b9aacb` |
| 1 | selfplay | `selfplay/games/game-000647.json` | 155079 | `8e803045a6c4ebacf4c38a59fa96715f7589ad5f1fc6215fdb5bfb4a46b6dfeb` |
| 1 | selfplay | `selfplay/games/game-000648.json` | 154907 | `9feb7ff3d9d235970adf1c2db0d83ed2d4cea95b969e6d8fe5315f00ac2ac8eb` |
| 1 | selfplay | `selfplay/games/game-000649.json` | 155090 | `decc54b3de15e69afe9e0353d4cba0366321e680a1cf4cb8ac5c0280b6e796fe` |
| 1 | selfplay | `selfplay/games/game-000650.json` | 235059 | `00f5f3f6447a63b65465a9989afb0cbd18f251fe4ab9e8314562d6d699dbcb14` |
| 1 | selfplay | `selfplay/games/game-000651.json` | 126459 | `ddaa45a3ad48dac17fe2732e8bfd122d10df10ae5505a9e71ed587c4a8101f94` |
| 1 | selfplay | `selfplay/games/game-000652.json` | 226654 | `4c835f5d1a270eb16bf999751bc858f880d30a5332eb947e7911d51e358cb0a6` |
| 1 | selfplay | `selfplay/games/game-000653.json` | 261065 | `84ec1fa031ed039b4030715d19dc4a7d25de756a6164ba0745c0cbabf2636041` |
| 1 | selfplay | `selfplay/games/game-000654.json` | 209235 | `24588a95ff5706632e94917b7bcd64df6af24a6f2c9f324154d9ba79c447cf34` |
| 1 | selfplay | `selfplay/games/game-000655.json` | 164827 | `92501364b25a031f74b12343e4047a73ba4a131dec848743a915171826361659` |
| 1 | selfplay | `selfplay/games/game-000656.json` | 155134 | `65a9bda98d429ecc1b5000626d74c7d46a7cae718e7e16a22a9da6cdafd1b38d` |
| 1 | selfplay | `selfplay/games/game-000657.json` | 154931 | `221fbd25c305d0883b6d4aa753e664af74a378911daa7ffbd9018d0fbaeacb37` |
| 1 | selfplay | `selfplay/games/game-000658.json` | 154873 | `569c18d6e6df21e91e4c4dc88c4d4fa504b955c3c958a24f19a6821f558bcaab` |
| 1 | selfplay | `selfplay/games/game-000659.json` | 154913 | `2743b4ca450d4652e26b32b46e9054404603475943b143ad45093b34dde50eb7` |
| 1 | selfplay | `selfplay/games/game-000660.json` | 182902 | `363024d57a210c55b4cbb30fddab55627924fd6e8aefbc0c6d880d81d5194818` |
| 1 | selfplay | `selfplay/games/game-000661.json` | 218388 | `cbc121e3818025cdea65e15a6813eb6db9a716dacfcfc880f77f51c19bdb4cda` |
| 1 | selfplay | `selfplay/games/game-000662.json` | 173433 | `7fab8e30b26919551a34f58591011b29068e3a09b44e46f03eaee1b8b8a704d8` |
| 1 | selfplay | `selfplay/games/game-000663.json` | 173678 | `3128102739e7f5e9c8e14d1f5e26a2ae1003762d878d41498fd1226cbaaa8806` |
| 1 | selfplay | `selfplay/games/game-000664.json` | 191375 | `266e523893cd26879675ad0f7c1e5acc9f8ea708badc06f947f16644952d4923` |
| 1 | selfplay | `selfplay/games/game-000665.json` | 242298 | `6ff1ffbe174cf47a90fe2254ec9336cbb75f9536ae17166524e9913a086d681c` |
| 1 | selfplay | `selfplay/games/game-000666.json` | 284568 | `c66597ee56e2fc52d7ad0a36a633952fbfc66d58a1f5e435d5ec447ea38ada76` |
| 1 | selfplay | `selfplay/games/game-000667.json` | 183409 | `9ae6714c2d0c07358289796456882b158028453d50310a8125178ddc83ba6e0e` |
| 1 | selfplay | `selfplay/games/game-000668.json` | 315185 | `44ed1f0a2d3d06c5ceb2de4411ae08582284af60314a10109643b0367d54f967` |
| 1 | selfplay | `selfplay/games/game-000669.json` | 201493 | `a78832f73816b9b3d44cd719edccdecde537251d422a26eeb38b8e4ab99df4b8` |
| 1 | selfplay | `selfplay/games/game-000670.json` | 225639 | `4e0d02f015552311ea6180e002da790be35749e5c79f3c54237b7c2a472a7212` |
| 1 | selfplay | `selfplay/games/game-000671.json` | 145792 | `ae9c05e31d59fa720467515708a1652efbaf336c3d9705c0c2d8adbe340b0a68` |
| 1 | selfplay | `selfplay/games/game-000672.json` | 154916 | `442d6cd5ad454d954602b1666d4251bbe7d0db477fd66dcb043574b96ab78851` |
| 1 | selfplay | `selfplay/games/game-000673.json` | 126623 | `72baf9e4aa4eef0095d93c1d9e2c5bdd713258336aa9f398b6b6ced92444a1cf` |
| 1 | selfplay | `selfplay/games/game-000674.json` | 225638 | `d5c06315e3c89fd87cbb0c90748da25ba31f2ba24e059eaa57aba75efddadef9` |
| 1 | selfplay | `selfplay/games/game-000675.json` | 154963 | `f7c21a4ca65fe740f2d29ac9f220f2f9b62e8899bf98b4144e2d73c5477727fd` |
| 1 | selfplay | `selfplay/games/game-000676.json` | 145590 | `7a31ea591cfd8b83750fc4c6506bebc0b99f252e3423b52797a77b087e77d16a` |
| 1 | selfplay | `selfplay/games/game-000677.json` | 155176 | `74a16810a71ff48dc6a73d7dd1c74367d7db1677f6a6725295282157ec21cee6` |
| 1 | selfplay | `selfplay/games/game-000678.json` | 218373 | `3df53d7794b6a408b2b778b2145f125f5f91e5430d9c8097b8a51461637b34c0` |
| 1 | selfplay | `selfplay/games/game-000679.json` | 201610 | `e205ff9815bf7c8ad94220485f0a32c6ed88a63f7dfa4b55f4f8dd24a43f9af1` |
| 1 | selfplay | `selfplay/games/game-000680.json` | 297805 | `560359d62aa644ce1b292d86ead423d08b8fd6a0ab6c9326ae955e443b98e823` |
| 1 | selfplay | `selfplay/games/game-000681.json` | 566928 | `25a23bb81b2afaa54f0de1c0d517288f5dd9a6ae93f22b5e42062a33f55337d4` |
| 1 | selfplay | `selfplay/games/game-000682.json` | 393476 | `d995d4e03743d2f768e0f1f991015332e19086dab3b13e7cd53e96552a6de844` |
| 1 | selfplay | `selfplay/games/game-000683.json` | 243117 | `efe56a0be74e63e7debaef05f7c3b208d72b87f1721b77f8cd9b3d746f5b38b3` |
| 1 | selfplay | `selfplay/games/game-000684.json` | 579024 | `db8d7207fb19590ef33c2c7502e96ae3fe9e76bb5e8d0158414d39e75d5a1741` |
| 1 | selfplay | `selfplay/games/game-000685.json` | 164881 | `e07719d454413ff223e9c7ae4647b122dd5de8f4720af72dbbd3043ebe6c52c7` |
| 1 | selfplay | `selfplay/games/game-000686.json` | 355539 | `a1d2e500bd8550ec5d50ec012e06959c513885f6d210359acc2a1f3a5c6f9dbf` |
| 1 | selfplay | `selfplay/games/game-000687.json` | 126473 | `24e510e8f968c1df0e5e37b1a8fb2253c5bf3c3b2f5e21d9d7ec121ee4861039` |
| 1 | selfplay | `selfplay/games/game-000688.json` | 126443 | `226739b2773027b85502f088cdc3cb76dd3ca36a818075457fa40c7e3a22a8ed` |
| 1 | selfplay | `selfplay/games/game-000689.json` | 173456 | `fac9ce462094f5d10dc1d4f1caba1e1f113c83becc51128376e37d2697629b4f` |
| 1 | selfplay | `selfplay/games/game-000690.json` | 155570 | `ac2acf5a5bf8b6f9d28ceee3d453997683ecffff60f7da262893f0ad9a93b98d` |
| 1 | selfplay | `selfplay/games/game-000691.json` | 164892 | `3d54ab8842048cb3b64539a7ea06b0e21d1e25ff6667f8754fe9524960df0c30` |
| 1 | selfplay | `selfplay/games/game-000692.json` | 155082 | `1e6364ff1b0032eefab9980c12c8b7b5634caa405d86879668e973b9f43969d6` |
| 1 | selfplay | `selfplay/games/game-000693.json` | 154946 | `d5af00018b16032f74fbef56bcd6a629488dbfc31ccba080d91b30fabe2ef4e2` |
| 1 | selfplay | `selfplay/games/game-000694.json` | 472880 | `c77706d70bd6ebfd9973a694ff8afc445cf96a85fde082c0b4e9b6484dbe2168` |
| 1 | selfplay | `selfplay/games/game-000695.json` | 126440 | `7affac6685c7cc7091435f0245eed55c6f0d42068c60272b8d5b810412d55bbe` |
| 1 | selfplay | `selfplay/games/game-000696.json` | 234971 | `ea235c1758aae4011c1e34cb26a8c06d1a968ae11ef8e0d379bd659437772349` |
| 1 | selfplay | `selfplay/games/game-000697.json` | 155060 | `c1dbfedbfbf077514035cd52de68f25315e9237c007d535c1e041bcc0bc83a23` |
| 1 | selfplay | `selfplay/games/game-000698.json` | 315136 | `7082e9e836f64771b086b402aa8c2d42ae12278f2f18327ef8d8dc52fa56300c` |
| 1 | selfplay | `selfplay/games/game-000699.json` | 126467 | `4b475d1188ca539d84c0b8ddafd91a0a6445d449fe7e469c1b1b2cc1ab0f0c64` |
| 1 | selfplay | `selfplay/games/game-000700.json` | 192101 | `792d1a0c09b37d3a96e84411a94cb371beb4c03df68184f49dbdc1e2cba5da25` |
| 1 | selfplay | `selfplay/games/game-000701.json` | 155074 | `d50b0d72abf42c84e08efd2b5f57def29ed2f95abccdb7959fc06c60c89d9efb` |
| 1 | selfplay | `selfplay/games/game-000702.json` | 200822 | `8e46c09406ce5c113b716319f8c4d661a40e70fcc3aa0202690a0c4d35dc6849` |
| 1 | selfplay | `selfplay/games/game-000703.json` | 126586 | `4fde8192c2a87cccb221402f935c5a5bc75cf4a41e984193dd4a9d22a2b47f76` |
| 1 | selfplay | `selfplay/games/game-000704.json` | 126625 | `8cae85d6a5ca8063a0d32bc2c26f6583633ebe60cb752698c835b90cf1d90e69` |
| 1 | selfplay | `selfplay/games/game-000705.json` | 393686 | `fd70235a48115f3918292470d7408b5998ba752265fcb686424cbef610d4c149` |
| 1 | selfplay | `selfplay/games/game-000706.json` | 218404 | `319460ded3384dc2db62df61dcb2420e2717d41531d22d21828c6ad4e9bcfe7a` |
| 1 | selfplay | `selfplay/games/game-000707.json` | 173471 | `37a637bd7fda929e5796ace4435e9d3f1e6681542e9cc14d55c278a1e86832ca` |
| 1 | selfplay | `selfplay/games/game-000708.json` | 243944 | `0261be6ce721c220fbdd8a69d855e567184a690923dfe56cc1497b9abc2f666c` |
| 1 | selfplay | `selfplay/games/game-000709.json` | 154951 | `3151a6b08a14b4c0cbedae94c39a22f98ff973e113438bbc8dc0d626b20dce46` |
| 1 | selfplay | `selfplay/games/game-000710.json` | 392787 | `a7e7f3f5683e1734e4e51e0edf48b6790a3fabbd2bc97d5a1f05eb5d83782872` |
| 1 | selfplay | `selfplay/games/game-000711.json` | 154904 | `474c29ced6536b1b5a8947c09241ce675a13116eb6e2354dbc7e5bf542c1ab29` |
| 1 | selfplay | `selfplay/games/game-000712.json` | 218254 | `121cd3f14f9ea7f833f7e9d83ba5f27358c31d4cba4999d4ce0e1ae5f3b530ca` |
| 1 | selfplay | `selfplay/games/game-000713.json` | 164867 | `235e76e05507d959a1f1808a15c4a36d46cd386bb8d9e3ead29e9f52d337e37b` |
| 1 | selfplay | `selfplay/games/game-000714.json` | 183461 | `085b1960a07b7a5868313c16b62f625563b400743092ea13da179b9ccc4c9ab6` |
| 1 | selfplay | `selfplay/games/game-000715.json` | 478861 | `6ef0141099c8e81e85a50b1781febfe72d824c0aa93c17d1d7119f1b9c9f184d` |
| 1 | selfplay | `selfplay/games/game-000716.json` | 155071 | `277f102f4b71bd74d969b7382d85f67abfe5ed94c43c74f8f34717f623b0db2b` |
| 1 | selfplay | `selfplay/games/game-000717.json` | 154961 | `aba43963e085b1d813eeb57118ae1b6a3ab58fee83526d72c74db7907bf22c4c` |
| 1 | selfplay | `selfplay/games/game-000718.json` | 183115 | `a971958ced8615a4938878233466f0631d2862be31bc17456303316e1ddf7e99` |
| 1 | selfplay | `selfplay/games/game-000719.json` | 284046 | `4f73a7693068ebd1e273c6893d1c1bdb29003f2a0cf3f4f1d65b2dcf8e04cb02` |
| 1 | selfplay | `selfplay/games/game-000720.json` | 327723 | `730fa4a38a1271f08f6851a863d89094fd197161ad2dd8c15d69e4d3ed52b507` |
| 1 | selfplay | `selfplay/games/game-000721.json` | 258837 | `e45ce48bad1e83a3d633821b6c3f7911717df20219d7be818772b370b2b108bb` |
| 1 | selfplay | `selfplay/games/game-000722.json` | 274933 | `5153c38c413dc51b0671214496401ced34b4a4b0a773252abce45380eac62674` |
| 1 | selfplay | `selfplay/games/game-000723.json` | 191935 | `fe6ae0d7a1e5ecf3b244c55ddddf039c0dc16fab4d1af25e910beb8d958a0255` |
| 1 | selfplay | `selfplay/games/game-000724.json` | 165022 | `49df8f2a93d9895192717b33750caea65ac3ed702d53dc6bce2a648a48d62671` |
| 1 | selfplay | `selfplay/games/game-000725.json` | 284248 | `4fb35fd3e11cc717db64157d33ca6dd903d331da166fbb548aba5b27d7b53d24` |
| 1 | selfplay | `selfplay/games/game-000726.json` | 259165 | `d63b2381ed7a235ea15ece50544780fd5feeb31067151b1a9c23b975a79ddf13` |
| 1 | selfplay | `selfplay/games/game-000727.json` | 164340 | `c7863b701c483fb7b4fa3c634af14388b049fe6ed6ddd5fa98a0c12c8f25ea63` |
| 1 | selfplay | `selfplay/games/game-000728.json` | 200926 | `6ee6afc332de2e929d915f634839c3a5b52dc679d81d35026bee9f6c707a1dbe` |
| 1 | selfplay | `selfplay/games/game-000729.json` | 434978 | `3a54315e81579dacb8fc895f3a9dcf2fa4dc7caea12811f41a493a397d589a08` |
| 1 | selfplay | `selfplay/games/game-000730.json` | 447862 | `a9655e4e744ef543058eb7ee1eaf13dfc1de20b795ae58c9fc9bc88d2c408fb3` |
| 1 | selfplay | `selfplay/games/game-000731.json` | 155065 | `216a59cd60f7e235282a63e6ca47aead272800bd4f01f6add34f2dd878a2c67f` |
| 1 | selfplay | `selfplay/games/game-000732.json` | 126497 | `d399bedf07b71818208476fad008c26373176b01d379c31d34926fe586b0db05` |
| 1 | selfplay | `selfplay/games/game-000733.json` | 391367 | `c94a507942c1d6485eb7ae3d4ac9c1d6dd12c1c664b0825446151c73aa63bc7a` |
| 1 | selfplay | `selfplay/games/game-000734.json` | 126310 | `c05a42d8720828ae3bbe78d8330d615300294404773fefb10a839e2cfdf3b196` |
| 1 | selfplay | `selfplay/games/game-000735.json` | 234946 | `f9691df9b726c9a0f9842d65b8423891b5ecf34cb1978e78633d83c393b3f950` |
| 1 | selfplay | `selfplay/games/game-000736.json` | 368338 | `c17d779ca79e0ff7e3f6eea2b6f341c2d19546d33a1bbb118c0bc79db7d7f674` |
| 1 | selfplay | `selfplay/games/game-000737.json` | 218761 | `a3e46d2e897587b9e4c0810884037d3077821f06d9ad2ef76a813bd79c6196d3` |
| 1 | selfplay | `selfplay/games/game-000738.json` | 154905 | `0c8ab368cbcd67f6f218eaa2e752e220ad9e9e20e36c5728970e0c1209145553` |
| 1 | selfplay | `selfplay/games/game-000739.json` | 155087 | `44a8e2bfab15a4de81ea713e336ae90c44acb3a7301779810fee25b548091976` |
| 1 | selfplay | `selfplay/games/game-000740.json` | 173477 | `3560a75d83cc4ba0f60add764cbf376e44056fd356295b9403fc32205484d711` |
| 1 | selfplay | `selfplay/games/game-000741.json` | 284507 | `605b3639f42d58e8ba58b1d7b24814b7bf33ee8efb254ab39ff7d52a8c66feec` |
| 1 | selfplay | `selfplay/games/game-000742.json` | 154890 | `bd568b428cf0ee1ffe2784674d1842cc7ff074666543cbdc4b6d40baf5a4faeb` |
| 1 | selfplay | `selfplay/games/game-000743.json` | 201144 | `063cdf325134da04192341db47dbcb948de6a4aa4a711bf27f70453a5178053c` |
| 1 | selfplay | `selfplay/games/game-000744.json` | 218263 | `11fe684c37af6a60a218d802d0fd3568ce2900320098ab1e4d1d923d44245843` |
| 1 | selfplay | `selfplay/games/game-000745.json` | 126540 | `6a6c3cd96d90f3a72fb41e596be9071936325020610068bdd27af97298d9a5f9` |
| 1 | selfplay | `selfplay/games/game-000746.json` | 164659 | `b618c3ab272692e04659725b7d73a023260dad6bd620b4c47a64dc076c7b99a1` |
| 1 | selfplay | `selfplay/games/game-000747.json` | 562946 | `6d2eeee59989c379d1221cc1774457c581338c4e19d33d07fd2dd5e40d36c639` |
| 1 | selfplay | `selfplay/games/game-000748.json` | 154922 | `b7f134fbe5accbfecb2d40a8ff57a97e517b1108bc61306569b8b4b63591d1f4` |
| 1 | selfplay | `selfplay/games/game-000749.json` | 258791 | `7434d545b985e799cc81cc5a18b4128f8cd098c24904fcd2bbb7874e1ec169e8` |
| 1 | selfplay | `selfplay/games/game-000750.json` | 174084 | `d31cfbfdea34899040f51dbeea31b907c3318bf423b6e42fb74e954e4cb9e9d4` |
| 1 | selfplay | `selfplay/games/game-000751.json` | 352345 | `8eb2a25b9edc31ebce4a4dc52e7963178da91908d90cdd2f00d785f9f16ce217` |
| 1 | selfplay | `selfplay/games/game-000752.json` | 191851 | `eaa3452a4e57873f97c1a2c41823be2b5c8d20f2e2d4b46a16b57421a24e8de2` |
| 1 | selfplay | `selfplay/games/game-000753.json` | 173294 | `58af71bcef865237ee71add233606e298a4a7064eecf3c2b4eed5da3b4578298` |
| 1 | selfplay | `selfplay/games/game-000754.json` | 183494 | `554e32acd76b4d8ff8606cbe612b43b0e4ec796b152a92d748253f90d5e27b79` |
| 1 | selfplay | `selfplay/games/game-000755.json` | 145704 | `c5be3b455b6ce482126caf5ec0f237bd5974ab3ac35a7d61260e9469e61136d0` |
| 1 | selfplay | `selfplay/games/game-000756.json` | 462873 | `3d4b162c9825b6b05771186ff478df131c5c135de0f4a54b65168f3bb4c503d7` |
| 1 | selfplay | `selfplay/games/game-000757.json` | 164565 | `eba817efac4b75769e05ac5cdbd9ca42d5c6ff4f4bee5ef9b0ded620b0baf570` |
| 1 | selfplay | `selfplay/games/game-000758.json` | 283376 | `2ee287e15a6f2ce42dfb106f591e60103af3ce7acb247219198cbe6696fdecc7` |
| 1 | selfplay | `selfplay/games/game-000759.json` | 218164 | `06d284abc8f8559a93db98d495553996a6b824ecde22feb47506757896bb18d8` |
| 1 | selfplay | `selfplay/games/game-000760.json` | 200559 | `9e8e1673db7504040daa12c4717fb6c4c516e018143ecb826a4b9cd627dc09b2` |
| 1 | selfplay | `selfplay/games/game-000761.json` | 252670 | `01eee38b1bf6d8a0327fb2abaf9b9163b793d9231b284c58dce8d822e2ab4259` |
| 1 | selfplay | `selfplay/games/game-000762.json` | 234687 | `b28073349d8cfb8df9ba765be1d6f1c7a2c3faba2122f486ba6d486f18996e1f` |
| 1 | selfplay | `selfplay/games/game-000763.json` | 173688 | `41ea1fbafa5f684b6848aa0c2e92232f58eda623128f12accced54a96e36e468` |
| 1 | selfplay | `selfplay/games/game-000764.json` | 154902 | `8352b43ebd43a3c8c1d1b6ef4601ae229267f6a12b79c70843a13ebcd897b97e` |
| 1 | selfplay | `selfplay/games/game-000765.json` | 208606 | `5ad3f4e542b7403a82417636bb2a610520affee11f1569167d56037a8c33bf69` |
| 1 | selfplay | `selfplay/games/game-000766.json` | 126438 | `1be10b0adaa10a2ee622a509ca326b07ba972c543edc24d2bc9bae6932cbb96c` |
| 1 | selfplay | `selfplay/games/game-000767.json` | 126638 | `61fb35f233ea3b392322c54ecb91498f3ae7b3404163edc77f7cf31c0db9051e` |
| 1 | selfplay | `selfplay/games/game-000768.json` | 318617 | `2d7772d97f59736d62d81a91c69e4e106619ee7c794bb8bd504547abe686117e` |
| 1 | selfplay | `selfplay/games/game-000769.json` | 235523 | `68227d8a6256fb1b64e0de5891504b93176f4bd3c8fb050f28a76562d38d6b0e` |
| 1 | selfplay | `selfplay/games/game-000770.json` | 409817 | `7ad1be054d47ae3bf156693d11c7028e79d1509f2a3c70803dccf426bbed9450` |
| 1 | selfplay | `selfplay/games/game-000771.json` | 267973 | `93a889a7ae374cbd683f776c2085afcf79ae6e6c1a0e47753dce57e15b20a268` |
| 1 | selfplay | `selfplay/games/game-000772.json` | 227257 | `7338b1eba5bcdc0959add31351631572cd162761252520a13d0bfeb8ab1f7957` |
| 1 | selfplay | `selfplay/games/game-000773.json` | 126604 | `92fbb18a02359b72d0518bb9280a713fc351cc5d8a7738924278e101a9d66ebb` |
| 1 | selfplay | `selfplay/games/game-000774.json` | 155103 | `65afe1a4114b2edbbb3729b118111cc20fd476adb3d8250e1fd391f50e798d31` |
| 1 | selfplay | `selfplay/games/game-000775.json` | 381888 | `16a78e11ed20432c7421ac7dae589bb5a11fbf44a2f1a57b6b6e2eab6d67ea74` |
| 1 | selfplay | `selfplay/games/game-000776.json` | 145906 | `9de16fe915bfcf3db612a5ae3e226810ed6573a5ead0fe8191b0f58ba5f4949a` |
| 1 | selfplay | `selfplay/games/game-000777.json` | 242502 | `78d197d7e47a6a7d6c5b7269ed1d6a5abd6e7e47965bff1ee9542374fc120bcd` |
| 1 | selfplay | `selfplay/games/game-000778.json` | 154952 | `a3544ab9848f4ea4a794a50d1afadd3687257171e5e0ceeab84927be2db8a6cd` |
| 1 | selfplay | `selfplay/games/game-000779.json` | 251069 | `414cf9bf991a8a9d22ac763bea73765de3c2a72e0ef2b6a457c500f2add000f4` |
| 1 | selfplay | `selfplay/games/game-000780.json` | 173267 | `ab0291836643809832bef18f2635f5d334d729c80c0ab28c22be81522f79371f` |
| 1 | selfplay | `selfplay/games/game-000781.json` | 225601 | `20614dbf44870cb1a6631582c5e93bf90589a572d55711e51f1bbbc8ae8109e5` |
| 1 | selfplay | `selfplay/games/game-000782.json` | 164861 | `bc9610365f7b22f89863809d85bbd052f0eb0eece9c62ea4574cf7fea2daa99b` |
| 1 | selfplay | `selfplay/games/game-000783.json` | 283720 | `534402e1550f55fe782a4811158cf5109b3b401a8226e401035ef8653519bff8` |
| 1 | selfplay | `selfplay/games/game-000784.json` | 155120 | `2a3ff7f4bf4e1bc1f8d36d0c4559369fb345218ea54f4fa100851a2fb375c563` |
| 1 | selfplay | `selfplay/games/game-000785.json` | 126424 | `42467668eb626d31d76f18cbb59ed1990ed4cc6980ee401dcfe3da90f900a6b2` |
| 1 | selfplay | `selfplay/games/game-000786.json` | 474042 | `6b9618ee160b5a5b5e78f210361a9adf16d3013b348bf6c5e2d00bfae85786e0` |
| 1 | selfplay | `selfplay/games/game-000787.json` | 298122 | `b98f95864da6851a548077e20cc7305b121cd4e4e39006ce6e6711312575f520` |
| 1 | selfplay | `selfplay/games/game-000788.json` | 155100 | `2223ce91e6a48ec496e06ebbc0256a19fe76b45cd31b442966a10e3e10016a14` |
| 1 | selfplay | `selfplay/games/game-000789.json` | 299414 | `d98d7ca1ed87748db08d0b606837291c2d09e8af43d4f3b6a1fca93ed88a0d46` |
| 1 | selfplay | `selfplay/games/game-000790.json` | 164515 | `750e14f7108cdc068ccaa0ecb06b3ba4aba79df61e659192f6a25c66199c94c9` |
| 1 | selfplay | `selfplay/games/game-000791.json` | 235532 | `949799f098338a9399c5be2f79adcae622894d9c13b868fd5e580f0e6a1255c5` |
| 1 | selfplay | `selfplay/games/game-000792.json` | 304651 | `6af4d34800ee85125023c47d1c6a55a3f9ccf685546669f3c82ee724ebfd0b4f` |
| 1 | selfplay | `selfplay/games/game-000793.json` | 154926 | `a7ddf31b05f3a637b31c271ea0d82ebf123b6df38c144f79ba04d423dac04fd6` |
| 1 | selfplay | `selfplay/games/game-000794.json` | 183500 | `9bb1d3c86e94ebb56aab06eff707021e69424af74fddfe08b23db5e15a335c16` |
| 1 | selfplay | `selfplay/games/game-000795.json` | 276090 | `afd0ab97ad3516f1e77c92a4408ce070f8f4cc777c7e21322f48f241d7bc857b` |
| 1 | selfplay | `selfplay/games/game-000796.json` | 284010 | `1a3258487f5b2132c2a82ec9b41b8d5c0c69ed78b7f6de9563e5336aa7901880` |
| 1 | selfplay | `selfplay/games/game-000797.json` | 173463 | `b057fdb767806e14b2fee09be1831ea354a40fe082359a75e565a075b3685ce2` |
| 1 | selfplay | `selfplay/games/game-000798.json` | 566274 | `955e9caa11ebdc6aeb4869f59b17394e8312e4e8b119646aa5e5cc08c40bee93` |
| 1 | selfplay | `selfplay/games/game-000799.json` | 155439 | `27f796dda965501d426c7ae7882d92a82205435ad8c8152ddaca8ac7384ddf02` |
| 1 | selfplay | `selfplay/games/game-000800.json` | 268942 | `2c33169a02a6dcbce2163bb8ff0c5ea6c180e320ae78e3a8297c728875a121df` |
| 1 | selfplay | `selfplay/games/game-000801.json` | 154896 | `72997e5333d70a26819f154eeb8e7af7883d0b5563071fd893b6d6c59b83c136` |
| 1 | selfplay | `selfplay/games/game-000802.json` | 126500 | `c2bc6358d8ff4358bc5d1649fdc51899a056cf1b4d946724b5dff45387c1f7ff` |
| 1 | selfplay | `selfplay/games/game-000803.json` | 234837 | `cf4e1c8b886438387d8bcd68ffbef5a937c3638ac9c05b58e8beff95c1a6226d` |
| 1 | selfplay | `selfplay/games/game-000804.json` | 252277 | `ca3c1e097e6fde9e9af4c15580ee0e759920aaa11b60e129b4238b4779d06981` |
| 1 | selfplay | `selfplay/games/game-000805.json` | 182631 | `7b299e8d31d46f911fa2bbf61e3af13b0d997bd0252e3d6ff8824c89b520c91f` |
| 1 | selfplay | `selfplay/games/game-000806.json` | 155057 | `c8fd2a5f3888f906f4dd959f768786813e3921129f09c76d570fa0ff3747b60c` |
| 1 | selfplay | `selfplay/games/game-000807.json` | 164532 | `9540ae6a954eaf6c22ef042dc8836bd09230d1f9465dde25a3a0dc341529f22f` |
| 1 | selfplay | `selfplay/games/game-000808.json` | 164797 | `fd36a71eb5d171485d3919d2b7d648347c19e8d2f55920597c2de6ee043e6b58` |
| 1 | selfplay | `selfplay/games/game-000809.json` | 417567 | `691989a593c7befd1bdcf3262bf2dd907acebed1fe792ed10ccbb381b32944a0` |
| 1 | selfplay | `selfplay/games/game-000810.json` | 321905 | `eff1c5f4ec6bf3472749328e327b497d3ccc28f09613db1c183544d726ce367b` |
| 1 | selfplay | `selfplay/games/game-000811.json` | 342921 | `e860d726c9c65db1ebe512f730aecaee5c58372f16c3a33c972d171d29f66609` |
| 1 | selfplay | `selfplay/games/game-000812.json` | 382588 | `0c1cfe70d957bd9baf965b347d8d95b4d02ff9dc56beef203dd2d30f25e2366f` |
| 1 | selfplay | `selfplay/games/game-000813.json` | 341891 | `4de9dcd55670424c3f7881e82ea13aee77be2ed9c86df05f0171c98450ebe33e` |
| 1 | selfplay | `selfplay/games/game-000814.json` | 299551 | `954a41493e1c467db99d60a7909bc352f12ec50b51e08d15f062334d4b65effa` |
| 1 | selfplay | `selfplay/games/game-000815.json` | 342417 | `8d3cbe94fd9bf1c7c47a03f43fa7d1c465069193b6e2caf0bcea2e56f97d3825` |
| 1 | selfplay | `selfplay/games/game-000816.json` | 385067 | `4e74988aebeb746b48488bfec5c36e41a1ab628a0dbda868a4440af35190535a` |
| 1 | selfplay | `selfplay/games/game-000817.json` | 218065 | `a8e137e4c46a1883622f09e837312727195e66393588432e520ac38062f010af` |
| 1 | selfplay | `selfplay/games/game-000818.json` | 284335 | `5dbd0aa36b614466f03e4cd31dea12a108bba7e16a05b13164200606cc593b61` |
| 1 | selfplay | `selfplay/games/game-000819.json` | 267534 | `dcce83db63a4fd5f726f887974cb6012ed7c3cb7b1d9994a84ae6cbb59d2e885` |
| 1 | selfplay | `selfplay/games/game-000820.json` | 182960 | `2f8a1ca437087b5e1a889a678c5703609b1522b2a4b7c22fdc16462d9e705583` |
| 1 | selfplay | `selfplay/games/game-000821.json` | 155072 | `e45277b11c156911b9519811bcb5fbcd6eb2a2a4ebc34d02bd4ce7d04033b684` |
| 1 | selfplay | `selfplay/games/game-000822.json` | 251201 | `2a62ac26b6cf6add97b47d5e949fb8e1d59b1c57fde166ed75a5e7f31395fa25` |
| 1 | selfplay | `selfplay/games/game-000823.json` | 400374 | `7a0442dc2aee648d416251df69fa12d4f812b4f2618fa6b4f42dd42588a9a8a3` |
| 1 | selfplay | `selfplay/games/game-000824.json` | 200661 | `6b7d0c450f020e13794d56b16c41f67b938c4faf27081d96adfdbb26f5b4734c` |
| 1 | selfplay | `selfplay/games/game-000825.json` | 145887 | `d25321a3e78fad0a92738505e39deabb7d677ae0d2d226f062575b799af60f51` |
| 1 | selfplay | `selfplay/games/game-000826.json` | 145989 | `680ff80fa56eb8f0aad35b2043344c0da9af3490e74831a3173c9a583e3095a5` |
| 1 | selfplay | `selfplay/games/game-000827.json` | 236554 | `b99f9615f29d45fd1bd3ed341e9e0f9cb61c5499b0d802e31c1f57a525a137c7` |
| 1 | selfplay | `selfplay/games/game-000828.json` | 327748 | `9ba6d3c6827664f3005ca47425aae60762170cc940f8a6ed65de0e71db49bea4` |
| 1 | selfplay | `selfplay/games/game-000829.json` | 225657 | `992414d9a730c451a7d17b4e91b5e01ae1e45aec5af25d844d6164e896166b34` |
| 1 | selfplay | `selfplay/games/game-000830.json` | 154922 | `a60bbe9d451616fb9016d836a00b2d81d9a4a6e8ca45aecad6236d37407b6f06` |
| 1 | selfplay | `selfplay/games/game-000831.json` | 218190 | `627726fdb2047d5fdb0d81ea6c4fbab7872fdba07a2cbb673b91ea7a97b3b80e` |
| 1 | selfplay | `selfplay/games/game-000832.json` | 126664 | `e3376d748657924fa15039f44c66deeb8baefdbbc239724de0c9da0cd6ee10bb` |
| 1 | selfplay | `selfplay/games/game-000833.json` | 155490 | `a0857ff786d2d850c9570f574762f3cf7f7af657142096f4c064aa57239fb7a3` |
| 1 | selfplay | `selfplay/games/game-000834.json` | 551529 | `3fbb39366835513e014d6e795e0eeb1380826fa424e98a9fc6a5f425ea69621f` |
| 1 | selfplay | `selfplay/games/game-000835.json` | 565528 | `06aea1a042c10cba3be7526a0a971453654e4645c670e4e913c7d2fd273ba508` |
| 1 | selfplay | `selfplay/games/game-000836.json` | 235486 | `a2655ef5ed61db29a1acf0a7fa9a761ac5a674ad3c3f3998dd93e9a27ced84e0` |
| 1 | selfplay | `selfplay/games/game-000837.json` | 218471 | `cf201b403bc40e97357c5fc87c7928a645579dedf188002c37e60bdd1b6ca2b6` |
| 1 | selfplay | `selfplay/games/game-000838.json` | 283469 | `7fe32dbd5a629b5861994531dae1a952b14f6cc894124f06e409a98ec7ba39ad` |
| 1 | selfplay | `selfplay/games/game-000839.json` | 355422 | `313e24787bd5bd899b7d78b9068dd3eba214dd56d7a9786c7e7544cb1c0f1ad0` |
| 1 | selfplay | `selfplay/games/game-000840.json` | 145811 | `12569c85ffeff324ef09c0ff3de81e221e4b20223f2b6867ae8258e9355af88b` |
| 1 | selfplay | `selfplay/games/game-000841.json` | 155233 | `568c8da3caf348ef6066bc720b9efafd06ba8b5327d4d5cd4aacef9687a1f1d8` |
| 1 | selfplay | `selfplay/games/game-000842.json` | 126508 | `850ad7b4158fef5bcac89bee5835e086a528db6725bdf0e58abe0e463e7ce5ca` |
| 1 | selfplay | `selfplay/games/game-000843.json` | 154952 | `d7558c19d761db48f56d516fecf503f284f9cebc8b6581d6c65bcd5b54b58595` |
| 1 | selfplay | `selfplay/games/game-000844.json` | 146045 | `f88f90e3951bfe324d59d8470aa41fdc927c7bb450832948dc6f586971a37501` |
| 1 | selfplay | `selfplay/games/game-000845.json` | 155128 | `ca64c2365f90c5b56a153fdf944cfc22c891021adfe632e6d709a766af3f68eb` |
| 1 | selfplay | `selfplay/games/game-000846.json` | 126503 | `85b24289a57254eb61f47e8e24c6bf95ed12856ea4fbd372b7a4148d61f4ab1b` |
| 1 | selfplay | `selfplay/games/game-000847.json` | 267683 | `5edb1aa301dc36f9f7b94d2d64e71da11e72bacd60becb1cff52d97914d627be` |
| 1 | selfplay | `selfplay/games/game-000848.json` | 145685 | `e0592d857e5b8579cb730e1141f50bbe6c2a1797b4a629e0f1e58dd8a543a3b2` |
| 1 | selfplay | `selfplay/games/game-000849.json` | 155081 | `a895cf234a128ec5f532ac081c3dca1eb94cb19c0ca86a951ac3a5795bde03c4` |
| 1 | selfplay | `selfplay/games/game-000850.json` | 328544 | `2892f05a67a3678bafe6c34d400a50b7f8adb3a62bc8d8e4af644a3679384b4d` |
| 1 | selfplay | `selfplay/games/game-000851.json` | 126630 | `5dd5540898ffe68331b39e6bb963a5dc39f854e851ca784e22241cd03ef204a1` |
| 1 | selfplay | `selfplay/games/game-000852.json` | 145787 | `a7e06270bdaab015413ed2309d0b86de786136191166f08edbd0ab661af4b634` |
| 1 | selfplay | `selfplay/games/game-000853.json` | 200692 | `91b38ab3ce875568bc92f0fd93612be6083c89008a88192d8f8a0201fc2fd513` |
| 1 | selfplay | `selfplay/games/game-000854.json` | 217998 | `f31a54d4df0d4e5b01e6e376b71405dfe01bd794d37d0bdca0bec771ecb84da2` |
| 1 | selfplay | `selfplay/games/game-000855.json` | 126452 | `fe0ca0a769bb6dbbde708c6dc1f3b162140300d851e1cc5292a12676402d5664` |
| 1 | selfplay | `selfplay/games/game-000856.json` | 183389 | `6b7a601b8bc4646cb88a44c619a2c00615704e0556e4563b4ba032a0630c395b` |
| 1 | selfplay | `selfplay/games/game-000857.json` | 354877 | `d48ed856dc36de7b7ac85063cbd936f94b1dfcc445aa97c8b03bfde52ded0ff7` |
| 1 | selfplay | `selfplay/games/game-000858.json` | 173550 | `de6d9491ee3a7b069ca1dbd732be83492b7d0157c9b8ac32ba9318405719f00b` |
| 1 | selfplay | `selfplay/games/game-000859.json` | 173307 | `3c29b3c2d4ec30e2196ea336c7e778fc2b47704800a5b349d7e523305c87b959` |
| 1 | selfplay | `selfplay/games/game-000860.json` | 155298 | `c06dcd9c6765c913894e3e3c67a6e17273fccf41a41290234d12c62d0896a378` |
| 1 | selfplay | `selfplay/games/game-000861.json` | 155441 | `605f9c4bcd32711d84fc47db23bfa66444662a1de013eb427982bce52fea45d7` |
| 1 | selfplay | `selfplay/games/game-000862.json` | 126620 | `e8c83b07334ec7fca721c16c2cd31c547d2c3536f7063c556a60c481bbd80588` |
| 1 | selfplay | `selfplay/games/game-000863.json` | 252285 | `e982d2e472340a7dee232a499cda92dae372d1d156fe5461b6a83279477b9458` |
| 1 | selfplay | `selfplay/games/game-000864.json` | 409561 | `6d7758cf92f5b2a1e00f409066fc373e1d783ca0bfcb5d7649287911878576d2` |
| 1 | selfplay | `selfplay/games/game-000865.json` | 234436 | `a2dd65cdf1170d128edf82f133876589c618606a705ca96501005f52979bfbfc` |
| 1 | selfplay | `selfplay/games/game-000866.json` | 191410 | `970eab81443214a2e05309f83fc7223e7dc864086d5df23fe6af0866efb082e5` |
| 1 | selfplay | `selfplay/games/game-000867.json` | 236715 | `7532e7a6af8db994452fd102eab1c17101464856284d281a9305247a9dc6c9d1` |
| 1 | selfplay | `selfplay/games/game-000868.json` | 154961 | `d59eda7d949d84f9a625f69d991231e9891c907a79c0398839f3968a352af534` |
| 1 | selfplay | `selfplay/games/game-000869.json` | 218744 | `f2a5453cd3bf49ee82fca7520a3d1855360b8fdc453a45cac89a40229b9a4168` |
| 1 | selfplay | `selfplay/games/game-000870.json` | 415763 | `49b3bf746be5f95fd6d1bec57d80b45a3daadc5c63ac4880682f2da4809f464a` |
| 1 | selfplay | `selfplay/games/game-000871.json` | 225619 | `56762fab32a9f56c838645a98ab7d6cd567048db9ecae3152d5517e291dd52a6` |
| 1 | selfplay | `selfplay/games/game-000872.json` | 145736 | `57622f871a7aa9aa98ded394669736bb9ab96435e7604e881688137434d97c8d` |
| 1 | selfplay | `selfplay/games/game-000873.json` | 307013 | `7bb7545ad473af306c1890a224493158970358a3894de2117248e5792cbf2f6e` |
| 1 | selfplay | `selfplay/games/game-000874.json` | 369553 | `35c56fc9db20e32f38ff5f98b137d1f30636a1ad71758895c15ade2b6b826f94` |
| 1 | selfplay | `selfplay/games/game-000875.json` | 242847 | `a9ef231d5c3a8b18ef4990dc7553584ee2e235ec38205da17d9fc2e04b5fb017` |
| 1 | selfplay | `selfplay/games/game-000876.json` | 235147 | `53c45404b18280ee4df7415e96259d0f5a2ae6883f6fe62c9134ca8bf47ed5ef` |
| 1 | selfplay | `selfplay/games/game-000877.json` | 328087 | `d4b1d1470ce4ddb08657bd5eebc6536821dbc7aa958de0614c8321e9b5da9cb2` |
| 1 | selfplay | `selfplay/games/game-000878.json` | 154983 | `2c731bc7a6032b8c683f0b7f558b794b47dbffc1e0373596c805d440b00a14a0` |
| 1 | selfplay | `selfplay/games/game-000879.json` | 154913 | `61022a7a7255fe6278e52e93e387011899da1d149514c69bd2158254eaab408d` |
| 1 | selfplay | `selfplay/games/game-000880.json` | 126455 | `68108471b0e1fe230369c3e23b5646a6a30224578c1ee7a69a40407d08899ba9` |
| 1 | selfplay | `selfplay/games/game-000881.json` | 145911 | `a8a22391e4374f37c543d9aa5f7846f8203544d603a5a55d726c2bc45182423f` |
| 1 | selfplay | `selfplay/games/game-000882.json` | 191835 | `0e695e4ac8f6e36fed7601275eef95b6cc12912ab347b93672204578a7e091f4` |
| 1 | selfplay | `selfplay/games/game-000883.json` | 200238 | `f8e19c964c92358956aacf03c2dcffecd8f49224ec060e064821b97eaacb087a` |
| 1 | selfplay | `selfplay/games/game-000884.json` | 364398 | `7610a7ec2f9341513d9f55d03a9b0993626ee581b9fdeabe69efcb15454ba34c` |
| 1 | selfplay | `selfplay/games/game-000885.json` | 126453 | `d1b3b6d10eb55712249102a2802c86ec2cd76966b1e088daf27713b4d2177ca8` |
| 1 | selfplay | `selfplay/games/game-000886.json` | 173554 | `d9334b1ce8fdd48d31413d91c830332d5d87e5647ce7394c992df96b032b70d3` |
| 1 | selfplay | `selfplay/games/game-000887.json` | 208713 | `c782cf07a4af0aa04385f0cc96426548f7463a85eda708535946cbdb11050528` |
| 1 | selfplay | `selfplay/games/game-000888.json` | 283619 | `867d17d549ce3a4a4cf73e4656f883909026e290772cd89e248427bc41200cb2` |
| 1 | selfplay | `selfplay/games/game-000889.json` | 126614 | `5fffa3dcdd83a2d214d554a54b3150f93695f0e79e6617a142d98b92714e7533` |
| 1 | selfplay | `selfplay/games/game-000890.json` | 314427 | `7cafa9374c1f2f56cf446a700cbf13676eb4fa7bd022f3a7bda031ccd002d6dc` |
| 1 | selfplay | `selfplay/games/game-000891.json` | 427285 | `50e51451175473dbbf0bac556a1c9671c18eb0fe266fcfe80539a9acd92dc05a` |
| 1 | selfplay | `selfplay/games/game-000892.json` | 200645 | `13f09e80534df37f7208c04495daad9f953a884c5aedf5b2eda3a778f0ff1bff` |
| 1 | selfplay | `selfplay/games/game-000893.json` | 368136 | `e39963b535f829a68eaf6b39e91e23209611a20aa3ae5e0956a69b7a8b5da1c7` |
| 1 | selfplay | `selfplay/games/game-000894.json` | 155301 | `b228e9dc4252a94545bde0722fc993141af783efdeb1467b35935d4951ea1c6d` |
| 1 | selfplay | `selfplay/games/game-000895.json` | 356356 | `53092d0c486dc4a1d7b9112dfbce78effd62f3d849afca2096d55e5130a6de3d` |
| 1 | selfplay | `selfplay/games/game-000896.json` | 303977 | `0681b94b40fa44a4e65e2f60f8373d043702f8404bdf704a4104ebb30f4701b3` |
| 1 | selfplay | `selfplay/games/game-000897.json` | 173463 | `21109787578d7387d7263aefdf5d9683774ce6ed319e3c1282c02a0f52565569` |
| 1 | selfplay | `selfplay/games/game-000898.json` | 155084 | `2ee13ecbc5028bc0054246550a75f3bbec7b4626da8cb9213db2b362bcec1c57` |
| 1 | selfplay | `selfplay/games/game-000899.json` | 268215 | `06f0ab4d16fd7a52cdb660c075ceac982a9971720bd03ffc4267d9cf2b7eee78` |
| 1 | selfplay | `selfplay/games/game-000900.json` | 357244 | `0572006b6ea26bba57ed375b67f1b2584c96c59fbca458840ceefecbedd09f9b` |
| 1 | selfplay | `selfplay/games/game-000901.json` | 200963 | `59225edaa6c41ed0392ac926c4f395c11a9b76c5db09a18ab19757dd9d525d2e` |
| 1 | selfplay | `selfplay/games/game-000902.json` | 126430 | `14d79de1d8c79ac723ce16d586b5a8392462a30a4774c1d701e2f98d92d9a209` |
| 1 | selfplay | `selfplay/games/game-000903.json` | 173475 | `23b2a21fade4cda91314439a7279b03c3004fdbda67b9b005a6055b41f46d0e3` |
| 1 | selfplay | `selfplay/games/game-000904.json` | 311322 | `8b5934198c70d35ac0dfd73ce1af9d239e3472957613716866509df4940d85db` |
| 1 | selfplay | `selfplay/games/game-000905.json` | 418577 | `7850740e0faa4178a9554a104ab0df13961083636672cd1201302541f41ca94d` |
| 1 | selfplay | `selfplay/games/game-000906.json` | 145983 | `14965692bc363f1e91b260bc1d37cdeade68e2cc6babfd0b3f74c2fa0a057574` |
| 1 | selfplay | `selfplay/games/game-000907.json` | 191312 | `21ae6d715e6f2aba5e675ffb94bd144180465fbe47fa522b08034df54f51eaff` |
| 1 | selfplay | `selfplay/games/game-000908.json` | 561507 | `37a56d457a50eca534e8b6221723d61a01db579c49de01626d123a873290b59e` |
| 1 | selfplay | `selfplay/games/game-000909.json` | 299656 | `64547baa11d5f077be5add22afd918976e62e904aeb76aaceb259888c5736916` |
| 1 | selfplay | `selfplay/games/game-000910.json` | 164759 | `eaf2904447d93ca13fed40daca8d61f10a49198a2c318f4178e67e4db4947916` |
| 1 | selfplay | `selfplay/games/game-000911.json` | 154898 | `4fead69b214273d9ed70ec56011e43976bb1c8c369330fa69cb9823bff3452ce` |
| 1 | selfplay | `selfplay/games/game-000912.json` | 154934 | `6a3bcfb3ded75dfec8b39c3c46f54ebf1d47e5be44e6f66a46482fbc447e8bfd` |
| 1 | selfplay | `selfplay/games/game-000913.json` | 155438 | `5a80332730201d83a6044cf824b7906d46e6cc4dc0631c74910b36176a17f63b` |
| 1 | selfplay | `selfplay/games/game-000914.json` | 191937 | `d38a824d8ae87632d3ab46377ec29b1827570f6c9f0f2bc1feb07322fd7e0c93` |
| 1 | selfplay | `selfplay/games/game-000915.json` | 553555 | `e01242d019a44330d88c89b31da7fa513fdd9a89ed9a6dd44191707341799f16` |
| 1 | selfplay | `selfplay/games/game-000916.json` | 154925 | `025e2c12b7f535512e854eaa9e65b8b216e1d541800b68f416deab6a469031d5` |
| 1 | selfplay | `selfplay/games/game-000917.json` | 155441 | `003327e57cccb691d93cd3a053d2e5432f3a2c31150131315318de5b15ad16ec` |
| 1 | selfplay | `selfplay/games/game-000918.json` | 155175 | `60714ada0eaffc50143698a684b5ba94f3df42392ff0adfa74bfe99586ecc5ac` |
| 1 | selfplay | `selfplay/games/game-000919.json` | 126500 | `e9fb747ba316f3107250d09b282501dd157c9ea073a78212002a63bbb01c3698` |
| 1 | selfplay | `selfplay/games/game-000920.json` | 290196 | `71f4632bb74591c6635dbcaf72e04f773f819f1e7a08c967123c2a9175bd0fee` |
| 1 | selfplay | `selfplay/games/game-000921.json` | 344539 | `f96ff1f799de62e853ebd10357a6ef76b007ade9b6fd8c04d89f5fa0a4f018e6` |
| 1 | selfplay | `selfplay/games/game-000922.json` | 357402 | `9cc4e0e45f2a8edd8155a150b77f659dd51db035fafa3571c887c4a5b7562070` |
| 1 | selfplay | `selfplay/games/game-000923.json` | 208917 | `98171e5abbd44c19e434911ac5d33e2e8a48552de55683ac6d3093158ce5a2bb` |
| 1 | selfplay | `selfplay/games/game-000924.json` | 126496 | `61a3562809597b8224b81d45690fdf532d7e63a0e1cb7b5ca6aee37d4d23dc0d` |
| 1 | selfplay | `selfplay/games/game-000925.json` | 155486 | `f25fc62e1ea01dbf6068b60a9586dd338714a6ee35b03d2ac9f7dede745f25c2` |
| 1 | selfplay | `selfplay/games/game-000926.json` | 235604 | `b3a670fb2df53dfabecef0efa93e57d4858d4196e5948466f21735ae095f7217` |
| 1 | selfplay | `selfplay/games/game-000927.json` | 183297 | `2bde4ed9a1deac5d31c1e7a36fa74de51bd549a86125429461a51648f75bc2de` |
| 1 | selfplay | `selfplay/games/game-000928.json` | 173757 | `6ef4185c6500392e2fa6c917bf781660f9e9ab792f5a04c9173660c02c3a56ab` |
| 1 | selfplay | `selfplay/games/game-000929.json` | 327696 | `af42e9fc458cb9533343bce523feb7c0ed3482f5b66754504beba501a9a7cb3f` |
| 1 | selfplay | `selfplay/games/game-000930.json` | 164320 | `8dec175d03f57abf1bad8c6fe13eaf812eb9ff7b49953838d1345161e2eb7e98` |
| 1 | selfplay | `selfplay/games/game-000931.json` | 381155 | `b023ea8741ff5a8b29516353863a7877a631964fb4a47c69a045eecfe3366287` |
| 1 | selfplay | `selfplay/games/game-000932.json` | 164604 | `80c49a9581fd10e1f5e23f434a568182503662cc3bd0bb4bc102928f794badab` |
| 1 | selfplay | `selfplay/games/game-000933.json` | 154901 | `93766ff1a8d843f261ee4cbb4679ef7717d35c6295164372307b0bd6d04e650a` |
| 1 | selfplay | `selfplay/games/game-000934.json` | 284213 | `fb973b991a521921dff50fc383be3e5802e66eed9b59b4cc0ee5d82089431e00` |
| 1 | selfplay | `selfplay/games/game-000935.json` | 182634 | `3882fb3a54cfda0f27264dbafc527bf117d4173017f31b6fc787b2270b9b7d21` |
| 1 | selfplay | `selfplay/games/game-000936.json` | 291601 | `bad41550d039d887639c5262b1af76eb5cb35f0168bcf7f424fb26a593c459f0` |
| 1 | selfplay | `selfplay/games/game-000937.json` | 369321 | `95628e9b6fc21665c70ce43341094d3aecc983f87986161c123f50854b99e0b7` |
| 1 | selfplay | `selfplay/games/game-000938.json` | 155303 | `ddb3be1395822f03bc7b27bb433387c19caee49d138fa3cb64da1dd786ee15d3` |
| 1 | selfplay | `selfplay/games/game-000939.json` | 368889 | `6b9fe7ada0f54b9410ed8da4a6c7a512b461d89f7afc3bd3887b55952e45d903` |
| 1 | selfplay | `selfplay/games/game-000940.json` | 164679 | `b44637f65c719486a998270665e578de8897777ecd62a5f886d59aada0e7d23b` |
| 1 | selfplay | `selfplay/games/game-000941.json` | 404650 | `a2af903327fcac91711a99fe6c8f1b6fc3bd785a85c2a3198a3802117f256fb8` |
| 1 | selfplay | `selfplay/games/game-000942.json` | 251426 | `0d97062660e6c37a0c9e438cc822da8cc46971dd99ff2b507059404c96b3bbeb` |
| 1 | selfplay | `selfplay/games/game-000943.json` | 155024 | `f6a4999978e374676000d56aa4fb92bb066c2bc870e357bf9369ee6de12a1320` |
| 1 | selfplay | `selfplay/games/game-000944.json` | 200923 | `2d4480f55d1eea73f418b577ef0c44226d11d450046fec45f44a4494f4b4711f` |
| 1 | selfplay | `selfplay/games/game-000945.json` | 343399 | `286d73b1e793db0888360dcb49986acb39f42ed3a0b0e4abf03892e910824220` |
| 1 | selfplay | `selfplay/games/game-000946.json` | 146096 | `2d26cb674293c5a8b47ed30ad550fd3347a559a66b127fb0019ab4b5143d6259` |
| 1 | selfplay | `selfplay/games/game-000947.json` | 164734 | `54a3e2a8c6c442faf1352bdd92cc92aa60b6668eb0c5feba4be8e39dbb903e9c` |
| 1 | selfplay | `selfplay/games/game-000948.json` | 155108 | `39e9c9b4574cfbba81b70a1c9f412d8eb017bcbef10386f7172fc94bb77ddf98` |
| 1 | selfplay | `selfplay/games/game-000949.json` | 384459 | `038fc553cd0ed5ee71e60dfa6de75bfe85f009342b00f42fc83b8d20bc9cb120` |
| 1 | selfplay | `selfplay/games/game-000950.json` | 314448 | `6eb1421ab34f8301c331c537665c65bacf5e1ad1a22b818e88e4ae8009648c30` |
| 1 | selfplay | `selfplay/games/game-000951.json` | 299490 | `fd1134b924545b3569219c5d59c8a98f7812d80293196c020d84e8ae5e74570c` |
| 1 | selfplay | `selfplay/games/game-000952.json` | 182515 | `e225635c319bad0d5c978ac95aa1cbb6f0413206da066f5ddc3d5bed6be2b196` |
| 1 | selfplay | `selfplay/games/game-000953.json` | 182929 | `a0fdefe821b4b907ab3d835f6a5312661b63845e7d6e2a64ec90bd37f7f50f03` |
| 1 | selfplay | `selfplay/games/game-000954.json` | 394444 | `14102504c474aeb8659c97fb5f5fd99f34ad5e6dc3f0b331c2dab030a634a308` |
| 1 | selfplay | `selfplay/games/game-000955.json` | 145719 | `0286d9c53a825990f877ef7e925418adc6f45e2b0d2e1c136595f734275afb00` |
| 1 | selfplay | `selfplay/games/game-000956.json` | 155128 | `085f1cdb851219e8e98f4b4d6f1cea71d433f039111faf790b08dd825baeaa4c` |
| 1 | selfplay | `selfplay/games/game-000957.json` | 368363 | `af822acac56f93e281a6e428b1459e14202198d471a3f06e9801989998bb4620` |
| 1 | selfplay | `selfplay/games/game-000958.json` | 183041 | `9a75e996fff76f2b4668783ccf02e8d92cce141b176914da07e1a8eafa8f2736` |
| 1 | selfplay | `selfplay/games/game-000959.json` | 154963 | `a85bfafa9a6ca7c8f2a9d457c85315a5af61945fd4faa4acfebfef2a4e22c364` |
| 1 | selfplay | `selfplay/games/game-000960.json` | 182675 | `36806c98fda1be11ebc971829cea64b2c8b5010ea2a6a6e9f30a87bf37825d4e` |
| 1 | selfplay | `selfplay/games/game-000961.json` | 225626 | `357154e71d1b512833dc98561f9012d4111f98955a7806a8a25e6d208b04bcb2` |
| 1 | selfplay | `selfplay/games/game-000962.json` | 395779 | `bfee9783b475f838ad75b6003e7a2e3807718f0abda440269f4cc9f09605151f` |
| 1 | selfplay | `selfplay/games/game-000963.json` | 145957 | `e98b027fee6fe60e217d2100bb4f6c278f559d38abe3262b1922fb5c6cb8e867` |
| 1 | selfplay | `selfplay/games/game-000964.json` | 145850 | `3808414aa07a5a361b00fa2ac9677b21009028f2bc55acf91de067a1e4e25add` |
| 1 | selfplay | `selfplay/games/game-000965.json` | 154943 | `4c521c793416739bd5136933a18b89a2e540aea7c97971f3720a3aa220c6e5d3` |
| 1 | selfplay | `selfplay/games/game-000966.json` | 126435 | `826eaaab6339b3061a160bd126bc3570ab00d5a2d529ee1727e647debec2bf3d` |
| 1 | selfplay | `selfplay/games/game-000967.json` | 155117 | `3efee457b623c8db2a090969627812991ece9e2e18910285832674c89a365fb9` |
| 1 | selfplay | `selfplay/games/game-000968.json` | 244026 | `2942c824625465c4055fd8719a973965735c065b07d7074edabfeee7217ef847` |
| 1 | selfplay | `selfplay/games/game-000969.json` | 183163 | `641fdf562c69dcd04188b38942799cb1707772d36b049aeb3e4b697daf580269` |
| 1 | selfplay | `selfplay/games/game-000970.json` | 126496 | `72972c6a8c382b65e25e070fdfd63bba5a80423c5706a87a8712f962328b17c8` |
| 1 | selfplay | `selfplay/games/game-000971.json` | 258731 | `2fbf3690d74c5dc0368cd6dc235a529855444878b292ec376de89c2e477f24a0` |
| 1 | selfplay | `selfplay/games/game-000972.json` | 259006 | `a9e3245b388c872d2d60db3a365a8fcb0c609fd1d533b4a20ccb8995d17fb762` |
| 1 | selfplay | `selfplay/games/game-000973.json` | 391784 | `9a453b756c6507bbd992da6295eb0f4924672fb5087e90231d69713f78d80768` |
| 1 | selfplay | `selfplay/games/game-000974.json` | 210005 | `d7b3756acd41e330143023eef8e39a1ef6dd3890cb5899794c158b225bca10b2` |
| 1 | selfplay | `selfplay/games/game-000975.json` | 568183 | `dce70d23f4464de0b730e40d5385164fee86fa42c5426873437eb1e8fbee99cc` |
| 1 | selfplay | `selfplay/games/game-000976.json` | 126638 | `dfd5dbcae801936e6e8b7f16cb3cccf89c67921dfda61868936333e2ae5ee294` |
| 1 | selfplay | `selfplay/games/game-000977.json` | 201192 | `341c179c67e1196f50a0afe10975e1ecb5494696a7e1b19e87c41325c75a56b6` |
| 1 | selfplay | `selfplay/games/game-000978.json` | 218233 | `65b4cba2f843b0bdc2cbc1e70fa8c4a70004f7ef7da316a2af54770be1d75caf` |
| 1 | selfplay | `selfplay/games/game-000979.json` | 155311 | `0de679b8996ef399b9e5c1f175ff6247f2fa7b29190c1d83817e5456ff7ed0b7` |
| 1 | selfplay | `selfplay/games/game-000980.json` | 315428 | `58d2f2ee00c235257752d2e257ab6ff2bb0dce2d2c1594e784b882b0f6a0576e` |
| 1 | selfplay | `selfplay/games/game-000981.json` | 234433 | `0c85ff712ad2358484f4b5dea21a044db5c9ddded6fa2f51d8f0b111defce513` |
| 1 | selfplay | `selfplay/games/game-000982.json` | 252760 | `9b1ccb3f4fb667171e0cad051a92cfa69418211935bdb1b07f774e7a8fd86dd7` |
| 1 | selfplay | `selfplay/games/game-000983.json` | 328726 | `1ec634183845b9d556b90893d0d3a4a3588e622e883f43578c3c2e083d911bcb` |
| 1 | selfplay | `selfplay/games/game-000984.json` | 225596 | `fdd2f89e7d6c7b538746f3521296e28fa908e02514e6a8c458c15c675d0ddd9c` |
| 1 | selfplay | `selfplay/games/game-000985.json` | 201367 | `e79552d9ac507c28358bbd9287b4717d9f4ce05354dfb5c4e5672ed56ef01dcc` |
| 1 | selfplay | `selfplay/games/game-000986.json` | 126623 | `9ecfcb585ba10fdbfbc10938a365fbfcbeea3c09bb4bb44bca89fa4eaf531390` |
| 1 | selfplay | `selfplay/games/game-000987.json` | 174152 | `e33af18dae41c46a3dadd04dfca3466e049259aa8560e59dce4d188c5914b286` |
| 1 | selfplay | `selfplay/games/game-000988.json` | 451966 | `81e0b71b7ad238433c145599dfa3efb9df0ceae7ce2dd4a8329aa046230c0cb0` |
| 1 | selfplay | `selfplay/games/game-000989.json` | 242441 | `ac61da25ac4c6fbfe43f31de3e32b984c03abe06c2d9aa1ab36c1650afffcd0f` |
| 1 | selfplay | `selfplay/games/game-000990.json` | 236320 | `9c3efe9c5a755e35d6e0ce15a243489346335c8f7c9606c1f86f907b5e8d89a9` |
| 1 | selfplay | `selfplay/games/game-000991.json` | 126528 | `1f07cc7cefb2b674437b2b7152205008e348a7a8a247c29ff9635e4b692faac4` |
| 1 | selfplay | `selfplay/games/game-000992.json` | 146119 | `05d49edd464f592c2338b68e574b31f1b5d741361093b3d866f3b01b03a7a188` |
| 1 | selfplay | `selfplay/games/game-000993.json` | 173453 | `3ccc84b71dcbcddcdca0596079e3560575390d2995eb0c2988de467494ddd101` |
| 1 | selfplay | `selfplay/games/game-000994.json` | 200677 | `fb8254bcd79b07fd123377008ca34c0ad3bd2d667ff6a475b60cf3b822da020b` |
| 1 | selfplay | `selfplay/games/game-000995.json` | 405927 | `721664a6b2951ea24814e6f30b751b5a6167b94f73b37e2b22c48bdb73a5832b` |
| 1 | selfplay | `selfplay/games/game-000996.json` | 274185 | `77ee717b97e7ac4fa0f9228b1cf948cac8ec65c69d47a61a1924308c28eb12fe` |
| 1 | selfplay | `selfplay/games/game-000997.json` | 469053 | `392378a8ef6afc07368fb219519cb86a0760f775f7f84f29aaf8d672031efc24` |
| 1 | selfplay | `selfplay/games/game-000998.json` | 356783 | `d617cd6d3cc4d062ed1c88ec26336bdc6b2716304276f87e689f0b97f2566c2b` |
| 1 | selfplay | `selfplay/games/game-000999.json` | 329358 | `5e260fd0e27d8b7a328f65cb644e306db4d32e5b8a74281e47b85a874e433308` |
| 1 | selfplay | `selfplay/summary.json` | 130792 | `bd05a6321a64afb890f42d9c5dcdf2cf39a83d2429f93cde3391e1510528eca8` |
| 1 | training | `candidate/best.pt` | 327034 | `5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2` |
| 1 | training | `candidate/latest.pt` | 328652 | `6adac54f55c915a2149d99bc93ab4bee5a20fff74439f6a7294b92e049eea452` |
| 1 | training | `candidate/metrics.jsonl` | 5634 | `46cba9fb7ebb96dadd3a5c3c429d55f2b4a9f777b4b5c6ff1fd9a5e85409cbbc` |
| 1 | training | `candidate/summary.json` | 8080 | `e2ef26aad0af5dd071e85ed7b731f53b12e7d4764b194e0bc0c16bd07e06edcc` |

## Training target distributions

| Gen | Policy support mean (min–max) | Mean max probability | Mean entropy (nats) | Value counts (-1 / 0 / +1) |
| ---: | --- | ---: | ---: | --- |
| 1 | 23.550 (1–24) | 0.385 | 2.280 | 11172 / 2853 / 11510 |

## Fixed policy/value probes

### initial champion

| Position | Side | Value | Top legal policy moves |
| --- | --- | ---: | --- |
| opening | red | 0.874 | (5,4) 0.204, (5,5) 0.170, (4,3) 0.118, (5,6) 0.038, (4,6) 0.033 |
| linked-opening | red | 0.777 | (4,4) 0.141, (5,4) 0.124, (5,5) 0.124, (3,6) 0.099, (3,3) 0.098 |
| contested-midgame | red | 0.575 | (3,6) 0.271, (5,5) 0.215, (5,4) 0.104, (3,3) 0.063, (4,7) 0.061 |

### generation 1 candidate, final champion

| Position | Side | Value | Top legal policy moves |
| --- | --- | ---: | --- |
| opening | red | -0.349 | (5,5) 0.344, (5,4) 0.150, (4,3) 0.072, (5,6) 0.036, (4,6) 0.032 |
| linked-opening | red | -0.166 | (5,5) 0.216, (3,3) 0.097, (4,4) 0.093, (5,4) 0.074, (3,6) 0.063 |
| contested-midgame | red | 0.398 | (3,6) 0.325, (5,5) 0.202, (3,3) 0.069, (5,4) 0.057, (7,4) 0.045 |
