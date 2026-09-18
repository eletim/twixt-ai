"""Derive auditable sweep tables from completed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "summary"


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(path: Path, player: str) -> dict:
    report = read(path)
    result = report["summary"]["pairs"][0]
    games = report["games"]
    assert len(games) == 40
    seeds = {game["seed"] for game in games}
    assert len(seeds) == 20
    by_side = {}
    for side in ("red", "black"):
        selected = [game for game in games if game["agents"][side] == player]
        assert len(selected) == 20
        wins = sum(game["result"]["winner"] == player for game in selected)
        draws = sum(game["result"]["winner"] is None for game in selected)
        by_side[side] = {"wins": wins, "draws": draws,
                         "losses": len(selected) - wins - draws}
    return {"wins": result["wins"][player], "draws": result["draws"],
            "losses": result["games"] - result["wins"][player] - result["draws"],
            "games": result["games"],
            "score_rate": (result["wins"][player] + .5 * result["draws"]) / result["games"],
            "red_games": len([g for g in games if g["agents"]["red"] == player]),
            "black_games": len([g for g in games if g["agents"]["black"] == player]),
            "by_side": by_side}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for n in (0, 2, 4, 6):
        home = ROOT / f"n{n}"
        report = read(home / "generation/report.json")
        summary = report["selfplay"]["summary"]
        games = summary["games"]
        counts = summary["aggregate"]["wins"]
        manifest = home / "generation/dataset/manifest.json"
        pd = read(home / "policy-diagnostics.json")["metrics"]["candidate"]["overall"]
        vd = read(home / "value-diagnostics.json")["metrics"]["candidate"]
        moves = [g["move_count"] for g in games]
        positions = sum(moves)
        excluded = sum(g["opening_length"] for g in games)
        training_positions = sum(g["training_positions"] for g in games)
        assert len(games) == 5000 and positions == training_positions + excluded
        assert read(manifest)["examples"] == training_positions
        stats = report["selfplay"]["inference"]["statistics"]
        row = {
            "n": n, "games": len(games), "positions": positions,
            "training_positions": training_positions, "random_opening_excluded_positions": excluded,
            "training_positions_per_game": training_positions / len(games),
            "avg_move_count": mean(moves), "median_move_count": median(moves),
            "first_player_win_rate": counts["red"] / len(games),
            "draw_rate": counts["draw"] / len(games), "winner_distribution": counts,
            "games_per_hour": report["selfplay"]["games_per_hour"],
            "simulations_per_second": 64 * training_positions / report["selfplay"]["runtime_seconds"],
            "effective_batch": stats["requests"] / stats["batches"],
            "dataset_manifest_sha256": sha(manifest),
            "initial_checkpoint_sha256": report["champion"]["sha256"],
            "best_checkpoint_sha256": sha(home / "training/best.pt"),
            "policy_ce": pd["cross_entropy"], "policy_entropy": pd["entropy"],
            "policy_top1_probability": pd["top1_probability"],
            "policy_top3_cumulative_probability": pd["top3_cumulative_probability"],
            "visit_top1_agreement": pd["top1_visit_agreement"],
            "value_mse": vd["groups"]["overall"]["mse"],
            "value_mae": vd["groups"]["overall"]["mae"],
            "calibration_error": vd["calibration_error"],
            "decisive_sign_error": vd["decisive"]["overall"]["sign_error"],
            "vs_frozen": pair(home / "strength-evaluation.json", f"n{n}"),
        }
        (home / "selfplay-diagnostics.json").write_text(json.dumps({
            key: row[key] for key in ("games", "positions", "training_positions",
                "random_opening_excluded_positions", "training_positions_per_game",
                "avg_move_count", "median_move_count", "first_player_win_rate",
                "draw_rate", "winner_distribution", "games_per_hour",
                "simulations_per_second", "effective_batch")}, indent=2) + "\n")
        rows.append(row)
    pairs = {}
    for i, a in enumerate((0, 2, 4, 6)):
        for b in (0, 2, 4, 6)[i + 1:]:
            pairs[f"n{a}-vs-n{b}"] = pair(OUT / f"n{a}-vs-n{b}.json", f"n{a}")
    cross = {}
    for test_n in (0, 2, 4, 6):
        directory = OUT / f"cross-n{test_n}"
        policy = read(directory / "policy-diagnostics.json")
        value = read(directory / "value-diagnostics.json")
        cross[f"n{test_n}"] = {}
        for model in ("bootstrap", "n0", "n2", "n4", "n6"):
            p = policy["metrics"][model]["overall"]
            v = value["metrics"][model]
            cross[f"n{test_n}"][model] = {
                "positions": p["positions"], "policy_ce": p["cross_entropy"],
                "visit_top1_agreement": p["top1_visit_agreement"],
                "value_mse": v["groups"]["overall"]["mse"],
                "decisive_sign_error": v["decisive"]["overall"]["sign_error"],
            }
    (OUT / "results.json").write_text(json.dumps({"conditions": rows, "pairwise": pairs,
                                                    "cross_validation": cross}, indent=2) + "\n")
    table = ["| N | Games | Training positions | Avg moves | Red win | Draw | Policy CE | Value MSE | Sign error | vs Frozen W-D-L |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        f = r["vs_frozen"]
        table.append(f"| {r['n']} | {r['games']} | {r['training_positions']} | {r['avg_move_count']:.2f} | {r['first_player_win_rate']:.1%} | {r['draw_rate']:.1%} | {r['policy_ce']:.4f} | {r['value_mse']:.4f} | {r['decisive_sign_error']:.1%} | {f['wins']}-{f['draws']}-{f['losses']} |")
    table += ["", "## Candidate pairwise", "", "| Pair | First candidate W-D-L | Score |", "| --- | ---: | ---: |"]
    for name, result in pairs.items():
        table.append(f"| {name} | {result['wins']}-{result['draws']}-{result['losses']} | {result['score_rate']:.1%} |")
    table += ["", "## Self-play and target distribution", "",
              "| N | Median moves | Excluded positions | Positions/game | Red | Black | Draw | Games/hour | Sims/s | Effective batch |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        w = r["winner_distribution"]
        table.append(f"| {r['n']} | {r['median_move_count']:.1f} | {r['random_opening_excluded_positions']} | {r['training_positions_per_game']:.2f} | {w['red']} | {w['black']} | {w['draw']} | {r['games_per_hour']:.0f} | {r['simulations_per_second']:.0f} | {r['effective_batch']:.2f} |")
    table += ["", "Throughput was measured while some generation jobs ran concurrently; its differences cannot be attributed to N alone.",
              "", "## Held-out Policy and Value diagnostics", "",
              "| N | Policy entropy | Policy top1 | Policy top3 | Visit top1 agreement | Value MAE | Calibration error |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        table.append(f"| {r['n']} | {r['policy_entropy']:.4f} | {r['policy_top1_probability']:.4f} | {r['policy_top3_cumulative_probability']:.4f} | {r['visit_top1_agreement']:.1%} | {r['value_mae']:.4f} | {r['calibration_error']:.4f} |")
    table += ["", "Each row above uses that condition's own game-held-out validation set. The target distributions differ between rows.",
              "", "## Common held-out comparison", "",
              "Each model was also scored on every condition's validation set. Each cell is Policy CE / Value MSE.", "",
              "| Test set | Frozen | N=0 | N=2 | N=4 | N=6 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for test_n in (0, 2, 4, 6):
        scores = cross[f"n{test_n}"]
        cells = [f"{scores[name]['policy_ce']:.3f} / {scores[name]['value_mse']:.3f}"
                 for name in ("bootstrap", "n0", "n2", "n4", "n6")]
        table.append(f"| N={test_n} | " + " | ".join(cells) + " |")
    table += ["", "## Artifact hashes", "",
              "| N | Dataset manifest SHA-256 | Best checkpoint SHA-256 |",
              "| ---: | --- | --- |"]
    for r in rows:
        table.append(f"| {r['n']} | `{r['dataset_manifest_sha256']}` | `{r['best_checkpoint_sha256']}` |")
    table += ["", "## Interpretation", "",
              "Random opening reduced the extreme Red win rate and increased game length and training positions per game. N=2 produced the most training positions. N=6 had the lowest own-set Value MSE among opening conditions; N=2 had the lowest Policy CE among candidates on the N=2 and N=4 test sets.",
              "", "Playing-strength improvement was not established. All four candidates scored 20/40 against Frozen Gen11. Five of six candidate pairs split 20-20; N=0 versus N=4 ended 20-1-19. Across all ten 40-game matchups, Red won 399 games and one game was drawn. The normal-start evaluation cannot identify a strongest N under this severe first-player effect. No promotion decision was made."]
    (OUT / "comparison.md").write_text("# Random opening sweep results\n\n" + "\n".join(table) + "\n")


if __name__ == "__main__":
    main()
