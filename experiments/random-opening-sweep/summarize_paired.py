"""Render complete paired opening results as a reviewable Markdown report."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "paired-4ply"
FROZEN = [f"n{n}-vs-frozen" for n in (0, 2, 4, 6)]
PAIRWISE = [f"n{a}-vs-n{b}" for a, b in ((0, 2), (0, 4), (0, 6), (2, 4), (2, 6), (4, 6))]


def wdl(value: dict) -> str:
    return f"{value['win']}-{value['draw']}-{value['loss']}"


def load(name: str) -> dict:
    path = ROOT / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    results = {name: load(name) for name in (*FROZEN, *PAIRWISE)}
    suite = [pair["start_position_hash"] for pair in results[FROZEN[0]]["pairs"]]
    for name, result in results.items():
        hashes = [pair["start_position_hash"] for pair in result["pairs"]]
        if hashes != suite[:len(hashes)]:
            raise ValueError(f"{name} uses a different opening suite")
        expected = 200 if name in FROZEN else 100
        if len(hashes) != expected or result["summary"]["games"] + 2 * result["summary"]["opening_terminal_count"] != 2 * expected:
            raise ValueError(f"{name} is incomplete")
    lines = ["# Paired 4-ply random opening evaluation", "",
        "Each pair uses identical opening moves and position hash with colors exchanged. W-D-L is from the left candidate's perspective. Paired result is better/equal/worse pairs, followed by candidate game-point rate.", "",
        "| Candidate | vs Frozen paired result | Overall W-D-L | Red W-D-L | Black W-D-L | Draw | Avg moves |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in FROZEN:
        s = results[name]["summary"]
        lines.append(f"| {name.split('-')[0].upper()} | {s['better_pairs']}/{s['equal_pairs']}/{s['worse_pairs']} ({s['paired_score']:.1%}) | {wdl(s['overall'])} | {wdl(s['red'])} | {wdl(s['black'])} | {s['overall']['draw']} ({s['draw_rate']:.1%}) | {s['average_move_count']:.1f} |")
    lines += ["", "| Candidate pair | Better / Equal / Worse | Overall W-D-L | Both win / Split / Both loss | Opening terminal |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name in PAIRWISE:
        s = results[name]["summary"]
        lines.append(f"| {name} | {s['better_pairs']}/{s['equal_pairs']}/{s['worse_pairs']} ({s['paired_score']:.1%}) | {wdl(s['overall'])} | {s['both_win']}/{s['split']}/{s['both_loss']} | {s['opening_terminal_count']} |")
    lines += ["", "## Opening diagnostics", "",
        f"The 200-pair suite has {len(set(suite))} unique positions; maximum multiplicity {max(suite.count(item) for item in set(suite))}. Side to move is Red after four plies. The 100-pair candidate comparisons use the exact first 100 positions of this suite.", "",
        "| Matchup | Games | Opening terminal | Both win / Split / Both loss | Red win rate |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name in FROZEN:
        s = results[name]["summary"]
        lines.append(f"| {name} | {s['games']} | {s['opening_terminal_count']} | {s['both_win']}/{s['split']}/{s['both_loss']} | {s['red_win_rate']:.1%} |")
    lines += ["", "## Decision", "",
        "Random-opening training improved playing strength against Frozen Gen11 for N=2, 4, and 6 on this suite. N=2 has the highest Frozen score (68.6%) and the most favorable direct pair comparison against Frozen (84 better, 8 worse). N=0 falls below Frozen (34.2%) and loses decisively to every random-opening candidate.", "",
        "N=2 is the preferred setting for the next training run: it beats N=4 by 25 better to 7 worse pairs and N=6 by 17 to 10. The N=2 versus N=6 margin is smaller, so the exact ordering deserves another suite before promotion. N=4 and N=6 are effectively tied here (14 better, 15 worse).", "",
        "Four-ply evaluation reduces Red wins to 65.5–71.2% in the Frozen matchups, compared with 399/400 Red wins in the earlier normal-start screens. Red remains favored, so role-swapped pair results are the primary strength measure. No champion promotion was performed."]
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "comparison.md")


if __name__ == "__main__":
    main()
