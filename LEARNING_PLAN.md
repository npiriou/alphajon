# AlphaJon Learning Plan

AlphaJon is the machine-learning fork of SimuDonjon. The goal is to evolve from
the current heuristic simulator into a full game-playing AI while keeping the
rules engine trustworthy and measurable.

## Final Goal

Train an AI that can play the full game:

- draft items;
- manage hero and item abilities;
- decide when to flee, pass, or keep drawing;
- choose which object to break, discard, repair, or replace;
- adapt risk to medals, hero level, standings, and remaining rounds;
- beat the current heuristic AI with statistically significant results.

The AI should not learn the rules from pixels or logs. It should learn decisions
from structured game states produced by the simulator.

## Current Codebase Shape

Important files inherited from SimuDonjon:

- `simu.py`: main turn loop, `ordonnanceur(...)`.
- `joueurs.py`: current in-game decision heuristics.
- `draft.py`: item draft logic and Monte Carlo/prior picker.
- `party.py`: full evening simulation with medals and hero leveling.
- `bench_fuite.py`: policy-vs-policy benchmark pattern for flee decisions.
- `objets.py`: 265 object classes with hook-based effects.
- `heros.py`: 19 hero classes, available at levels 1 and 2.
- `monstres.py`: dungeon deck and card definitions.

The main technical obstacle is that `ordonnanceur(...)` currently runs a whole
game internally. For machine learning, it must be refactored so the game can
pause at decision points, expose legal actions, receive an action, and continue.

## Architecture Direction

Create a policy interface used by the simulator:

```python
class GamePolicy:
    def decide_flee(self, state, legal_actions):
        ...

    def decide_replay(self, state, legal_actions):
        ...

    def choose_item_to_break(self, state, legal_actions):
        ...

    def choose_draft_pick(self, state, legal_items):
        ...

    def choose_item_activation(self, state, legal_actions):
        ...
```

Initial policies:

- `HeuristicPolicy`: wraps the current behavior.
- `RandomPolicy`: sanity-check baseline.
- `ScriptedPolicy`: deterministic testing policy.
- `ModelPolicy`: loads a trained ML model.

The simulator should call a policy instead of directly calling hard-coded
heuristics. At first, only one method needs to be ML-backed; the rest can
delegate to the heuristic policy.

## Training Stages

### Stage 0: Stabilize the Simulator Boundary

Purpose: make the current game runnable through policy objects without changing
game behavior.

Tasks:

- Add a `policies/` module.
- Move current decision behavior behind `HeuristicPolicy`.
- Keep outputs identical or statistically equivalent to the existing simulator.
- Add reproducible seed handling for games, drafts, and evenings.
- Add small regression tests around one-game and many-game simulations.

Success criteria:

- `donjon.py`, `draft.py`, and `party.py` still work.
- The heuristic policy produces the same benchmark ranges as the current code.
- Policy-vs-policy benchmarking can run without editing core game files.

### Stage 1: Learn Flee / Continue

Purpose: first real ML module and first proof that the simulator can train an
agent.

Decision points:

- attempt flee;
- do not flee.

Later extension:

- continue drawing;
- pass turn.

Observation should include:

- player PV;
- current score;
- medals;
- turn number;
- hero id and level;
- flee modifier;
- object count and intact count;
- active combat/survival option count;
- covered monster types and powers;
- number of cards remaining;
- remaining monster power histogram;
- remaining event count;
- known next card features, if available;
- opponent score summary;
- number of players alive and still in dungeon.

Reward:

- primary: win/loss;
- negative: death;
- small positive: survival and final score;
- careful penalty for fleeing too early.

Recommended algorithm:

- PPO via Stable-Baselines3.

Success criteria:

- Beat or match current `politique_fuite = "ev"` in mixed-seat benchmarks.
- Report win%, death%, flee%, clear%, average score, and confidence intervals.

### Stage 2: Learn Replay / Pass

Purpose: teach the AI whether to voluntarily keep drawing after resolving a
card.

Actions:

- pass;
- draw again when legal.

Important risk:

- This policy interacts heavily with fleeing. Train it after the flee policy is
stable, or train both as one small action head.

Success criteria:

- Better score/win tradeoff than the heuristic `deciderDeRejouer`.
- No pathological behavior such as endless greedy drawing when death risk is
obvious.

### Stage 3: Learn Object Break / Discard Choices

Purpose: replace `decideBriseObjet` and similar discard/replace heuristics.

Actions:

- choose one legal intact object;
- possibly choose "do nothing" when legal.

Observation additions:

- per-object id;
- intact flag;
- PV bonus;
- priority;
- active/passive flag;
- target type tags;
- target power tags;
- number of matching targets remaining in the dungeon.

Success criteria:

- Equal or better winrate than current object-breaking heuristic.
- Lower death rate in Limon and object-loss-heavy scenarios.

### Stage 4: Learn Full Item Use

Purpose: make every item hook visible to the policy layer before draft values are
learned. Draft cannot be trusted until the AI can actually use the items it
picks.

Current learned coverage:

- `en_combat`: 172 item classes routed through `choose_item_activation`;
- `en_survie`: 11 item classes routed through `choose_item_activation`;
- survival labels currently preserve baseline behavior by treating survival
  activation as "use".

Missing item decision surfaces:

- `en_fuite`: 6 item classes;
- `debut_tour`: 26 item classes;
- `fin_tour`: 11 item classes;
- `en_vaincu`: 17 item classes;
- `en_rencontre`: 5 item classes;
- `en_rencontre_event`: 4 item classes;
- `en_subit_dommages`: 5 item classes;
- `en_activated`: 5 item classes;
- `en_mort`: 3 item classes;
- `en_fuite_definitive`: 4 item classes;
- `en_roll`: 6 item classes.

Important distinction:

- Some hooks are real optional item-use decisions, such as spending an item at
  the start of turn, after victory, during flee, or when taking damage.
- Some hooks are passive reactions or mandatory consequences, such as repairing
  after a trigger, medal protection, or score modifiers. These should be routed
  through the rules engine and counted as implemented, but not exposed as model
  actions unless there is a legal "use / skip" choice.

Passive and mandatory item handling:

- A passive item is always applied when its rule condition is true and has no
  player choice. Examples include score modifiers, automatic repairs after a
  matching trigger, medal protection, stat bonuses, and bookkeeping counters.
  These should remain deterministic rule code.
- A mandatory item is an effect that the real player cannot decline once the
  trigger happens. These should also remain deterministic rule code, even if the
  effect can be good or bad.
- A forced trigger with an internal target choice should not be modeled as
  "use / skip". Only the target should be exposed to the policy. For example,
  if an item must repair or discard something, the policy can choose which legal
  object/card is affected, but not whether the trigger exists.
- A random effect is not a model decision unless the player chooses whether to
  activate it before the roll. Once activated, the roll and consequences stay in
  the rules engine.
- A hook is optional only when the player can legally decline the activation and
  preserve the item/state for later. Those hooks should use
  `choose_item_activation`.
- A hook is a target-choice decision when activation is already determined but
  the player can choose a legal object, card, player, discard, repair, replace,
  or copied item. Those hooks need a specific policy method instead of a binary
  activation action.
- The coverage report for this stage must list every item hook as one of:
  `optional_activation`, `passive`, `mandatory`, `target_choice`,
  `discard_choice`, `repair_choice`, `replace_choice`, `roll_choice`, or
  `random_no_decision`.

Tasks:

- Audit every non-combat item hook and classify it as one of:
  - optional activation;
  - forced/passive trigger;
  - target choice;
  - replacement/discard/repair choice;
  - random effect with no player decision.
- Add explicit policy methods only where the player has agency. Likely methods:
  - `choose_item_activation`;
  - `choose_repair_target`;
  - `choose_discard_target`;
  - `choose_replace_target`;
  - `choose_roll_modifier`;
  - `choose_item_target`.
- Keep passive hooks deterministic and rule-owned.
- Add counters per hook so benchmarks can report item-use coverage beyond the
  current aggregate `Use%`.
- Build a full item-use environment that can pause at any optional item decision
  and replay prior actions.
- Start with supervised imitation of the current heuristic behavior.
- Fine-tune only after the classifier proves that mandatory/passive effects are
  not being skipped.
- Add focused regression tests for representative items from every hook group.

Success criteria:

- Every one of the 265 item classes is either policy-routed or explicitly marked
  passive/no-decision.
- No item hook is silently skipped by learned policies.
- Full item-use model matches or beats the combat+survival item model in
  held-out benchmarks.
- Benchmark reports per-hook decision counts and use rates.
- Draft training is not started until this stage has a coverage report.

### Stage 5: Learn Draft Picking

Purpose: replace `choisirObjet(...)` and later `draft_soiree(...)`.

Decision:

- choose one item from the current hand.

Observation:

- hero id and level;
- current picked items;
- current hand item ids and features;
- visible opponent picks;
- pick number;
- player medals and opponent medals in evening mode;
- number of players.

Reward:

- game win for single-round draft;
- evening win for full party mode;
- optional auxiliary reward for round win.

Training approach:

- Start with imitation learning from the current prior/Monte Carlo picker.
- Fine-tune with PPO/self-play.
- Keep epsilon exploration during self-play so under-rated items remain sampled.

Success criteria:

- Better item pick winrate than current `draft.py` priors in held-out seeds.
- Better evening winrate than `party.py` draft heuristic after Stage 6.

### Stage 6: Full Evening Strategy

Purpose: optimize the actual play mode with medals, hero levels, and multiple
rounds.

Decision changes:

- risk tolerance depends on current medals;
- draft values change when ahead or behind;
- death is worse when holding medals;
- survival has long-term value through hero level 2.

Observation additions:

- current round number;
- planned number of rounds;
- medals by player;
- hero levels by player;
- whether the current round is a tiebreaker;
- remaining available hero pool if exposed to the policy.

Reward:

- primary: evening win;
- secondary: medal gain/loss;
- small round-win signal.

Success criteria:

- Statistically significant evening winrate improvement over the current
  `party.py` heuristic.
- No degradation in obvious cases such as protecting medals when ahead.

### Stage 7: Self-Play and Policy League

Purpose: avoid overfitting to the current heuristic AI.

Approach:

- Train against a pool of opponents:
  - current heuristic;
  - random weak baseline;
  - previous model checkpoints;
  - latest model.
- Freeze checkpoints periodically.
- Evaluate on seeds not used for training.

Success criteria:

- New models beat old checkpoints.
- The model remains strong against the original heuristic.
- Performance is stable across 3-player and 4-player games.

## Environment Design

Create a Gymnasium-like API:

```python
obs, info = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action)
```

Likely environments:

- `FleeEnv`: only flee decisions are learned.
- `ReplayEnv`: flee plus replay/pass.
- `BreakObjectEnv`: object sacrifice decisions.
- `DraftEnv`: item picking.
- `DungeonEnv`: complete single dungeon run.
- `PartyEnv`: complete evening.

The first environments can internally delegate most choices to `HeuristicPolicy`.

## Model Input Strategy

Start simple with fixed-size vectors.

Good first representation:

- numeric scalar features normalized to reasonable ranges;
- one-hot hero ids;
- bag/count vectors for item ids;
- bag/count vectors for monster/card ids or monster features;
- aggregate opponent features.

Later, if needed:

- item embeddings;
- card embeddings;
- transformer or attention over variable-length hands/items/deck summaries.

Avoid raw object references, logs, and strings as model inputs.

## Evaluation Protocol

Every trained policy must be compared through repeated simulations:

- same seed schedule for compared policies;
- randomized seats;
- mixed tables when comparing policies;
- separate train and evaluation seed ranges;
- confidence intervals for winrate differences.

Report at minimum:

- games/evenings played;
- win%;
- death%;
- flee%;
- clear%;
- average score placed;
- median score placed;
- medal loss rate in evening mode;
- confidence interval for winrate difference.

## Suggested Dependencies

Current dependencies:

- `numpy`;
- `pandas`;
- `tqdm`.

Likely additions:

- `gymnasium`;
- `stable-baselines3`;
- `torch`;
- `tensorboard`;
- `pytest`.

Do not add these until the first environment wrapper is ready.

## Near-Term Implementation Order

1. Add policy interface and `HeuristicPolicy`.
2. Route `deciderDeFuir` through the policy without changing behavior.
3. Build `FleeEnv`.
4. Add benchmark script comparing heuristic, random, and model policies.
5. Train first PPO flee model.
6. Validate against `bench_fuite.py` style evaluation.
7. Expand to replay/pass.
8. Expand to object break/discard.
9. Expand to full item-use coverage.
10. Expand to draft.
11. Expand to full evening self-play.

## Non-Goals

- Do not train from terminal logs.
- Do not replace the rules engine with ML.
- Do not start with all 265 object hooks as learned actions.
- Do not optimize only final score while ignoring winrate.
- Do not trust training reward without held-out policy benchmarks.

## Definition of Done for AlphaJon

AlphaJon is successful when a trained policy:

- plays full evenings end to end;
- makes draft and in-game decisions through model-backed policies;
- beats the current heuristic AI over a large held-out benchmark;
- provides reproducible training and evaluation scripts;
- leaves the simulator usable for ordinary balance testing.
