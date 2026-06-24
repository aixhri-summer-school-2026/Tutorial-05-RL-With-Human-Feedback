# MILE Concepts: Intervention Learning at a Glance

Welcome! This is a 5-minute primer on the core ideas behind MILE (Model-based Intervention Learning). You don't need to know the math yet—just the intuition. Ready?

## What is intervention learning?

Imagine teaching a robot by watching it work. Most of the time you let it act on its own. But when you see it's *about to fail*—say, reaching toward an empty spot instead of a cube—you step in and take over for a moment. MILE learns from *both* of these moments: the times you intervened (correcting the robot) and the times you didn't (trusting the robot to figure it out). This "learning from human supervision *and* non-supervision" is the key idea.

## The intervention flag: ν (nu)

Every timestep has a binary decision: did the human intervene or not?

- **ν = 1**: the human took over. The human is giving the robot an example of the correct action.
- **ν = 0**: the robot acted autonomously. The human trusted it and let it continue.

That's it. ν is just a binary label on each step of data: "human acted here" or "human didn't act here."

## The intervention model: p(ν=1|s)

MILE doesn't know in advance when you (the human) will intervene. So it learns to *predict* when a human would intervene, given just the current state s.

This is the **intervention model**: p(ν=1|s) — the probability that you'd step in right now.

### How does it predict this?

It uses a **probit model**, which compares the robot's current action plan against what you (the human) think it should do. The model says: "if the robot and human agree on the action, the robot probably doesn't need help (low ν). If they disagree a lot, you probably will intervene (high ν)."

### COST_LOOKUP: tuning when you ask for help

The intervention decision isn't free. Asking for human help has a cost (your time, attention). So the model includes a **cost parameter** for each task—this is where `COST_LOOKUP` comes in.

For example:
- `peg-insert-side-v2`: cost = 75 (fairly cheap to help; the human is trigger-happy)
- `Franka-Stack-Sim-v0`: cost = 2 (helping is very cheap in sim)

A lower cost means the robot asks for help sooner. A higher cost means the robot tries harder before asking. This lets you tune the method to match how humans *actually* intervene.

## Joint training: the policy π_θ and mental model π̃_ξ

MILE trains *two* neural networks together, not just one:

1. **The policy π_θ** — this is what the robot executes. It takes the current observation and outputs an action. This is what you care about at test time.

2. **The mental model π̃_ξ** — this is what the robot thinks *you* (the human) believe it will do. It's the robot's model of your expectations. During training, it learns from the interventions you gave it. At test time, it helps the intervention model decide when to ask for help.

Why train them together? Because they constrain each other. If the policy starts doing something clearly wrong, the mental model (trained on your interventions) will flag it, and the policy gets corrected. If the robot and human actions align, they reinforce each other.

## The MILE loss: BCE + Gaussian NLL

Now for the math (in plain terms):

The loss has two parts:

### Part 1: Binary Cross-Entropy (BCE) on ν

This term asks: "did I predict the right intervention flag?"

For each step, we predict: "will the human intervene?" (a probability). We compare it against the ground truth ν ∈ {0, 1}. We use BCE to penalize wrong guesses.

### Part 2: Gaussian NLL on ν=1 steps (only when human intervened)

This term asks: "when the human did intervene, did we predict the right action they'd take?"

- Ignore all ν=0 steps (the robot handled those).
- For ν=1 steps, the human gave us their action. We model your action as a Gaussian (normal distribution) with mean μ and standard deviation σ.
- We compute the log-likelihood: "how probable is the human's actual action under our Gaussian model?"
- We minimize the *negative* log-likelihood (maximizing likelihood = minimizing loss).

### Putting them together

```
Total Loss = λ₁ × (Gaussian NLL) + λ₂ × (BCE on ν)
```

The λ weights let you balance: "should we focus more on predicting when humans intervene, or on getting the right action once they do?" Usually, both matter equally, so λ₁ = λ₂ = 1.

## The 4-tier ladder: where MILE runs

You'll implement the loss once, then watch it train on four different tasks:

| Tier | Task | Backend | You control the human? | Point |
|---|---|---|---|---|
| **Tier 1** | Peg-Insert (MetaWorld) | MetaWorld simulator | No—automated synthetic expert | Validate the loss works (clean signal from sim) |
| **Tier 2** | Franka Block Stacking (fake) | Kinematic fake backend | No—headless mediocre policy | Same loss, new task; no ROS needed |
| **Tier 3** | Franka Block Stacking (sim) | MuJoCo multipanda sim | **Yes—you!** keyboard teleop | Your interventions train your policy; watch improvement |
| **Tier 4** | Franka Block Stacking (real) | Real FR3 robot | **Yes—you!** keyboard (or Vive) | Sim→real payoff on live hardware |

All four tiers run in one Docker image and use the *same* loss you implement.

## Next step

You've got the concepts. Now it's time to implement the loss and see it work.

→ **[02-loss-exercise.md](02-loss-exercise.md)**: Let's code it up.
