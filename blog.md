# introducing iv: my first ai-native software package

Software engineering has rapidly evolved from paying humans to think hard about specific algorithms to having humans orchestrate entire codebases while delegating most algo writing to AI agents. In other words, the human attention has moved to a higher level of abstraction—making design decisions, goal setting, and so on. Most algorithms have been written a thousand times before, and are easily reproduced by AI agents. There is little novelty or complexity in most of these tasks.

As us humans focus on different parts of the process due to AI, we need new tools. Our existing software is largely written for humans—but AI has different strengths and weaknesses and in general makes different classes of mistakes at different rates. I noticed this while developing my WNBA advanced stats model (vorp.app). So I wrote some software for AI agents, which I believe is a new category of thing we’ll all be doing a lot of.

## the problem

Each day, my WNBA stats model gets the injury report for today, game scores for yesterday, and play by play data for yesterday. It folds this in to the current season’s data by updating a file on disk, and it regenerates artifacts such as a now-cast of player and team quality.

I ran into a problem where AI agents were awful about reasoning about the steps here, and subsequently terrible at setting up code to only re-run stale stuff. My build “DAG” had cycles, and minor data changes would frequently cause the entire thing to re-run.

In the old days, I would have built the pipeline myself and thought carefully about ordering and invalidation. The solution would have been correct and bespoke. But now, that feels slow. AI agents can get it mostly right, with a few cycles or duplicated computations that differ slightly. It seems much better to design some AI guard rails so we can identify the few cases it’s missing on.

## meet iv

To address this, I designed invalidator, or iv, which you can find here:
https://github.com/georgeberry/iv

iv is a dependency tracker for data pipelines. 
The whole thing rests on one decision: **there is no index.** A derived file is named
`<part>.<key>.<fp>` — `key` is a digest of the upstreams it was built from, `fp` a digest
of its own bytes. "Is this current?" recomputes the key from the declared upstreams and
the files on disk, then asks whether a file by that name is there.

That sounds like a small implementation detail. It isn't, and it's the first thing I'd
say to anyone building tools for agents to use. A state file is a second place the truth
can live, and a second place is a place an agent can leave inconsistent. Every "the
manifest says this is built but it isn't" bug is a bug you can't have if the filenames
*are* the manifest. The tree can't disagree with the record, because it is the record.

## declare it where a machine can read it

The second decision took me longer to get right, and I got it wrong once first.

To skip a stage you have to know what it reads *before* you run it. My first version read
that off the source with an AST parse, hunting for `iv.reads("...")` calls in the function
body. That works until it doesn't: a selector written as a call argument —
`features(before=season)` — only exists once the call happens. The one thing you must
know first is the one thing a running program can't be asked for.

So the declaration moved into the signature, where it's just a value:

```python
@iv.data("processed/cohorts/", why="a fit per cohort", part="season")
def cohorts(past=iv.before_part("processed/features/", why="every prior season")):
    return past.group_by("player").agg(pl.col("z").mean())
```

`inspect.signature` reads that with nothing executed and no source text needed. `past` is a
walk-forward bound — it selects *files*, so this cohort physically cannot open a later
season. Not filtered after loading. Never opened.

The `why=` is required everywhere. There's nowhere else for that line to live, which is
what stops it going stale.

## it refuses rather than guesses

This is the part that's actually about agents. When an agent gets a pipeline wrong, it
usually doesn't crash — it produces a number that is quietly, confidently stale. So iv
stops the run for anything it can't account for:

- an undeclared read or write inside the data tree
- a second producer for one dataset, unless each names a different partition
- building one stage from inside another
- a read that selects nothing, unless you said it was optional
- a declared write that wrote nothing
- a stray file in a dataset directory

Each of those used to pass silently. They're the reason the pipeline can be trusted to
skip: if the tool can't tell whether something is current, it stops, and you find out in
one line instead of three weeks later in a leaderboard.

## the thing I actually learned

I ported vorp.app's pipeline onto it — 49 stages, every one declared, nothing rebuilt.

Porting it surfaced three bugs, and all three were the same bug. iv had two ways to
answer "what does this stage read": the CLI read it one way, the run read it another.
A key that counted one upstream twice. A status that judged every shard of a dataset
against whichever stage it found first. A read that raised where the key had waved
through. Each was invisible from either side, because neither side could see the other's
answer.

That's the lesson I'd generalise. The guardrail isn't a linter you bolt on. It's making
the wrong state unrepresentable — one route to every fact, and a loud stop when the tool
can't compute one. An agent will cheerfully maintain two things that disagree. It will
not get past a `DeclError`.

iv is a week old and runs exactly one pipeline, so take the usual pinch of salt. But the
daily run is honest now, and when it skips something I believe it.
