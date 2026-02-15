# Summary Statistics

This section presents the summary statistics for the three datasets used in our project: the Twitter Customer Support dataset, the Reddit (DSTC8) Corpus, and the OpenAssistant Conversations dataset. For each, we report the distribution of the target variable, missing-value rates, and a non-trivial visualization.

---

## 1. Twitter Customer Support Dataset

**Size:** ~2.8 million tweets | **Columns:** 7 | **Time Period:** 2017–2018

### Target Variable — Inbound vs Outbound Tweets

The target signal is whether a tweet is **inbound** (customer → brand) or **outbound** (brand → customer). The dataset splits approximately **58% inbound / 42% outbound**. Among customer (inbound) tweets, a significant fraction never receive a brand reply — the `response_tweet_id` field is missing for these unanswered messages. Conversation thread lengths (built by tracing reply chains) have a median of 2–4 turns, with a long right tail; the 90th percentile is around 6 turns.

### Missing Values

| Column | Missing % |
|--------|-----------|
| `in_response_to_tweet_id` | ~61% (root tweets have no parent) |
| `response_tweet_id` | ~35% (customer tweets with no brand reply) |
| Other columns | 0% |

*[Insert Figure: Missing Value Rates bar chart — Twitter dataset]*

### Non-Trivial Visualization — Brand Response Rate vs Volume

A scatter plot of each brand's **inbound tweet volume** (log scale) against its **response rate** reveals that high-volume brands do *not* uniformly maintain high response rates. There is clear dispersion: some large brands reply to >90% of customer tweets, while others with comparable volume fall below 50%. This tension between scale and responsiveness directly motivates our RL approach — intelligently routing tickets (bot vs. human) can improve response rates at the same cost budget.

*[Insert Figure: Brand Response Rate vs Customer Tweet Volume scatter plot]*

---

## 2. Reddit (DSTC8) Corpus

**Size:** ~5.1 M conversations (training split) + validation splits | **Domains:** ~1,000 subreddits | **Time Period:** 2017–2018

### Target Variable — Conversation Length (Number of Turns)

Number of turns serves as a proxy for **time-to-resolution** and **cost-to-serve**. The distribution is heavily right-skewed: the median is ~4 turns, the 90th percentile is ~8 turns, and the 95th percentile reaches ~11. Most conversations are short, but a meaningful tail of complex, multi-turn threads exists — exactly the cases where an RL agent's routing decision has the highest expected cost impact.

*[Insert Figure: Histogram and CDF of conversation length (number of turns)]*

### Missing Values

The dataset has **no null values** in any field. However, the `bot_id` and `user_id` columns are populated with empty strings across effectively all rows — these identifiers were not collected in the DSTC8 release. All substantive fields (`domain`, `turns`, `task_id`) are fully populated.

### Non-Trivial Visualization — Domain Complexity Map

A bubble chart plotting each subreddit's **conversation volume** (x-axis, log scale) against **mean turns per conversation** (y-axis), with bubble size encoding average opening-message length and color encoding the 90th-percentile turn count, reveals significant variation in conversation complexity across domains. Tech-support-style subreddits (e.g., networking, programming) consistently produce longer, more complex threads — analogous to harder customer issues in our SaaS scenario. This heterogeneity validates using domain/topic as a contextual feature in our bandit model.

*[Insert Figure: Domain Complexity bubble chart — volume vs mean turns, colored by 90th %ile]*

---

## 3. OpenAssistant Conversations Dataset

**Size:** 161,443 messages across 66,497 conversation trees | **Languages:** 35+ | **Time Period:** 2021–2023

### Target Variable — Quality Ratings & Rank

The primary quality signal is **rank** (human-annotated, lower = better) and a continuous **quality score** (0–1 scale) extracted from annotator labels. The rank distribution is heavily right-skewed — most ranked assistant responses receive rank 0 (best), indicating strong annotator agreement on what constitutes a good response. The median quality score is ~0.75. Approximately 60% of messages lack a rank annotation (not yet evaluated), and ~95% have no `model_name` (human-written, not synthetic).

*[Insert Figure: Rank distribution bar chart + Quality score histogram with median line]*

### Missing Values

| Column | Missing % |
|--------|-----------|
| `model_name` | ~95% (human-written messages) |
| `rank` | ~60% (messages not yet ranked) |
| `parent_id` | ~41% (root messages / conversation starters) |
| `detoxify` | ~3% |
| Other columns | <1% |

*[Insert Figure: Missing Value Rates bar chart — OpenAssistant dataset]*

### Non-Trivial Visualization — Quality vs Toxicity by Role

Scatter plots of **quality score** against **toxicity score** (from the Detoxify model) for prompter and assistant messages reveal near-zero correlation (Pearson *r* ≈ 0.0 for both roles). High-quality responses are *not* inherently more toxic, and low-toxicity messages span the full quality range. This decoupling is critical for our SaaS bot: we can optimize the RL agent for response quality without introducing safety risks.

*[Insert Figure: Quality vs Toxicity scatter plots (prompter and assistant side by side)]*

---

## Cross-Dataset Summary

| Metric | Twitter | Reddit (DSTC8) | OpenAssistant |
|--------|---------|----------------|---------------|
| Total records | ~2.8 M tweets | ~5.1 M conversations | ~161 K messages |
| Target variable | Inbound/outbound + response rate | Number of turns (resolution proxy) | Rank + quality score |
| Median target | 58% inbound; 2–4 turn threads | 4 turns | Rank 0 (best); quality 0.75 |
| Key missing field | `response_tweet_id` (35%) | None (empty `bot_id`/`user_id`) | `rank` (60%), `model_name` (95%) |
| Non-trivial insight | Scale ≠ responsiveness | Domain drives complexity | Quality–toxicity decoupled |
