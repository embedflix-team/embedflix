"""agent/knowledge_base.py — curated reference chunks for the read_papers()
MCP tool, domain-tagged so each specialist can retrieve only its own area
(the "specialized knowledge base routing" idea from the build-plan doc's
advanced-design note: a debugger node reads debugging docs, a feature node
reads feature-selection docs, etc.).

Role in the system: a fallback / grounding layer alongside web_search, not a
replacement for it. web_search already degrades gracefully to the LLM's own
training knowledge when a search fails -- but if the whole run happens
somewhere without internet access (a real risk: this repo's own cloud
sandbox can't reach Zenodo), every web_search call fails for the entire run,
not just occasionally. This gives specialists something more specific than
pure guesswork to fall back on in that case, and costs nothing to query
(local, no API key) so it's cheap to use even when web_search is working
fine.

Persisted locally via ChromaDB's default embedding function
(all-MiniLM-L6-v2, ONNX, downloaded once on first use, no external API key)
so it works fully offline once the model's cached.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_HERE, "..", "knowledge_db")
_COLLECTION_NAME = "embedflix_knowledge"

# ── Curated chunks ──────────────────────────────────────────────────────────
# domain matches a specialist's node name so it can scope its own queries;
# "general" is dataset/organizer-hint content useful to the supervisor and
# judge for overall context, not tied to one specialist.

_CHUNKS = [
    {
        "id": "bpr-loss",
        "domain": "loss_function_changer",
        "title": "BPR loss (Bayesian Personalized Ranking)",
        "text": """BPR is a pairwise ranking loss: instead of predicting an absolute click
probability per (user, item) like log-loss does, it trains the model to score a user's
positive (clicked/liked) item higher than a sampled negative item. For a user u, a positive
item i, and a sampled negative item j, the loss is:

    L = -log(sigmoid(score(u, i) - score(u, j)))

which is minimized when score(u, i) > score(u, j) by a wide margin. This directly targets
ranking order rather than calibrated probability, which is why it tends to help GAUC (a
ranking-order metric) even when it doesn't change accuracy on the pointwise click-prediction
task at all.

Minimal numpy sketch, given a model that produces score(u, i) via embedding dot product:

    def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
        pos_score = np.sum(user_emb * pos_item_emb, axis=1)
        neg_score = np.sum(user_emb * neg_item_emb, axis=1)
        return -np.mean(np.log(sigmoid(pos_score - neg_score) + 1e-10))

Negative sampling: for each positive (user, item) interaction in a training batch, sample one
item the user did NOT interact with (uniformly at random from the item catalog, or from
items the user saw but didn't click if impression logs are available -- the latter is a
harder, more informative negative and usually converges faster).

Where it tends to help: log-loss trains a well-calibrated click predictor but doesn't
explicitly optimize the relative ordering of items for a given user, which is exactly what
GAUC and nDCG@5 measure. BPR is a standard fix for this mismatch and is well-established in
the recommender-systems literature (Rendle et al., 2009).""",
    },
    {
        "id": "listwise-softmax",
        "domain": "loss_function_changer",
        "title": "Listwise softmax loss for ranking",
        "text": """Listwise softmax treats all items shown to one user in a session as a group and
optimizes the probability mass on the positive item(s) within that group, rather than
comparing to one negative at a time (as BPR does). For a user u with candidate items
{i_1, ..., i_k} and scores {s_1, ..., s_k}, the loss for a positive item i_p is:

    L = -log( exp(s_p) / sum_j exp(s_j) )

i.e. standard cross-entropy softmax over the candidate set, with the positive item as the
target class. This is closer to how nDCG@5 actually evaluates a model (the ranking of a whole
list, not one pairwise comparison), so it tends to help nDCG@5 more directly than BPR does,
at the cost of needing a well-defined "candidate set" per training example (harder to
construct than a single random negative if the data isn't already grouped by session/user).

Where it tends to help: when a user has multiple candidate items in the training signal (not
just one positive and one implicit negative), listwise softmax generally out-performs
pairwise BPR on top-k ranking metrics like nDCG@5, since it optimizes the full ordering
jointly rather than one pair at a time.""",
    },
    {
        "id": "focal-loss",
        "domain": "loss_function_changer",
        "title": "Focal loss for class imbalance",
        "text": """Focal loss modifies standard log-loss to down-weight easy, already-well-classified
negatives and focus training signal on hard examples:

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

where p_t is the model's predicted probability for the true class, and gamma (typically 1-5)
controls how aggressively easy examples are down-weighted. As p_t -> 1 (an easy, correct
prediction), (1 - p_t)^gamma -> 0, so easy examples contribute almost nothing to the
gradient; hard/misclassified examples keep a large gradient.

Relevant here because KuaiRand-Pure is heavily imbalanced (~95% negatives -- most
impressions are not clicked/liked). Plain log-loss on this data spends most of its gradient
on the easy, already-correct negative predictions rather than the harder positive cases that
actually determine ranking quality. Focal loss is a lower-risk change than switching loss
families entirely (BPR/softmax) since it keeps the same pointwise training setup and doesn't
need a negative-sampling or candidate-grouping scheme -- worth trying if BPR/softmax overhaul
is too large a change to land safely in the time remaining.""",
    },
    {
        "id": "din-sequence",
        "domain": "sequence_modeller",
        "title": "Deep Interest Network (DIN) — user history aggregation",
        "text": """DIN's core idea: instead of representing a user with one static embedding, represent
them by an attention-weighted aggregation of their recent interaction history, where the
attention weights depend on the CANDIDATE item being scored -- i.e. the user representation
changes depending on what's being ranked.

Given a user's history of item embeddings [h_1, ..., h_n] and a candidate item embedding c,
compute attention weights and a weighted sum:

    def din_attention(history_embs, candidate_emb):
        # simple additive attention -- history_embs: (n, d), candidate_emb: (d,)
        scores = history_embs @ candidate_emb  # (n,) -- how relevant each past item is to
                                                 # the candidate being scored right now
        weights = softmax(scores)
        user_repr = weights @ history_embs      # (d,) -- weighted history summary
        return user_repr

This user_repr then feeds into the rest of the model (e.g. concatenated with a static user
embedding and the candidate embedding before the final scoring layer) in place of, or
alongside, a static user embedding.

Relevant here because the baseline FM model does not use interaction history at all -- it
only sees a static user_id embedding. Users with rich recent history (many prior
watches/likes) are exactly the case where a static embedding loses information a
history-aware model would capture. Implementation cost is real: this needs per-user
interaction sequences to be constructed from the raw logs first (group by user_id, sort by
timestamp, take last N interactions) before any attention mechanism can be added --
budget time for that data-prep step, not just the model change.""",
    },
    {
        "id": "mtl-recsys",
        "domain": "multitask_trainer",
        "title": "Multi-task learning for recommenders",
        "text": """The idea: train one shared model to predict several correlated signals at once
(e.g. is_click, is_like, is_follow, is_comment, is_forward, and a play_time_ms-derived
target) instead of only the target label (long_view), with a shared representation and
separate small output heads per task:

    shared_repr = shared_encoder(user_features, item_features)
    click_pred = click_head(shared_repr)
    like_pred = like_head(shared_repr)
    ...
    loss = w1*BCE(click_pred, is_click) + w2*BCE(like_pred, is_like) + ... + w_main*BCE(main_pred, long_view)

The auxiliary tasks act as regularizers and inject additional signal into the shared
representation -- particularly valuable when the main label (long_view) is sparse, since the
model can still learn useful structure from the denser auxiliary signals (clicks are far more
common than follows or comments) even on examples where the main label alone would carry
little gradient.

Task-weighting is the main practical risk: naive equal weighting often lets whichever task
has the most examples (click) dominate the shared representation and can hurt the main
task's performance rather than help it. Start with the main task weighted several times
higher than any single auxiliary task, and watch validation primary specifically -- if it
regresses relative to a single-task run, the auxiliary weights are too high, not too low.""",
    },
    {
        "id": "deepfm-dcn",
        "domain": "model_swapper",
        "title": "DeepFM / DCN / xDeepFM — higher-capacity architectures",
        "text": """DeepFM combines the same second-order feature-interaction term the baseline FM
already computes with a parallel deep MLP tower over the same embeddings, then sums both
towers' outputs. DCN (Deep & Cross Network) instead uses explicit cross layers that learn
bounded-degree feature interactions (not just pairwise, like plain FM) alongside a deep
tower. xDeepFM combines both ideas (compressed interaction network + deep tower).

Organizer note (from the starter kit's own README, worth weighing before spending time
here): bigger embeddings (k=8/16/32) barely moved the baseline's score, and the stated reason
is that user_id x video_id crossing already captures most of the available signal --
capacity does not appear to be the bottleneck for this dataset. A higher-capacity
architecture is unlikely to help much unless it's paired with genuinely new input signal
(sequence features, auxiliary tasks) rather than just more parameters over the same features
FM already sees. This is why the organizer's own ranked-priority guess puts this option
below loss-function and sequence-modeling changes -- worth trying only after those show
diminishing returns, not as a first move.""",
    },
    {
        "id": "training-tuning",
        "domain": "training_optimizer",
        "title": "Learning rate, batch size, regularization, early stopping",
        "text": """General guidance for tuning an FM-style model on click/ranking data:

- Learning rate: FM models with embedding tables are sensitive to LR being too high (embedding
  norms blow up, training destabilizes) more often than too low. If a loss-function or
  architecture change causes training to diverge or oscillate, halving the LR before touching
  anything else is usually the fastest fix.
- Batch size: larger batches give more stable gradient estimates for the embedding tables but
  reduce the number of gradient steps per epoch: if increasing batch size, consider scaling LR
  up proportionally (or using a short LR warmup) to compensate.
- Regularization (weight decay, embedding L2): most useful when a model is memorizing rather
  than generalizing -- symptom is validation score peaking early then degrading while training
  loss keeps improving. Not the right lever if validation score is simply low and flat; that's
  usually an underfitting or wrong-signal problem, not an overfitting one.
- Early stopping / patience: stop training when validation primary hasn't improved for N
  epochs (the baseline already does this via its `patience` parameter). If a change to the
  loss function or architecture needs a different training dynamic (e.g. BPR often needs more
  epochs to converge than log-loss since each gradient step only compares one pair), the
  patience value may need to increase alongside it, not stay fixed.

This is a lower-leverage lever than loss-function or sequence-modeling changes for THIS
dataset specifically (see the organizer-hints chunk) -- most useful for stabilizing training
after a more substantive change, not as a first move on its own.""",
    },
    {
        "id": "organizer-hints",
        "domain": "general",
        "title": "Organizer README hints — what's already been tried",
        "text": """From the starter kit's own README (not the public wiki doc) -- things already
tried by the organizers with no meaningful gain over the baseline FM, worth NOT re-testing:

- More static features (13 CWM domain features added) -- came out noise-level versus the
  baseline's 5 fields.
- Bigger embeddings (k=8/16/32) -- barely moved the score.
- Why: the user_id x video_id crossing already captures most of the available signal for
  this task; pure user-side static features contribute close to zero because ranking is
  computed WITHIN one user's candidate set -- a feature that's constant across all of a
  user's candidates cannot change the relative order of those candidates, no matter how
  informative it is in isolation.

Organizer's own ranked guess at where the remaining headroom is (highest to lowest expected
value): (1) loss function -- pointwise to pairwise/listwise, (2) user history / sequence
modeling -- currently completely unused by the baseline, (3) multi-task learning using the
other engagement signals, (4) watch-time censored regression, (5) bigger/different model
architectures -- deprioritized, capacity isn't believed to be the bottleneck, (6) temporal
features / train-test drift, (7) unbiased validation via the randomized-exposure log.""",
    },
    {
        "id": "dataset-facts",
        "domain": "general",
        "title": "KuaiRand-Pure dataset & metric facts",
        "text": """Metrics: GAUC (grouped AUC -- AUC computed per-user then averaged, so it measures
within-user ranking quality rather than being dominated by users with the most interactions)
and nDCG@5 (normalized discounted cumulative gain at rank 5 -- rewards getting truly relevant
items into the TOP 5 specifically, discounted by rank position). primary = mean(GAUC, nDCG@5).

Known reference numbers for this benchmark: baseline (FM) hidden-test primary is 0.5946
(GAUC 0.6610, nDCG@5 0.5282); this repo's local validation baseline is ~0.6016-0.6015. Oracle
ceiling is 0.8645, NOT 1.0 -- because 27.1% of users in this dataset have zero positive
interactions in their candidate set, nDCG for those users is always exactly 0 regardless of
model quality, which caps the achievable average. Random-ranking baseline is 0.4753, so the
gap between random and oracle is the real achievable range to reason about, not the gap to
1.0.

The dataset carries several engagement signals beyond the primary long_view label: is_click,
is_like, is_follow, is_comment, is_forward, and play_time_ms -- these are the auxiliary
signals a multi-task approach would use, and they vary widely in density (clicks are common,
follows/comments/forwards are rare), which is the main practical complication for any
multi-task weighting scheme.""",
    },
]

_client = None
_collection = None
_chroma_unavailable = False  # sticky -- once embedding/chromadb fails, stop retrying every call


def _get_collection():
    """Lazily creates/opens the persistent Chroma collection and populates it
    on first use. Safe to call repeatedly -- only writes once. Raises on
    failure (e.g. the embedding model's first-time download being blocked by
    network restrictions) -- callers must catch and fall back, same
    contract as web_search's own try/except around the Tavily call."""
    global _client, _collection
    if _collection is not None:
        return _collection
    import chromadb

    os.makedirs(_DB_PATH, exist_ok=True)
    _client = chromadb.PersistentClient(path=_DB_PATH)
    _collection = _client.get_or_create_collection(_COLLECTION_NAME)
    if _collection.count() == 0:
        _collection.add(
            ids=[c["id"] for c in _CHUNKS],
            documents=[c["text"] for c in _CHUNKS],
            metadatas=[{"domain": c["domain"], "title": c["title"]} for c in _CHUNKS],
        )
    return _collection


def _keyword_fallback(query: str, domain: str, n_results: int) -> str:
    """Plain substring/overlap scoring over the same _CHUNKS, no embedding
    model needed. Used when ChromaDB (or its embedding model's first-time
    download) isn't available -- e.g. this repo's own cloud sandbox can't
    reach the download CDN, same restriction that blocks Zenodo. Cruder than
    real vector search, but keeps read_papers() from being just as
    network-dependent as web_search on a machine where neither has internet."""
    pool = [c for c in _CHUNKS if domain is None or c["domain"] == domain] or _CHUNKS
    query_words = set(query.lower().split())
    scored = []
    for c in pool:
        text_lower = (c["title"] + " " + c["text"]).lower()
        score = sum(1 for w in query_words if w in text_lower)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for score, c in scored[:n_results]]
    if not top:
        return "No matching reference material found in the local knowledge base."
    return "\n\n---\n\n".join(f"REFERENCE: {c['title']}\n{c['text']}" for c in top)


def query_knowledge_base(query: str, domain: str = None, n_results: int = 3) -> str:
    """Returns the top n_results chunks (optionally scoped to one domain) as
    a single formatted string, ready to drop into a specialist's prompt --
    same "REFERENCE: <title>\\n<text>" shape as web_search's "SOURCE:"/
    "CONCEPT:" formatting, joined the same way, for a consistent prompt feel
    regardless of which tool supplied the grounding.

    Falls back to plain keyword matching (no embedding model, no network)
    if ChromaDB/its embedding model isn't available -- never raises.
    """
    global _chroma_unavailable
    if not _chroma_unavailable:
        try:
            collection = _get_collection()
            where = {"domain": domain} if domain else None
            capped_n = max(min(n_results, len(_CHUNKS)), 1)
            results = collection.query(query_texts=[query], n_results=capped_n, where=where)
            docs = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            if not docs:
                return "No matching reference material found in the local knowledge base."
            parts = [f"REFERENCE: {meta.get('title', 'untitled')}\n{doc}" for doc, meta in zip(docs, metas)]
            return "\n\n---\n\n".join(parts)
        except Exception:
            # Embedding model download blocked, chromadb misconfigured, etc.
            # -- degrade once, not on every future call in this run.
            _chroma_unavailable = True

    return _keyword_fallback(query, domain, n_results)
