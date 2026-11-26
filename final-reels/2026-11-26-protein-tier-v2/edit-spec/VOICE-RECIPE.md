# Liam voice continuity

Use local Kokoro v1.0 82M and kokoro-onnx 0.6.1. The approved comparison sample text was: “Inconsistent routines. Poor sleep. Frequent restaurant meals. No consistent calorie deficit.” This produces 103 tokens and uses am_liam style row 102.

For new item announcements, retain that exact approved style vector for the whole context. Generate the names together, then split at aligned low-energy boundaries. Avoid isolated 1–3-word synthesis: it selects different short-context voice conditioning and sounded unlike the approved sample.

Match speed 1.02, en-us, sentence pause .45, clause pause .12, continuous=False, trim=True, and narrator loudness -18 LUFS / true peak -1.5 / LRA9. No pitch shifting, voice blending, paid API or subscription. Only item names; presenter keeps all spoken tiers and explanations.

The reproduction test using this conditioning produced a byte-identical copy of the approved raw Liam sample. See liam-reference-verification.json for SHA256 evidence.
